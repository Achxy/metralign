"""Reproducible adapters around established OpenCV registration methods.

These baselines intentionally do not call Metralign's representations,
transform estimator, ambiguity rule, or subpixel refinement.  They adapt
general-purpose OpenCV primitives to the known 10x reference/search sampling
relationship so every method returns a reference-center coordinate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import time

import cv2
import numpy as np

try:
    from skimage.feature import match_template as skimage_match_template
    from skimage.registration import phase_cross_correlation
    from skimage.transform import resize as skimage_resize
except ImportError:  # pragma: no cover - exercised in core-only installations
    skimage_match_template = None
    phase_cross_correlation = None
    skimage_resize = None


METHODS = (
    "opencv_template",
    "opencv_grid_template",
    "opencv_template_phase",
    "opencv_ecc_affine",
    "opencv_sift_ransac",
    "skimage_template_phase",
)


@dataclass(frozen=True)
class BaselineConfig:
    nominal_scale: float = 0.1
    scale_range: float = 0.006
    scale_steps: int = 5
    rotation_range: float = 3.0
    rotation_steps: int = 5
    ecc_downsample: float = 0.5


@dataclass(frozen=True)
class BaselineResult:
    method: str
    x: float | None
    y: float | None
    status: str
    score: float | None
    runtime_ms: float
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _float32(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("baseline inputs must be finite grayscale images")
    return values


def _unit_float(image: np.ndarray) -> np.ndarray:
    values = _float32(image)
    low, high = np.percentile(values, (1.0, 99.0))
    if float(high - low) <= 1e-8:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(_unit_float(image) * 255.0).astype(np.uint8)


def _template(reference: np.ndarray, scale: float, rotation: float = 0.0) -> np.ndarray:
    reference = _float32(reference)
    height = max(8, int(round(reference.shape[0] * scale)))
    width = max(8, int(round(reference.shape[1] * scale)))
    resized = cv2.resize(reference, (width, height), interpolation=cv2.INTER_AREA)
    if abs(rotation) <= 1e-12:
        return resized
    matrix = cv2.getRotationMatrix2D(
        ((width - 1.0) / 2.0, (height - 1.0) / 2.0), rotation, 1.0
    )
    return cv2.warpAffine(
        resized,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _match_template(search: np.ndarray, template: np.ndarray) -> tuple[int, int, float]:
    search = _float32(search)
    template = _float32(template)
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        raise ValueError("template is larger than the search image")
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    return int(location[0]), int(location[1]), float(score)


def _center(origin_x: float, origin_y: float, template: np.ndarray) -> tuple[float, float]:
    return (
        float(origin_x + (template.shape[1] - 1.0) / 2.0),
        float(origin_y + (template.shape[0] - 1.0) / 2.0),
    )


def _resolved(
    method: str,
    start_ns: int,
    x: float,
    y: float,
    score: float | None,
    **diagnostics: object,
) -> BaselineResult:
    return BaselineResult(
        method=method,
        x=float(x),
        y=float(y),
        status="resolved",
        score=None if score is None else float(score),
        runtime_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        diagnostics=diagnostics,
    )


def _unresolved(method: str, start_ns: int, reason: str, **diagnostics: object) -> BaselineResult:
    return BaselineResult(
        method=method,
        x=None,
        y=None,
        status="unresolved",
        score=None,
        runtime_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        diagnostics={"reason": reason, **diagnostics},
    )


def _opencv_template(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    start = time.perf_counter_ns()
    template = _template(reference, cfg.nominal_scale)
    origin_x, origin_y, score = _match_template(search, template)
    x, y = _center(origin_x, origin_y, template)
    return _resolved(
        "opencv_template",
        start,
        x,
        y,
        score,
        selected_scale=cfg.nominal_scale,
        selected_rotation_deg=0.0,
    )


def _opencv_grid_template(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    start = time.perf_counter_ns()
    scales = (
        np.asarray([cfg.nominal_scale])
        if cfg.scale_steps == 1
        else np.linspace(
            cfg.nominal_scale - cfg.scale_range,
            cfg.nominal_scale + cfg.scale_range,
            cfg.scale_steps,
        )
    )
    rotations = (
        np.asarray([0.0])
        if cfg.rotation_steps == 1
        else np.linspace(-cfg.rotation_range, cfg.rotation_range, cfg.rotation_steps)
    )
    best: tuple[float, float, float, int, int, np.ndarray] | None = None
    for scale, rotation in itertools.product(scales, rotations):
        template = _template(reference, float(scale), float(rotation))
        origin_x, origin_y, score = _match_template(search, template)
        trial = (score, -abs(float(scale) - cfg.nominal_scale), -abs(float(rotation)))
        if best is None or trial > best[:3]:
            best = (score, trial[1], trial[2], origin_x, origin_y, template)
            selected_scale = float(scale)
            selected_rotation = float(rotation)
    assert best is not None
    score, _, _, origin_x, origin_y, template = best
    x, y = _center(origin_x, origin_y, template)
    return _resolved(
        "opencv_grid_template",
        start,
        x,
        y,
        score,
        selected_scale=selected_scale,
        selected_rotation_deg=selected_rotation,
        evaluated_transforms=int(cfg.scale_steps * cfg.rotation_steps),
    )


def _opencv_template_phase(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    start = time.perf_counter_ns()
    template = _template(reference, cfg.nominal_scale)
    origin_x, origin_y, template_score = _match_template(search, template)
    crop = _float32(search)[
        origin_y : origin_y + template.shape[0],
        origin_x : origin_x + template.shape[1],
    ]
    window = cv2.createHanningWindow(
        (template.shape[1], template.shape[0]), cv2.CV_32F
    )
    (shift_x, shift_y), response = cv2.phaseCorrelate(
        _float32(template), crop, window
    )
    if not all(math.isfinite(value) for value in (shift_x, shift_y, response)):
        return _unresolved(
            "opencv_template_phase", start, "nonfinite_phase_solution"
        )
    # The phase shift is the offset of the extracted crop relative to the
    # template, so it is added to the coarse crop origin.
    x, y = _center(origin_x + shift_x, origin_y + shift_y, template)
    return _resolved(
        "opencv_template_phase",
        start,
        x,
        y,
        float(response),
        coarse_template_score=template_score,
        phase_shift=[float(shift_x), float(shift_y)],
        selected_scale=cfg.nominal_scale,
        selected_rotation_deg=0.0,
    )


def _opencv_ecc_affine(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    start = time.perf_counter_ns()
    # A general-purpose intensity optimizer needs a translation initialization;
    # use OpenCV's own normalized template matcher, not a Metralign prediction.
    template = _template(reference, cfg.nominal_scale)
    origin_x, origin_y, coarse_score = _match_template(search, template)
    factor = float(cfg.ecc_downsample)
    ref_small = cv2.resize(
        _unit_float(reference), None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )
    search_small = cv2.resize(
        _unit_float(search), None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
    )
    warp = np.array(
        [
            [cfg.nominal_scale, 0.0, origin_x * factor],
            [0.0, cfg.nominal_scale, origin_y * factor],
        ],
        dtype=np.float32,
    )
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        100,
        1e-6,
    )
    try:
        ecc, warp = cv2.findTransformECC(
            ref_small,
            search_small,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error as exc:
        return _unresolved(
            "opencv_ecc_affine",
            start,
            "optimizer_failed",
            coarse_template_score=coarse_score,
            opencv_error=str(exc).splitlines()[0],
        )
    reference_center = np.array(
        [(ref_small.shape[1] - 1.0) / 2.0, (ref_small.shape[0] - 1.0) / 2.0, 1.0],
        dtype=np.float64,
    )
    predicted_small = np.asarray(warp, dtype=np.float64) @ reference_center
    x, y = (predicted_small / factor).tolist()
    linear = np.asarray(warp[:, :2], dtype=np.float64)
    equivalent_scale = math.sqrt(abs(float(np.linalg.det(linear))))
    if (
        not all(math.isfinite(value) for value in (ecc, x, y, equivalent_scale))
        or not (0.05 <= equivalent_scale <= 0.20)
        or not (0.0 <= x <= search.shape[1] - 1.0)
        or not (0.0 <= y <= search.shape[0] - 1.0)
    ):
        return _unresolved(
            "opencv_ecc_affine",
            start,
            "implausible_affine_solution",
            coarse_template_score=coarse_score,
            equivalent_scale=equivalent_scale,
            warp=np.asarray(warp).tolist(),
        )
    return _resolved(
        "opencv_ecc_affine",
        start,
        x,
        y,
        float(ecc),
        coarse_template_score=coarse_score,
        equivalent_scale=equivalent_scale,
        warp=np.asarray(warp).tolist(),
    )


def _opencv_sift_ransac(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    del cfg
    start = time.perf_counter_ns()
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
    keypoints_ref, descriptors_ref = sift.detectAndCompute(_uint8(reference), None)
    keypoints_search, descriptors_search = sift.detectAndCompute(_uint8(search), None)
    if descriptors_ref is None or descriptors_search is None:
        return _unresolved(
            "opencv_sift_ransac",
            start,
            "no_descriptors",
            reference_keypoints=len(keypoints_ref),
            search_keypoints=len(keypoints_search),
        )
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(descriptors_ref, descriptors_search, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(matches) < 4:
        return _unresolved(
            "opencv_sift_ransac",
            start,
            "insufficient_ratio_matches",
            reference_keypoints=len(keypoints_ref),
            search_keypoints=len(keypoints_search),
            ratio_matches=len(matches),
        )
    source = np.float32([keypoints_ref[match.queryIdx].pt for match in matches])
    destination = np.float32([keypoints_search[match.trainIdx].pt for match in matches])
    matrix, mask = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    inliers = int(np.count_nonzero(mask)) if mask is not None else 0
    if matrix is None or inliers < 4:
        return _unresolved(
            "opencv_sift_ransac",
            start,
            "ransac_failed",
            reference_keypoints=len(keypoints_ref),
            search_keypoints=len(keypoints_search),
            ratio_matches=len(matches),
            inliers=inliers,
        )
    linear = np.asarray(matrix[:, :2], dtype=np.float64)
    equivalent_scale = math.sqrt(abs(float(np.linalg.det(linear))))
    reference_center = np.array(
        [(reference.shape[1] - 1.0) / 2.0, (reference.shape[0] - 1.0) / 2.0, 1.0],
        dtype=np.float64,
    )
    x, y = (np.asarray(matrix, dtype=np.float64) @ reference_center).tolist()
    if (
        not all(math.isfinite(value) for value in (x, y, equivalent_scale))
        or not (0.05 <= equivalent_scale <= 0.20)
        or not (0.0 <= x <= search.shape[1] - 1.0)
        or not (0.0 <= y <= search.shape[0] - 1.0)
    ):
        return _unresolved(
            "opencv_sift_ransac",
            start,
            "implausible_similarity_solution",
            ratio_matches=len(matches),
            inliers=inliers,
            equivalent_scale=equivalent_scale,
            matrix=np.asarray(matrix).tolist(),
        )
    return _resolved(
        "opencv_sift_ransac",
        start,
        x,
        y,
        inliers / len(matches),
        reference_keypoints=len(keypoints_ref),
        search_keypoints=len(keypoints_search),
        ratio_matches=len(matches),
        inliers=inliers,
        equivalent_scale=equivalent_scale,
        matrix=np.asarray(matrix).tolist(),
    )


def _skimage_template_phase(
    reference: np.ndarray, search: np.ndarray, cfg: BaselineConfig
) -> BaselineResult:
    start = time.perf_counter_ns()
    if (
        skimage_match_template is None
        or phase_cross_correlation is None
        or skimage_resize is None
    ):
        raise RuntimeError(
            "skimage_template_phase requires the optional comparison dependencies"
        )
    reference_float = _unit_float(reference)
    search_float = _unit_float(search)
    target_shape = (
        max(8, int(round(reference.shape[0] * cfg.nominal_scale))),
        max(8, int(round(reference.shape[1] * cfg.nominal_scale))),
    )
    template = skimage_resize(
        reference_float,
        target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    scores = skimage_match_template(search_float, template, pad_input=False)
    origin_y, origin_x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    coarse_score = float(scores[origin_y, origin_x])
    crop = search_float[
        origin_y : origin_y + template.shape[0],
        origin_x : origin_x + template.shape[1],
    ]
    shift, phase_error, phase_difference = phase_cross_correlation(
        template,
        crop,
        upsample_factor=20,
        normalization=None,
    )
    # scikit-image reports the shift applied to the moving crop.  Negating it
    # converts the result to a correction of the coarse crop origin.
    shift_y, shift_x = map(float, shift)
    values = (shift_x, shift_y, float(phase_error), float(phase_difference))
    if not all(math.isfinite(value) for value in values):
        return _unresolved(
            "skimage_template_phase",
            start,
            "nonfinite_phase_solution",
            coarse_template_score=coarse_score,
        )
    x, y = _center(origin_x - shift_x, origin_y - shift_y, template)
    return _resolved(
        "skimage_template_phase",
        start,
        x,
        y,
        1.0 - float(phase_error),
        coarse_template_score=coarse_score,
        moving_to_reference_shift=[shift_x, shift_y],
        phase_error=float(phase_error),
        phase_difference=float(phase_difference),
        selected_scale=cfg.nominal_scale,
        selected_rotation_deg=0.0,
    )


def run_baseline(
    method: str,
    reference: np.ndarray,
    search: np.ndarray,
    cfg: BaselineConfig | None = None,
) -> BaselineResult:
    """Run one named external baseline with deterministic OpenCV randomness."""
    if method not in METHODS:
        raise ValueError(f"unknown external baseline: {method}")
    cfg = cfg or BaselineConfig()
    if (
        cfg.nominal_scale <= 0
        or cfg.scale_range < 0
        or cfg.scale_range >= cfg.nominal_scale
    ):
        raise ValueError("invalid scale configuration")
    if cfg.scale_steps < 1 or cfg.rotation_steps < 1 or cfg.rotation_range < 0:
        raise ValueError("invalid transform grid configuration")
    if not (0 < cfg.ecc_downsample <= 1):
        raise ValueError("ecc_downsample must be in (0, 1]")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    implementation = {
        "opencv_template": _opencv_template,
        "opencv_grid_template": _opencv_grid_template,
        "opencv_template_phase": _opencv_template_phase,
        "opencv_ecc_affine": _opencv_ecc_affine,
        "opencv_sift_ransac": _opencv_sift_ransac,
        "skimage_template_phase": _skimage_template_phase,
    }[method]
    return implementation(reference, search, cfg)
