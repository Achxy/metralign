"""Optional adapter for the pinned official XFeat implementation.

The module deliberately does not import PyTorch or XFeat at import time.  Core
Metralign installations therefore retain their small dependency set; the
official upstream source and its optional environment are supplied only to the
separate comparison script.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
import io
from pathlib import Path
import sys
import time
from contextlib import redirect_stdout
from typing import Protocol

import cv2
import numpy as np


OFFICIAL_COMMIT = "e92685f57f8318b18725c5c8c0bd28c7fe188d9a"
OFFICIAL_CODE_BUNDLE_SHA256 = (
    "3ea50cd28a4f753efe7d296fabbdf067bf060c1bd79d7f2fb38a545abd4596ca"
)
OFFICIAL_WEIGHTS_SHA256 = (
    "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
)
OFFICIAL_CODE_FILES = (
    "modules/xfeat.py",
    "modules/model.py",
    "modules/interpolator.py",
)


class XFeatStarMatcher(Protocol):
    """Small structural type for the one upstream API used by this adapter."""

    dev: object

    def match_xfeat_star(
        self, reference: np.ndarray, search: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class XFeatBaselineConfig:
    """Outcome-independent settings copied from the locked protocol."""

    nominal_scale: float = 0.1
    top_k: int = 8000
    ransac_reprojection_threshold_px: float = 3.5
    ransac_max_iterations: int = 1000
    ransac_confidence: float = 0.999
    minimum_matches: int = 4
    minimum_inliers: int = 4
    opencv_rng_seed: int = 0


@dataclass(frozen=True)
class XFeatBaselineResult:
    method: str
    x: float | None
    y: float | None
    status: str
    runtime_ms: float
    diagnostics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialXFeatIdentity:
    commit: str
    code_bundle_sha256: str
    weights_sha256: str
    source_root: str
    weights_file: str
    model_load_ms: float
    upstream_stdout: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_code_bundle_sha256(source_root: Path) -> str:
    """Fingerprint exactly the pinned upstream files used for inference."""

    root = source_root.resolve()
    digest = sha256()
    for relative in OFFICIAL_CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"official XFeat source is missing {relative}")
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_official_xfeat(source_root: Path, weights_path: Path) -> dict[str, str]:
    """Fail closed when upstream code or checkpoint differs from the lock."""

    code_digest = official_code_bundle_sha256(source_root)
    if code_digest != OFFICIAL_CODE_BUNDLE_SHA256:
        raise ValueError(
            "official XFeat inference source does not match pinned commit "
            f"{OFFICIAL_COMMIT}: {code_digest}"
        )
    weights = weights_path.resolve()
    if not weights.is_file():
        raise ValueError(f"official XFeat checkpoint does not exist: {weights}")
    weights_digest = _sha256_file(weights)
    if weights_digest != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError(
            "official XFeat checkpoint SHA-256 mismatch: "
            f"{weights_digest}"
        )
    return {
        "commit": OFFICIAL_COMMIT,
        "code_bundle_sha256": code_digest,
        "weights_sha256": weights_digest,
    }


def load_official_xfeat(
    source_root: Path,
    weights_path: Path,
    *,
    top_k: int = 8000,
) -> tuple[XFeatStarMatcher, OfficialXFeatIdentity]:
    """Load the verified upstream class without vendoring or patching it."""

    identity = verify_official_xfeat(source_root, weights_path)
    root = source_root.resolve()
    existing = sys.modules.get("modules")
    if existing is not None:
        existing_file = Path(str(getattr(existing, "__file__", ""))).resolve()
        if not existing_file.is_relative_to(root):
            raise RuntimeError(
                "a different top-level 'modules' package is already imported; "
                "run the XFeat benchmark in its isolated process"
            )

    inserted = False
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        inserted = True
    start = time.perf_counter_ns()
    captured = io.StringIO()
    try:
        module = importlib.import_module("modules.xfeat")
        with redirect_stdout(captured):
            model = module.XFeat(weights=str(weights_path.resolve()), top_k=top_k)
    finally:
        if inserted:
            sys.path.remove(str(root))
    model_load_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    if str(model.dev) != "cpu":
        raise RuntimeError(
            f"locked XFeat protocol requires CPU, but upstream selected {model.dev}"
        )
    record = OfficialXFeatIdentity(
        commit=identity["commit"],
        code_bundle_sha256=identity["code_bundle_sha256"],
        weights_sha256=identity["weights_sha256"],
        source_root=str(root),
        weights_file=str(weights_path.resolve()),
        model_load_ms=float(model_load_ms),
        upstream_stdout=captured.getvalue().strip(),
    )
    return model, record


def _three_channel_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError("XFeat baseline inputs must be two-dimensional grayscale images")
    if not np.all(np.isfinite(values)):
        raise ValueError("XFeat baseline inputs must contain only finite values")
    if values.dtype == np.uint8:
        gray = values
    else:
        gray = np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def prepare_xfeat_inputs(
    reference: np.ndarray,
    search: np.ndarray,
    nominal_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the task's known sampling-ratio adaptation."""

    if not 0.0 < nominal_scale <= 1.0:
        raise ValueError("nominal_scale must lie in (0, 1]")
    reference_rgb = _three_channel_uint8(reference)
    search_rgb = _three_channel_uint8(search)
    height = max(1, int(round(reference_rgb.shape[0] * nominal_scale)))
    width = max(1, int(round(reference_rgb.shape[1] * nominal_scale)))
    if min(height, width) < 64:
        raise ValueError(
            "nominally resampled reference is too small for XFeat* dual-scale inference"
        )
    reference_nominal = cv2.resize(
        reference_rgb,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return reference_nominal, search_rgb


def _unresolved(
    start_ns: int,
    reason: str,
    **diagnostics: object,
) -> XFeatBaselineResult:
    return XFeatBaselineResult(
        method="xfeat_official_star_homography",
        x=None,
        y=None,
        status="unresolved",
        runtime_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
        diagnostics={"reason": reason, **diagnostics},
    )


def estimate_reference_center(
    reference_points: np.ndarray,
    search_points: np.ndarray,
    reference_shape: tuple[int, int],
    config: XFeatBaselineConfig,
    *,
    start_ns: int | None = None,
) -> XFeatBaselineResult:
    """Estimate a homography exactly as in the pinned official notebook."""

    start = time.perf_counter_ns() if start_ns is None else start_ns
    points0 = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    points1 = np.asarray(search_points, dtype=np.float32).reshape(-1, 2)
    if points0.shape != points1.shape:
        raise ValueError("XFeat match arrays must have identical shape")
    match_count = int(len(points0))
    if match_count < config.minimum_matches:
        return _unresolved(start, "insufficient_matches", match_count=match_count)

    cv2.setRNGSeed(config.opencv_rng_seed)
    homography, mask = cv2.findHomography(
        points0,
        points1,
        cv2.USAC_MAGSAC,
        config.ransac_reprojection_threshold_px,
        maxIters=config.ransac_max_iterations,
        confidence=config.ransac_confidence,
    )
    if homography is None or mask is None:
        return _unresolved(start, "homography_failure", match_count=match_count)
    inlier_count = int(np.count_nonzero(mask))
    if inlier_count < config.minimum_inliers:
        return _unresolved(
            start,
            "insufficient_inliers",
            match_count=match_count,
            inlier_count=inlier_count,
        )

    height, width = reference_shape
    center = np.asarray(
        [[[(width - 1.0) / 2.0, (height - 1.0) / 2.0]]],
        dtype=np.float32,
    )
    projected = cv2.perspectiveTransform(center, homography)[0, 0]
    if not np.all(np.isfinite(projected)):
        return _unresolved(
            start,
            "nonfinite_projection",
            match_count=match_count,
            inlier_count=inlier_count,
        )
    return XFeatBaselineResult(
        method="xfeat_official_star_homography",
        x=float(projected[0]),
        y=float(projected[1]),
        status="resolved",
        runtime_ms=(time.perf_counter_ns() - start) / 1_000_000.0,
        diagnostics={
            "match_count": match_count,
            "inlier_count": inlier_count,
            "inlier_fraction": float(inlier_count / match_count),
            "homography": np.asarray(homography, dtype=np.float64).tolist(),
            "reference_nominal_shape": [int(height), int(width)],
        },
    )


def run_xfeat_official_star(
    model: XFeatStarMatcher,
    reference: np.ndarray,
    search: np.ndarray,
    config: XFeatBaselineConfig = XFeatBaselineConfig(),
) -> XFeatBaselineResult:
    """Run the fixed official XFeat* matcher and project the reference center."""

    start = time.perf_counter_ns()
    reference_nominal, search_rgb = prepare_xfeat_inputs(
        reference, search, config.nominal_scale
    )
    try:
        points0, points1 = model.match_xfeat_star(
            reference_nominal,
            search_rgb,
            top_k=config.top_k,
        )
    except (RuntimeError, ValueError, cv2.error) as exc:
        return _unresolved(
            start,
            "matcher_runtime_error",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            reference_nominal_shape=list(reference_nominal.shape[:2]),
        )
    return estimate_reference_center(
        points0,
        points1,
        reference_nominal.shape[:2],
        config,
        start_ns=start,
    )
