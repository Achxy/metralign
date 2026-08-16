"""Deterministic baseline and lattice-aware localization pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import itertools
import time

import numpy as np
from scipy import ndimage

from .ambiguity import choose_candidate
from .candidates import top_k_candidates
from .confidence import confidence_from_evidence
from .correlation import (
    balanced_residual_score_map,
    candidate_supported_peak,
    weighted_score_map,
    zncc_map,
)
from .lattice import estimate_lattice, estimate_relative_transform, reciprocal_to_real_basis
from .refine import dft_peak, dft_peak_1d, parabolic_peak
from .representations import (
    PeriodicTransformEstimate,
    build_channels,
    estimate_periodic_transform,
    periodic_difference_channels,
)
from .safety import assess_absolute_site
from .spectral import robust_float

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class LocalizationConfig:
    method: str = "full"
    nominal_scale: float = 0.1
    scale_range: float = 0.006
    scale_steps: int = 7
    rotation_range: float = 3.0
    rotation_steps: int = 5
    top_k: int = 32
    nms_radius: int = 6
    structural_weight: float = 0.50
    gradient_weight: float = 0.18
    residual_weight: float = 0.32
    ambiguity_margin: float = 0.004
    # Calibrated above the 0.07129 worst observed top-to-center valid-peak gap
    # across the 100-pair development periodic family, while normal residual
    # evidence uses the much tighter ``ambiguity_margin`` below.
    periodic_stability_margin: float = 0.075
    residual_evidence_floor: float = 0.25
    transform_fallback_confidence: float = 0.35
    # Cumulative ablation controls for the shared ``full`` pipeline.  Defaults
    # preserve the calibrated production path; disabling a stage never routes
    # through one of the legacy method aliases below.
    enable_phase_calibration: bool = True
    periodic_evidence_channel: str = "structural"
    enable_spatial_residual: bool = True
    enable_lattice_grouping: bool = True
    enable_ambiguity_rule: bool = True
    subpixel_refinement: str = "parabolic"
    prior_center_x: float | None = None
    prior_center_y: float | None = None


@dataclass
class Prediction:
    x: float
    y: float
    method: str
    score: float
    selected_score: float
    runner_up_score: float
    score_margin: float
    ambiguity_flag: bool
    tied_count: int
    confidence: float
    selected_scale: float
    selected_rotation_deg: float
    spectral_scale: float
    spectral_rotation_deg: float
    spectral_confidence: float
    lattice_offset: list[float] | None
    ambiguity_evidence: dict[str, object]
    decision_support: dict[str, object]
    hypothesis_count: int
    hypotheses_truncated: bool
    hypotheses: list[dict[str, object]]
    pipeline_stages: dict[str, bool | str]
    channel_scores: dict[str, float]
    runtime_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    h = max(8, int(round(image.shape[0] * scale)))
    w = max(8, int(round(image.shape[1] * scale)))
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
        return cv2.resize(np.asarray(image, dtype=np.float32), (w, h), interpolation=interpolation)
    if scale < 1:
        sigma = max(0.0, 0.5 / scale - 0.5)
        filtered = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), sigma, mode="reflect")
    else:
        filtered = np.asarray(image, dtype=np.float32)
    return ndimage.zoom(filtered, (h / image.shape[0], w / image.shape[1]), order=3, mode="reflect")


def _rotate(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-8:
        return image
    h, w = image.shape
    if cv2 is not None:
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return ndimage.rotate(image, angle_deg, reshape=False, order=3, mode="reflect")


def _transform_template(reference: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    return _rotate(_resize(reference, scale), rotation).astype(np.float32)


def _grid(center: float, radius: float, steps: int) -> list[float]:
    if steps <= 1 or radius <= 0:
        return [float(center)]
    return [float(value) for value in np.linspace(center - radius, center + radius, steps)]


def _pipeline_stages(cfg: LocalizationConfig) -> dict[str, bool | str]:
    """Machine-readable controls for clean cumulative full-pipeline ablations."""
    return {
        "phase_calibration": bool(cfg.enable_phase_calibration),
        "evidence_channel": cfg.periodic_evidence_channel,
        "spatial_residual": bool(cfg.enable_spatial_residual),
        "lattice_family_candidates": bool(cfg.enable_lattice_grouping),
        "ambiguity_rule": bool(cfg.enable_ambiguity_rule),
        "subpixel_refinement": cfg.subpixel_refinement,
    }


def _prior_center(cfg: LocalizationConfig) -> tuple[float, float] | None:
    if cfg.prior_center_x is None:
        return None
    assert cfg.prior_center_y is not None
    return float(cfg.prior_center_x), float(cfg.prior_center_y)


def _single_hypothesis(x: float, y: float, score: float) -> list[dict[str, object]]:
    return [
        {
            "rank": 1,
            "center": [float(x), float(y)],
            "score": float(score),
            "score_delta": 0.0,
            "selected": True,
            "lattice_offset_from_selected": None,
        }
    ]


def _hypothesis_diagnostics(
    decision: object,
    template_shape: tuple[int, int],
    real_basis: np.ndarray | None,
) -> list[dict[str, object]]:
    """Convert score-map peaks to output-coordinate hypotheses."""
    half_x = (template_shape[1] - 1.0) / 2.0
    half_y = (template_shape[0] - 1.0) / 2.0
    selected = decision.candidate
    best_score = max((item.score for item in decision.hypotheses), default=selected.score)
    result: list[dict[str, object]] = []
    for rank, candidate in enumerate(decision.hypotheses, 1):
        lattice_offset = None
        if real_basis is not None:
            delta = np.array(
                [candidate.x - selected.x, candidate.y - selected.y], dtype=np.float64
            )
            try:
                lattice_offset = np.linalg.solve(real_basis, delta).tolist()
            except np.linalg.LinAlgError:
                lattice_offset = None
        result.append(
            {
                "rank": rank,
                "center": [float(candidate.x + half_x), float(candidate.y + half_y)],
                "score": float(candidate.score),
                "score_delta": float(best_score - candidate.score),
                "selected": bool(candidate == selected),
                "lattice_offset_from_selected": lattice_offset,
            }
        )
    return result


def _decision_support(
    confidence: float,
    decision: object,
    residual_evidence: float,
    transform_stability: float,
) -> dict[str, object]:
    return assess_absolute_site(
        match_confidence=confidence,
        # ``ambiguous`` here means that the image score supports a candidate
        # family, not that the optional center/prior fallback was applied.
        # Keeping those concepts separate lets diagnostics recommend review
        # even when strong residual evidence makes the score-best candidate the
        # least assumptive coordinate to return.
        ambiguous=bool(decision.score_tied),
        residual_evidence=float(residual_evidence),
        transform_stability=float(transform_stability),
    ).to_dict()


def _baseline0(reference: np.ndarray, search: np.ndarray, cfg: LocalizationConfig) -> Prediction:
    start = time.perf_counter()
    template = robust_float(_resize(reference, cfg.nominal_scale)).astype(np.float32)
    search_norm = robust_float(search).astype(np.float32)
    score_map = zncc_map(search_norm, template)
    y, x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    rx, ry, curvature = parabolic_peak(score_map, int(x), int(y))
    candidates = top_k_candidates(score_map, 2, cfg.nms_radius)
    runner = candidates[1].score if len(candidates) > 1 else -1.0
    score = float(score_map[y, x])
    margin = score - runner
    confidence = confidence_from_evidence(score, margin, curvature)
    output_x = rx + (template.shape[1] - 1) / 2.0
    output_y = ry + (template.shape[0] - 1) / 2.0
    return Prediction(
        x=output_x,
        y=output_y,
        method="baseline0",
        score=score,
        selected_score=score,
        runner_up_score=runner,
        score_margin=margin,
        ambiguity_flag=False,
        tied_count=1,
        confidence=confidence,
        selected_scale=cfg.nominal_scale,
        selected_rotation_deg=0.0,
        spectral_scale=cfg.nominal_scale,
        spectral_rotation_deg=0.0,
        spectral_confidence=0.0,
        lattice_offset=None,
        ambiguity_evidence={
            "score_threshold": 0.0,
            "local_perturbation": 0.0,
            "transform_stability": 1.0,
            "residual_evidence": score,
            "score_tied": False,
            "secondary_evidence": False,
            "local_perturbation_support": False,
            "transform_instability_support": False,
            "low_residual_support": False,
            "lattice_grouped": False,
            "lattice_group_count": 1,
            "lattice_group_coverage": 0.0,
        },
        decision_support=assess_absolute_site(
            match_confidence=confidence,
            ambiguous=False,
            residual_evidence=score,
            transform_stability=1.0,
        ).to_dict(),
        hypothesis_count=1,
        hypotheses_truncated=False,
        hypotheses=_single_hypothesis(output_x, output_y, score),
        pipeline_stages={
            "phase_calibration": False,
            "evidence_channel": "structural",
            "spatial_residual": False,
            "lattice_family_candidates": False,
            "ambiguity_rule": False,
            "subpixel_refinement": "parabolic",
        },
        channel_scores={"structural": score},
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )


def _supports_periodic_difference(
    image_shape: tuple[int, int],
    pitch_x: float,
    pitch_y: float,
) -> bool:
    """Return whether one-period differencing leaves a usable image interior."""
    shift_x = max(1, int(round(pitch_x)))
    shift_y = max(1, int(round(pitch_y)))
    return bool(
        image_shape[1] > 2 * shift_x + 4
        and image_shape[0] > 2 * shift_y + 4
    )


def _small_periodic_template_fallback(
    reference: np.ndarray,
    search: np.ndarray,
    cfg: LocalizationConfig,
    estimate: PeriodicTransformEstimate,
    pitch_x: float,
    pitch_y: float,
    template_shape: tuple[int, int],
    unsupported_inputs: list[str],
    start: float,
) -> Prediction:
    """Use the existing ``baseline0`` matcher when periodic differencing is undefined.

    This is an execution fallback, not a second periodic decision rule.  It is
    deliberately identical to the public ``baseline0`` matcher and is exposed
    in both stage and ambiguity diagnostics while retaining the requested
    ``full`` method in the output contract.
    """
    baseline = _baseline0(reference, search, cfg)
    stages = _pipeline_stages(cfg)
    stages["fallback"] = "baseline0_small_periodic_template"
    evidence = dict(baseline.ambiguity_evidence)
    evidence.update(
        {
            "fallback_applied": True,
            "fallback_reason": "periodic_difference_input_too_small",
            "fallback_unsupported_inputs": list(unsupported_inputs),
            "fallback_template_shape": [int(template_shape[0]), int(template_shape[1])],
            "estimated_pitch": [float(pitch_x), float(pitch_y)],
        }
    )
    decision_support = dict(baseline.decision_support)
    decision_support.update(
        {
            "status": "review",
            "review_recommended": True,
            "conservative_abstention_recommended": True,
            "absolute_site_confidence": 0.0,
            "reasons": ["periodic_model_unsupported"],
        }
    )
    return replace(
        baseline,
        method=cfg.method,
        spectral_scale=float(estimate.scale),
        spectral_rotation_deg=float(estimate.rotation_deg),
        spectral_confidence=float(estimate.confidence),
        ambiguity_evidence=evidence,
        decision_support=decision_support,
        pipeline_stages=stages,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )


def _parabolic_1d(values: np.ndarray, index: int) -> tuple[float, float]:
    """Return a bounded subpixel maximum and its one-dimensional curvature."""
    if index <= 0 or index >= values.size - 1:
        return float(index), 0.0
    left, center, right = (float(values[index - 1]), float(values[index]), float(values[index + 1]))
    denominator = left - 2.0 * center + right
    if denominator >= -1e-12:
        return float(index), 0.0
    offset = float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))
    return float(index + offset), float(-denominator)


def _refine_1d(
    values: np.ndarray,
    index: int,
    method: str,
) -> tuple[float, float]:
    if method == "none":
        return float(index), 0.0
    if method == "dft":
        return dft_peak_1d(values, index)
    return _parabolic_1d(values, index)


def _refine_2d(
    score_map: np.ndarray,
    x: int,
    y: int,
    method: str,
) -> tuple[float, float, float]:
    if method == "none":
        return float(x), float(y), 0.0
    if method == "dft":
        return dft_peak(score_map, x, y)
    return parabolic_peak(score_map, x, y)


def _axis_projection_maps(
    search_channels: dict[str, np.ndarray],
    template_channels: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Match independent x/y process-variation sequences in line arrays."""
    search_x = np.mean(search_channels["period_x"], axis=0, dtype=np.float64).astype(np.float32)
    template_x = np.mean(template_channels["period_x"], axis=0, dtype=np.float64).astype(np.float32)
    search_y = np.mean(search_channels["period_y"], axis=1, dtype=np.float64).astype(np.float32)
    template_y = np.mean(template_channels["period_y"], axis=1, dtype=np.float64).astype(np.float32)
    x_scores = zncc_map(search_x[None, :], template_x[None, :])[0]
    y_scores = zncc_map(search_y[:, None], template_y[:, None])[:, 0]
    return x_scores, y_scores


def _projection_energy_fraction(channel: np.ndarray, axis: int) -> float:
    """Fraction of residual energy explained by a line-constant sequence."""
    centered = np.asarray(channel, dtype=np.float32) - float(np.mean(channel))
    projection = np.mean(centered, axis=axis, dtype=np.float64)
    return float(np.mean(projection * projection) / max(float(np.mean(centered * centered)), 1e-9))


def _solve_axis_center(
    x_at_search_center: float,
    y_at_search_center: float,
    search_shape: tuple[int, int],
    x_vector: tuple[float, float],
    y_vector: tuple[float, float],
) -> tuple[float, float]:
    """Correct projected line positions for the measured lattice slant."""
    center_x = (search_shape[1] - 1.0) / 2.0
    center_y = (search_shape[0] - 1.0) / 2.0
    slope_x = -x_vector[1] / max(abs(x_vector[0]), 1e-9)
    slope_y = -y_vector[0] / max(abs(y_vector[1]), 1e-9)
    matrix = np.array([[1.0, -slope_x], [-slope_y, 1.0]], dtype=np.float64)
    rhs = np.array(
        [x_at_search_center - slope_x * center_y, y_at_search_center - slope_y * center_x],
        dtype=np.float64,
    )
    try:
        x, y = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - impossible for near-orthogonal axes
        return x_at_search_center, y_at_search_center
    return float(x), float(y)


def _periodic_backbone_tie_prediction(
    reference: np.ndarray,
    search: np.ndarray,
    cfg: LocalizationConfig,
    estimate: object,
    scale: float,
    rotation: float,
    template: np.ndarray,
    start: float,
    residual_evidence: float,
) -> Prediction:
    """Choose the nearest stable phase when residual evidence is insufficient."""
    template_norm = robust_float(template).astype(np.float32)
    search_norm = robust_float(search).astype(np.float32)
    score_map = zncc_map(search_norm, template_norm)
    candidates = top_k_candidates(score_map, cfg.top_k, cfg.nms_radius)
    decision, real_basis = _choose_candidate_with_evidence(
        candidates,
        score_map,
        search.shape,
        template.shape,
        max(cfg.ambiguity_margin, cfg.periodic_stability_margin),
        cfg.nms_radius,
        search,
        residual_evidence,
        float(estimate.confidence),
        cfg.residual_evidence_floor,
        cfg.enable_lattice_grouping,
        cfg.enable_ambiguity_rule,
        _prior_center(cfg),
    )
    chosen = decision.candidate
    peak_x, peak_y, curvature = _refine_2d(
        score_map, chosen.x, chosen.y, cfg.subpixel_refinement
    )
    best_score = candidates[0].score
    runner = candidates[1].score if len(candidates) > 1 else -1.0
    best_margin = best_score - runner
    confidence = confidence_from_evidence(chosen.score, decision.margin, curvature)
    hypotheses = _hypothesis_diagnostics(decision, template.shape, real_basis)
    return Prediction(
        x=peak_x + (template.shape[1] - 1.0) / 2.0,
        y=peak_y + (template.shape[0] - 1.0) / 2.0,
        method=cfg.method,
        score=best_score,
        selected_score=chosen.score,
        runner_up_score=runner,
        score_margin=best_margin,
        ambiguity_flag=decision.ambiguous,
        tied_count=decision.tied_count,
        confidence=confidence,
        selected_scale=scale,
        selected_rotation_deg=rotation,
        spectral_scale=float(estimate.scale),
        spectral_rotation_deg=float(estimate.rotation_deg),
        spectral_confidence=float(estimate.confidence),
        lattice_offset=_lattice_offset_for_choice(
            search,
            candidates[0],
            chosen,
            real_basis,
            allow_estimate=cfg.enable_lattice_grouping,
        ),
        ambiguity_evidence=_ambiguity_evidence(
            decision, residual_evidence, float(estimate.confidence)
        ),
        decision_support=_decision_support(
            confidence, decision, residual_evidence, float(estimate.confidence)
        ),
        hypothesis_count=decision.tied_count,
        hypotheses_truncated=bool(decision.hypotheses_truncated),
        hypotheses=hypotheses,
        pipeline_stages=_pipeline_stages(cfg),
        channel_scores={"structural": chosen.score, "residual": residual_evidence},
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )


def _reliable_real_basis(search: np.ndarray) -> np.ndarray | None:
    lattice = estimate_lattice(search)
    if lattice.confidence < 0.5:
        return None
    real_basis = reciprocal_to_real_basis(lattice)
    if real_basis is None or not np.all(np.isfinite(real_basis)):
        return None
    lengths = np.linalg.norm(real_basis, axis=0)
    if np.any(lengths < 3.0) or np.any(lengths > 40.0) or np.linalg.cond(real_basis) > 20.0:
        return None
    return real_basis


def _choose_candidate_with_evidence(
    candidates: list,
    score_map: np.ndarray,
    search_shape: tuple[int, int],
    template_shape: tuple[int, int],
    absolute_margin: float,
    nms_radius: int,
    search: np.ndarray,
    residual_evidence: float,
    transform_stability: float,
    residual_floor: float,
    enable_lattice_grouping: bool = True,
    enable_ambiguity_rule: bool = True,
    prior_center: tuple[float, float] | None = None,
) -> tuple[object, np.ndarray | None]:
    decision = choose_candidate(
        candidates,
        score_map,
        search_shape,
        template_shape,
        absolute_margin,
        nms_radius,
        residual_evidence=residual_evidence,
        residual_floor=residual_floor,
        transform_stability=transform_stability,
        apply_center_rule=enable_ambiguity_rule,
        prior_center=prior_center,
    )
    real_basis = None
    if enable_lattice_grouping and decision.score_tied and decision.tied_count > 1:
        real_basis = _reliable_real_basis(search)
        if real_basis is not None:
            decision = choose_candidate(
                candidates,
                score_map,
                search_shape,
                template_shape,
                absolute_margin,
                nms_radius,
                real_basis=real_basis,
                residual_evidence=residual_evidence,
                residual_floor=residual_floor,
                transform_stability=transform_stability,
                apply_center_rule=enable_ambiguity_rule,
                prior_center=prior_center,
            )
    return decision, real_basis


def _ambiguity_evidence(
    decision: object,
    residual_evidence: float,
    transform_stability: float,
) -> dict[str, object]:
    return {
        "score_threshold": float(decision.threshold),
        "local_perturbation": float(decision.local_perturbation),
        "transform_stability": float(transform_stability),
        "residual_evidence": float(residual_evidence),
        "score_tied": bool(decision.score_tied),
        "secondary_evidence": bool(decision.secondary_evidence),
        "local_perturbation_support": bool(decision.local_perturbation_support),
        "transform_instability_support": bool(decision.transform_instability_support),
        "low_residual_support": bool(decision.low_residual_support),
        "lattice_grouped": bool(decision.lattice_grouped),
        "lattice_group_count": int(decision.lattice_group_count),
        "lattice_group_coverage": float(decision.lattice_group_coverage),
        "selection_prior_center": [
            float(decision.selection_prior_center[0]),
            float(decision.selection_prior_center[1]),
        ],
        "selection_prior_source": str(decision.selection_prior_source),
    }


def _lattice_offset_for_choice(
    search: np.ndarray,
    best: object,
    selected: object,
    real_basis: np.ndarray | None = None,
    allow_estimate: bool = True,
) -> list[float] | None:
    """Express an ambiguity tie-break displacement in real-lattice units."""
    if best.x == selected.x and best.y == selected.y:
        return None
    if real_basis is None and allow_estimate:
        real_basis = _reliable_real_basis(search)
    if real_basis is None:
        return None
    delta = np.array([selected.x - best.x, selected.y - best.y], dtype=np.float64)
    try:
        return np.linalg.solve(real_basis, delta).tolist()
    except np.linalg.LinAlgError:
        return None


def _bounded_residual_transform(
    reference: np.ndarray,
    search: np.ndarray,
    cfg: LocalizationConfig,
    pitch_x: float,
    pitch_y: float,
) -> tuple[float, float]:
    """Coarse-to-fine transform search used only when phase evidence is weak."""
    factor = 0.5
    coarse_search = _resize(search, factor)
    coarse_pitch_x = max(1.0, pitch_x * factor)
    coarse_pitch_y = max(1.0, pitch_y * factor)
    if not _supports_periodic_difference(
        coarse_search.shape, coarse_pitch_x, coarse_pitch_y
    ):
        return cfg.nominal_scale, 0.0
    search_channels = periodic_difference_channels(
        coarse_search,
        coarse_pitch_x,
        coarse_pitch_y,
        cfg.periodic_evidence_channel,
    )

    def search_grid(scales: list[float], rotations: list[float]) -> tuple[float, float, float] | None:
        best: tuple[float, float, float] | None = None
        for trial_scale, trial_rotation in itertools.product(scales, rotations):
            template = _transform_template(reference, trial_scale * factor, trial_rotation)
            try:
                channels = periodic_difference_channels(
                    template,
                    coarse_pitch_x,
                    coarse_pitch_y,
                    cfg.periodic_evidence_channel,
                )
            except ValueError:
                continue
            x_map = zncc_map(search_channels["period_x"], channels["period_x"])
            y_map = zncc_map(search_channels["period_y"], channels["period_y"])
            evidence = float(np.max(np.minimum(x_map, y_map)))
            trial = (evidence, trial_scale, trial_rotation)
            if best is None or trial > best:
                best = trial
        return best

    scales = _grid(cfg.nominal_scale, cfg.scale_range, 5)
    rotations = _grid(0.0, cfg.rotation_range, 5)
    coarse = search_grid(scales, rotations)
    if coarse is None:
        return cfg.nominal_scale, 0.0
    _, best_scale, best_rotation = coarse
    scale_radius = cfg.scale_range / 4.0
    rotation_radius = cfg.rotation_range / 4.0
    refined_scales = sorted(
        {
            float(
                np.clip(
                    value,
                    cfg.nominal_scale - cfg.scale_range,
                    cfg.nominal_scale + cfg.scale_range,
                )
            )
            for value in _grid(best_scale, scale_radius, 3)
        }
    )
    refined_rotations = sorted(
        {
            float(np.clip(value, -cfg.rotation_range, cfg.rotation_range))
            for value in _grid(best_rotation, rotation_radius, 3)
        }
    )
    refined = search_grid(
        refined_scales,
        refined_rotations,
    )
    if refined is None:
        return best_scale, best_rotation
    return float(refined[1]), float(refined[2])


def _periodic_localize(
    reference: np.ndarray,
    search: np.ndarray,
    cfg: LocalizationConfig,
) -> Prediction:
    """Fast periodic-backbone cancellation and residual disambiguation."""
    start = time.perf_counter()
    estimate = estimate_periodic_transform(reference, search, cfg.nominal_scale)
    phase_scale = float(
        np.clip(
            estimate.scale,
            cfg.nominal_scale - cfg.scale_range,
            cfg.nominal_scale + cfg.scale_range,
        )
    )
    pitch_x = float(np.clip(estimate.pitch_x, 4.0, 30.0))
    pitch_y = float(np.clip(estimate.pitch_y, 4.0, 30.0))
    if not cfg.enable_phase_calibration:
        scale, rotation = cfg.nominal_scale, 0.0
    elif estimate.confidence < cfg.transform_fallback_confidence:
        scale, rotation = _bounded_residual_transform(
            reference, search, cfg, pitch_x, pitch_y
        )
    else:
        scale = phase_scale
        rotation = float(np.clip(estimate.rotation_deg, -cfg.rotation_range, cfg.rotation_range))
    initial_template = _transform_template(reference, scale, rotation)
    if not cfg.enable_spatial_residual:
        return _periodic_backbone_tie_prediction(
            reference,
            search,
            cfg,
            estimate,
            scale,
            rotation,
            initial_template,
            start,
            0.0,
        )
    unsupported_inputs: list[str] = []
    if not _supports_periodic_difference(search.shape, pitch_x, pitch_y):
        unsupported_inputs.append("search")
    if not _supports_periodic_difference(initial_template.shape, pitch_x, pitch_y):
        unsupported_inputs.append("template")
    if unsupported_inputs:
        return _small_periodic_template_fallback(
            reference,
            search,
            cfg,
            estimate,
            pitch_x,
            pitch_y,
            initial_template.shape,
            unsupported_inputs,
            start,
        )
    search_channels = periodic_difference_channels(
        search, pitch_x, pitch_y, cfg.periodic_evidence_channel
    )
    initial_template_channels = periodic_difference_channels(
        initial_template, pitch_x, pitch_y, cfg.periodic_evidence_channel
    )
    template_scale = max(float(np.std(robust_float(initial_template))), 1e-9)
    residual_strength = max(
        float(np.std(initial_template_channels["period_x"])),
        float(np.std(initial_template_channels["period_y"])),
    ) / template_scale
    if residual_strength <= 0.03:
        # Exact periodicity contains no site-specific residual.  In that regime
        # small reflected-filter and resampling boundary effects are acquisition
        # artifacts, not localization evidence.  Score the periodic backbone and
        # apply the center-nearest rule across *all* tied phase-compatible peaks.
        return _periodic_backbone_tie_prediction(
            reference,
            search,
            cfg,
            estimate,
            scale,
            rotation,
            initial_template,
            start,
            residual_strength,
        )
    line_x_fraction = _projection_energy_fraction(initial_template_channels["period_x"], axis=0)
    line_y_fraction = _projection_energy_fraction(initial_template_channels["period_y"], axis=1)
    # A whole-fin/whole-gate generator is genuinely separable and benefits from
    # averaging over noise.  Segment-level roughness is not: independently
    # choosing its best row and column can create a nonexistent 2-D match.
    use_axis_projection = bool(
        estimate.axis_separable and min(line_x_fraction, line_y_fraction) >= 0.55
    )

    # Orthogonal line arrays expose independent, persistent fin/gate sequences.
    # Refining the transform with their 1-D correlations is inexpensive and is
    # substantially more stable than selecting a transform from one accidental
    # peak in a dense periodic score map.
    if use_axis_projection:
        scale_radius = min(cfg.scale_range, 0.00075)
        rotation_radius = min(cfg.rotation_range, 0.22)
        scales = sorted(
            {
                float(
                    np.clip(
                        value,
                        cfg.nominal_scale - cfg.scale_range,
                        cfg.nominal_scale + cfg.scale_range,
                    )
                )
                for value in _grid(scale, scale_radius, 5)
            }
        )
        rotations = sorted(
            {
                float(np.clip(value, -cfg.rotation_range, cfg.rotation_range))
                for value in _grid(rotation, rotation_radius, 5)
            }
        )
    else:
        scales = [scale]
        rotations = [rotation]

    best_axis: tuple[
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ] | None = None
    best_axis_key: tuple[float, float] | None = None
    for trial_scale, trial_rotation in itertools.product(scales, rotations):
        template = _transform_template(reference, trial_scale, trial_rotation)
        try:
            template_channels = periodic_difference_channels(
                template, pitch_x, pitch_y, cfg.periodic_evidence_channel
            )
        except ValueError:
            continue
        x_scores, y_scores = _axis_projection_maps(search_channels, template_channels)
        evidence = min(float(np.max(x_scores)), float(np.max(y_scores)))
        trial = (
            evidence,
            -abs(trial_scale - scale) - 1e-3 * abs(trial_rotation - rotation),
            trial_scale,
            trial_rotation,
            x_scores,
            y_scores,
            template,
            template_channels,
        )
        if best_axis_key is None or trial[:2] > best_axis_key:
            # Keep the score and a deterministic transform-prior tie breaker.
            best_axis = (
                trial[0],
                trial[2],
                trial[3],
                trial[4],
                trial[5],
                trial[6],
                trial[7],
            )
            best_axis_key = trial[:2]
    if best_axis is None:  # pragma: no cover - guarded by the image-size validation
        raise ValueError("unable to form a periodic residual template")
    axis_evidence, scale, rotation, x_scores, y_scores, template, template_channels = best_axis

    if use_axis_projection:
        x_index = int(np.argmax(x_scores))
        y_index = int(np.argmax(y_scores))
        refined_x, curvature_x = _refine_1d(x_scores, x_index, cfg.subpixel_refinement)
        refined_y, curvature_y = _refine_1d(y_scores, y_index, cfg.subpixel_refinement)
        projected_x = refined_x + (template.shape[1] - 1.0) / 2.0
        projected_y = refined_y + (template.shape[0] - 1.0) / 2.0
        center_x, center_y = _solve_axis_center(
            projected_x,
            projected_y,
            search.shape,
            estimate.x_vector,
            estimate.y_vector,
        )
        # The Cartesian map makes ambiguity and runner-up reporting explicit;
        # it is only ~3 MB for a 1000x1000 input and requires no dense 2-D FFT.
        # Addition preserves both independent rankings.  ``minimum`` creates a
        # flat ridge whenever one axis is more confident than the other and can
        # manufacture false ambiguity between otherwise distinct coordinates.
        score_map = (0.5 * (y_scores[:, None] + x_scores[None, :])).astype(np.float32)
        curvature = curvature_x + curvature_y
        channel_scores = {
            "period_x": float(x_scores[x_index]),
            "period_y": float(y_scores[y_index]),
        }
    else:
        x_map = zncc_map(search_channels["period_x"], template_channels["period_x"])
        y_map = zncc_map(search_channels["period_y"], template_channels["period_y"])
        # Both residual directions carry evidence. A hard minimum lets one
        # scan-distorted direction veto a strong correct match; equal-weight
        # fusion retains corroboration while the conservative minimum below is
        # still used to decide whether residual SNR is sufficient at all.
        score_map, conservative_map = balanced_residual_score_map(x_map, y_map)
        supported_peak = candidate_supported_peak(
            score_map, conservative_map, cfg.residual_evidence_floor
        )
        if supported_peak is None:
            raw_y, raw_x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
            return _periodic_backbone_tie_prediction(
                reference,
                search,
                cfg,
                estimate,
                scale,
                rotation,
                template,
                start,
                float(conservative_map[raw_y, raw_x]),
            )
        peak_y, peak_x = supported_peak
        refined_x, refined_y, curvature = _refine_2d(
            score_map, int(peak_x), int(peak_y), cfg.subpixel_refinement
        )
        center_x = refined_x + (template.shape[1] - 1.0) / 2.0
        center_y = refined_y + (template.shape[0] - 1.0) / 2.0
        channel_scores = {
            "period_x": float(x_map[peak_y, peak_x]),
            "period_y": float(y_map[peak_y, peak_x]),
        }
        residual_evidence = float(conservative_map[peak_y, peak_x])

    residual_evidence = float(
        axis_evidence if use_axis_projection else conservative_map[peak_y, peak_x]
    )
    candidates = top_k_candidates(score_map, cfg.top_k, cfg.nms_radius)
    decision, real_basis = _choose_candidate_with_evidence(
        candidates,
        score_map,
        search.shape,
        template.shape,
        cfg.ambiguity_margin,
        cfg.nms_radius,
        search,
        residual_evidence,
        float(estimate.confidence),
        cfg.residual_evidence_floor,
        cfg.enable_lattice_grouping,
        cfg.enable_ambiguity_rule,
        _prior_center(cfg),
    )
    chosen = decision.candidate
    # Make the center-nearest ambiguity rule authoritative rather than merely a
    # diagnostic.  Usually ``chosen`` is the raw maximum, so this is also the
    # single subpixel-refinement path for both unambiguous and tied evidence.
    if use_axis_projection:
        refined_x, curvature_x = _refine_1d(
            x_scores, chosen.x, cfg.subpixel_refinement
        )
        refined_y, curvature_y = _refine_1d(
            y_scores, chosen.y, cfg.subpixel_refinement
        )
        center_x, center_y = _solve_axis_center(
            refined_x + (template.shape[1] - 1.0) / 2.0,
            refined_y + (template.shape[0] - 1.0) / 2.0,
            search.shape,
            estimate.x_vector,
            estimate.y_vector,
        )
        curvature = curvature_x + curvature_y
        channel_scores = {
            "period_x": float(x_scores[chosen.x]),
            "period_y": float(y_scores[chosen.y]),
        }
    else:
        refined_x, refined_y, curvature = _refine_2d(
            score_map, chosen.x, chosen.y, cfg.subpixel_refinement
        )
        center_x = refined_x + (template.shape[1] - 1.0) / 2.0
        center_y = refined_y + (template.shape[0] - 1.0) / 2.0
        channel_scores = {
            "period_x": float(x_map[chosen.y, chosen.x]),
            "period_y": float(y_map[chosen.y, chosen.x]),
        }
    channel_scores["residual"] = float(
        min(channel_scores["period_x"], channel_scores["period_y"])
    )
    best_score = candidates[0].score
    runner = candidates[1].score if len(candidates) > 1 else -1.0
    selected_score = float(chosen.score)
    best_margin = best_score - runner
    confidence = confidence_from_evidence(selected_score, decision.margin, curvature)
    hypotheses = _hypothesis_diagnostics(decision, template.shape, real_basis)
    return Prediction(
        x=float(center_x),
        y=float(center_y),
        method=cfg.method,
        score=best_score,
        selected_score=selected_score,
        runner_up_score=runner,
        score_margin=best_margin,
        ambiguity_flag=decision.ambiguous,
        tied_count=decision.tied_count,
        confidence=confidence,
        selected_scale=float(scale),
        selected_rotation_deg=float(rotation),
        spectral_scale=float(estimate.scale),
        spectral_rotation_deg=float(estimate.rotation_deg),
        spectral_confidence=float(estimate.confidence),
        lattice_offset=_lattice_offset_for_choice(
            search,
            candidates[0],
            chosen,
            real_basis,
            allow_estimate=cfg.enable_lattice_grouping,
        ),
        ambiguity_evidence=_ambiguity_evidence(
            decision, residual_evidence, float(estimate.confidence)
        ),
        decision_support=_decision_support(
            confidence, decision, residual_evidence, float(estimate.confidence)
        ),
        hypothesis_count=decision.tied_count,
        hypotheses_truncated=bool(decision.hypotheses_truncated),
        hypotheses=hypotheses,
        pipeline_stages=_pipeline_stages(cfg),
        channel_scores=channel_scores,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )


def localize(reference: np.ndarray, search: np.ndarray, cfg: LocalizationConfig | None = None) -> Prediction:
    cfg = cfg or LocalizationConfig()
    reference = np.asarray(reference)
    search = np.asarray(search)
    if reference.ndim != 2 or search.ndim != 2:
        raise ValueError("reference and search must be two-dimensional grayscale images")
    if min(reference.shape) < 32 or min(search.shape) < 32:
        raise ValueError("images are too small")
    if (cfg.prior_center_x is None) != (cfg.prior_center_y is None):
        raise ValueError("prior center x and y must be provided together")
    if cfg.prior_center_x is not None:
        prior_x, prior_y = float(cfg.prior_center_x), float(cfg.prior_center_y)
        if not np.isfinite(prior_x) or not np.isfinite(prior_y):
            raise ValueError("prior center must be finite")
        if not (0.0 <= prior_x <= search.shape[1] - 1.0):
            raise ValueError("prior center x is outside the search image")
        if not (0.0 <= prior_y <= search.shape[0] - 1.0):
            raise ValueError("prior center y is outside the search image")
    if cfg.method == "baseline0":
        return _baseline0(reference, search, cfg)
    if cfg.method not in {"multiscale", "full", "structure_gradient", "structure_residual"}:
        raise ValueError(f"unknown method: {cfg.method}")
    if cfg.method == "full":
        if cfg.periodic_evidence_channel not in {"structural", "gradient", "raw"}:
            raise ValueError(
                f"unknown periodic evidence channel: {cfg.periodic_evidence_channel}"
            )
        if cfg.subpixel_refinement not in {"parabolic", "dft", "none"}:
            raise ValueError(f"unknown subpixel refinement: {cfg.subpixel_refinement}")
        return _periodic_localize(reference, search, cfg)

    start = time.perf_counter()
    nominal_template = _resize(reference, cfg.nominal_scale)
    spectral_correction, spectral_rotation, spectral_confidence = estimate_relative_transform(
        nominal_template, search
    )
    # The known 10x relation remains the prior. A spectral estimate only shifts
    # the grid when confident, and every shifted grid retains nominal coverage.
    estimated_scale = cfg.nominal_scale * spectral_correction
    scale_center = estimated_scale if spectral_confidence >= 0.45 else cfg.nominal_scale
    rotation_center = spectral_rotation if spectral_confidence >= 0.45 else 0.0
    scales = sorted(
        set(
            _grid(cfg.nominal_scale, cfg.scale_range, cfg.scale_steps)
            + _grid(scale_center, cfg.scale_range * 0.45, max(3, cfg.scale_steps // 2))
        )
    )
    rotations = sorted(
        set(
            _grid(0.0, cfg.rotation_range, cfg.rotation_steps)
            + _grid(rotation_center, cfg.rotation_range * 0.35, max(3, cfg.rotation_steps // 2))
        )
    )

    if cfg.method == "multiscale":
        weights = {"structural": 1.0}
        search_channels = {"structural": robust_float(search).astype(np.float32)}
    else:
        all_channels = build_channels(search)
        if cfg.method == "structure_gradient":
            weights = {"structural": 0.72, "gradient": 0.28}
        elif cfg.method == "structure_residual":
            weights = {"structural": 0.60, "residual": 0.40}
        else:
            weights = {
                "structural": cfg.structural_weight,
                "gradient": cfg.gradient_weight,
                "residual": cfg.residual_weight,
            }
        search_channels = {name: all_channels[name] for name in weights}

    best: tuple[float, float, float, np.ndarray, dict[str, np.ndarray], np.ndarray] | None = None
    for scale, rotation in itertools.product(scales, rotations):
        template = _transform_template(reference, scale, rotation)
        if cfg.method == "multiscale":
            template_channels = {"structural": robust_float(template).astype(np.float32)}
        else:
            channels = build_channels(template)
            template_channels = {name: channels[name] for name in weights}
        score_map, channel_maps = weighted_score_map(search_channels, template_channels, weights)
        peak = float(np.max(score_map))
        if best is None or peak > best[0]:
            best = (peak, scale, rotation, score_map, channel_maps, template)
    assert best is not None
    best_score, scale, rotation, score_map, channel_maps, template = best
    candidates = top_k_candidates(score_map, cfg.top_k, cfg.nms_radius)
    decision = choose_candidate(
        candidates,
        score_map,
        search.shape,
        template.shape,
        cfg.ambiguity_margin,
        cfg.nms_radius,
        prior_center=_prior_center(cfg),
    )
    chosen = decision.candidate
    peak_x, peak_y, curvature = parabolic_peak(score_map, chosen.x, chosen.y)
    best_score = candidates[0].score
    runner = candidates[1].score if len(candidates) > 1 else -1.0
    best_margin = best_score - runner
    channel_scores = {
        name: float(channel_map[chosen.y, chosen.x]) for name, channel_map in channel_maps.items()
    }

    lattice = estimate_lattice(search)
    real_basis = reciprocal_to_real_basis(lattice)
    lattice_offset = None
    if real_basis is not None and len(candidates) > 1:
        delta = np.array([chosen.x - candidates[0].x, chosen.y - candidates[0].y])
        try:
            lattice_offset = np.linalg.solve(real_basis, delta).tolist()
        except np.linalg.LinAlgError:
            lattice_offset = None
    confidence = confidence_from_evidence(chosen.score, decision.margin, curvature)
    residual_evidence = float(channel_scores.get("residual", chosen.score))
    hypotheses = _hypothesis_diagnostics(decision, template.shape, real_basis)
    return Prediction(
        x=peak_x + (template.shape[1] - 1) / 2.0,
        y=peak_y + (template.shape[0] - 1) / 2.0,
        method=cfg.method,
        score=best_score,
        selected_score=chosen.score,
        runner_up_score=runner,
        score_margin=best_margin,
        ambiguity_flag=decision.ambiguous,
        tied_count=decision.tied_count,
        confidence=confidence,
        selected_scale=scale,
        selected_rotation_deg=rotation,
        spectral_scale=estimated_scale,
        spectral_rotation_deg=spectral_rotation,
        spectral_confidence=spectral_confidence,
        lattice_offset=lattice_offset,
        ambiguity_evidence=_ambiguity_evidence(
            decision,
            residual_evidence,
            float(spectral_confidence),
        ),
        decision_support=_decision_support(
            confidence, decision, residual_evidence, float(spectral_confidence)
        ),
        hypothesis_count=decision.tied_count,
        hypotheses_truncated=bool(decision.hypotheses_truncated),
        hypotheses=hypotheses,
        pipeline_stages=_pipeline_stages(cfg),
        channel_scores=channel_scores,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
    )
