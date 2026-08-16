"""Evidence-based ambiguity handling for repeated lattice sites."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .candidates import Candidate, local_maxima_mask


@dataclass(frozen=True)
class AmbiguityDecision:
    candidate: Candidate
    ambiguous: bool
    margin: float
    tied_count: int
    threshold: float
    local_perturbation: float
    score_tied: bool
    secondary_evidence: bool
    local_perturbation_support: bool
    transform_instability_support: bool
    low_residual_support: bool
    lattice_grouped: bool
    lattice_group_count: int
    lattice_group_coverage: float
    hypotheses: tuple[Candidate, ...]
    hypotheses_truncated: bool
    selection_prior_center: tuple[float, float]
    selection_prior_source: str


def _diagnostic_hypotheses(
    tied: list[Candidate],
    best: Candidate,
    chosen: Candidate,
    center_x: float,
    center_y: float,
    limit: int,
) -> tuple[Candidate, ...]:
    """Select a compact, deterministic view of a potentially huge tie family."""
    if limit < 1:
        return ()
    ordered_by_score = sorted(tied, key=lambda item: (-item.score, item.y, item.x))
    ordered_by_center = sorted(
        tied,
        key=lambda item: (
            (item.x - center_x) ** 2 + (item.y - center_y) ** 2,
            -item.score,
            item.y,
            item.x,
        ),
    )
    result: list[Candidate] = []
    for candidate in (chosen, best, *ordered_by_center, *ordered_by_score):
        if candidate not in result:
            result.append(candidate)
        if len(result) >= limit:
            break
    return tuple(result)


def _group_by_lattice_offset(
    xs: np.ndarray,
    ys: np.ndarray,
    score_map: np.ndarray,
    best: Candidate,
    real_basis: np.ndarray,
    tolerance: float,
) -> list[Candidate]:
    """Collapse maxima representing the same approximate integer lattice site."""
    representatives: dict[tuple[int, int], Candidate] = {}
    for x, y in zip(xs, ys, strict=True):
        delta = np.array([float(x - best.x), float(y - best.y)])
        try:
            offset = np.linalg.solve(real_basis, delta)
        except np.linalg.LinAlgError:
            return []
        rounded = np.rint(offset)
        if float(np.max(np.abs(offset - rounded))) > tolerance:
            continue
        key = (int(rounded[0]), int(rounded[1]))
        candidate = Candidate(int(x), int(y), float(score_map[int(y), int(x)]))
        previous = representatives.get(key)
        if previous is None or candidate.score > previous.score:
            representatives[key] = candidate
    return list(representatives.values())


def choose_candidate(
    candidates: list[Candidate],
    score_map: np.ndarray,
    search_shape: tuple[int, int],
    template_shape: tuple[int, int],
    absolute_margin: float = 0.004,
    nms_radius: int = 6,
    real_basis: np.ndarray | None = None,
    lattice_tolerance: float = 0.30,
    residual_evidence: float | None = None,
    residual_floor: float = 0.25,
    transform_stability: float | None = None,
    apply_center_rule: bool = True,
    max_hypotheses: int = 8,
    prior_center: tuple[float, float] | None = None,
) -> AmbiguityDecision:
    if not candidates:
        raise ValueError("no candidates")
    best = candidates[0]
    runner_score = candidates[1].score if len(candidates) > 1 else -float("inf")
    margin = best.score - runner_score
    y, x = best.y, best.x
    y0, y1 = max(0, y - 1), min(score_map.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(score_map.shape[1], x + 2)
    perturbation = float(np.std(score_map[y0:y1, x0:x1]))
    # The 0.30 multiplier is calibrated to the score change produced by a
    # one-pixel perturbation around a resolved peak.  It avoids treating broad
    # but clearly ranked maxima as ties while retaining capture-noise stability.
    perturbation_threshold = 0.30 * perturbation
    threshold = max(absolute_margin, perturbation_threshold)
    # Top-K truncation is safe for ordinary ranking but not for the contractual
    # center-nearest tie break: a constant or highly periodic score map can have
    # thousands of equally valid maxima, with the central one absent from an
    # arbitrary argpartition subset.  Inspect every NMS maximum within the
    # evidence threshold without materializing a huge Candidate list.
    maxima = local_maxima_mask(score_map, nms_radius)
    tied_mask = maxima & (score_map >= best.score - threshold)
    tied_y, tied_x = np.nonzero(tied_mask)
    raw_tied_count = int(tied_y.size)
    lattice_grouped = False
    lattice_group_coverage = 0.0
    lattice_group_count = raw_tied_count
    raw_tied = [
        Candidate(int(x), int(y), float(score_map[int(y), int(x)]))
        for x, y in zip(tied_x, tied_y, strict=True)
    ]
    if real_basis is not None and tied_y.size > 1:
        grouped = _group_by_lattice_offset(
            tied_x,
            tied_y,
            score_map,
            best,
            real_basis,
            lattice_tolerance,
        )
        lattice_group_count = len(grouped)
        lattice_group_coverage = len(grouped) / max(raw_tied_count, 1)
        # A reciprocal basis is considered reliable for ambiguity grouping only
        # when it explains most tied maxima. Grouping is diagnostic: the
        # contractual center-nearest rule always considers every raw tied peak,
        # including the small fraction not explained by a noisy basis estimate.
        if grouped and lattice_group_coverage >= 0.65:
            lattice_grouped = True
    tied_count = len(raw_tied)
    score_tied = tied_count > 1 and margin <= threshold
    perturbation_support = margin <= perturbation_threshold + 1e-12
    transform_support = transform_stability is not None and transform_stability < 0.35
    residual_support = residual_evidence is not None and residual_evidence < residual_floor
    # Peak-neighborhood roughness is derived from the same score map as the tie
    # itself, so it is not independent evidence that absolute-site information
    # is absent.  It remains diagnostic, but center fallback requires weak
    # residual evidence or an unstable transform estimate.
    secondary_evidence = bool(transform_support or residual_support)
    half_x = (template_shape[1] - 1.0) / 2.0
    half_y = (template_shape[0] - 1.0) / 2.0
    if prior_center is None:
        output_prior_x = (search_shape[1] - 1.0) / 2.0
        output_prior_y = (search_shape[0] - 1.0) / 2.0
        prior_source = "image_center_default"
    else:
        output_prior_x, output_prior_y = map(float, prior_center)
        if not np.isfinite(output_prior_x) or not np.isfinite(output_prior_y):
            raise ValueError("prior center must be finite")
        if not (0.0 <= output_prior_x <= search_shape[1] - 1.0):
            raise ValueError("prior center x is outside the search image")
        if not (0.0 <= output_prior_y <= search_shape[0] - 1.0):
            raise ValueError("prior center y is outside the search image")
        prior_source = "user_supplied"
    center_x = output_prior_x - half_x
    center_y = output_prior_y - half_y
    if not apply_center_rule or not score_tied or not secondary_evidence:
        hypotheses = _diagnostic_hypotheses(
            raw_tied, best, best, center_x, center_y, max_hypotheses
        )
        return AmbiguityDecision(
            best,
            False,
            margin,
            tied_count,
            threshold,
            perturbation,
            score_tied,
            secondary_evidence,
            perturbation_support,
            transform_support,
            residual_support,
            lattice_grouped,
            lattice_group_count,
            lattice_group_coverage,
            hypotheses,
            len(raw_tied) > len(hypotheses),
            (output_prior_x, output_prior_y),
            prior_source,
        )
    chosen = min(
        raw_tied,
        key=lambda item: (
            (item.x - center_x) ** 2 + (item.y - center_y) ** 2,
            -item.score,
            item.y,
            item.x,
        ),
    )
    hypotheses = _diagnostic_hypotheses(
        raw_tied, best, chosen, center_x, center_y, max_hypotheses
    )
    return AmbiguityDecision(
        chosen,
        True,
        margin,
        tied_count,
        threshold,
        perturbation,
        score_tied,
        secondary_evidence,
        perturbation_support,
        transform_support,
        residual_support,
        lattice_grouped,
        lattice_group_count,
        lattice_group_coverage,
        hypotheses,
        len(raw_tied) > len(hypotheses),
        (output_prior_x, output_prior_y),
        prior_source,
    )
