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
    tied: list[Candidate]
    if real_basis is not None and tied_y.size > 1:
        tied = _group_by_lattice_offset(
            tied_x,
            tied_y,
            score_map,
            best,
            real_basis,
            lattice_tolerance,
        )
        lattice_group_count = len(tied)
        lattice_group_coverage = len(tied) / max(raw_tied_count, 1)
        # A reciprocal basis is considered reliable for ambiguity grouping only
        # when it explains most tied maxima.  Lower coverage indicates a bright
        # harmonic or incomplete basis and must not override center-nearest.
        if tied and lattice_group_coverage >= 0.65:
            lattice_grouped = True
        else:
            tied = [
                Candidate(int(x), int(y), float(score_map[int(y), int(x)]))
                for x, y in zip(tied_x, tied_y, strict=True)
            ]
    else:
        tied = [
            Candidate(int(x), int(y), float(score_map[int(y), int(x)]))
            for x, y in zip(tied_x, tied_y, strict=True)
        ]
    tied_count = len(tied)
    score_tied = tied_count > 1 and margin <= threshold
    perturbation_support = margin <= perturbation_threshold + 1e-12
    transform_support = transform_stability is not None and transform_stability < 0.35
    residual_support = residual_evidence is not None and residual_evidence < residual_floor
    secondary_evidence = bool(perturbation_support or transform_support or residual_support)
    if not apply_center_rule or not score_tied or not secondary_evidence:
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
        )
    center_x = (search_shape[1] - template_shape[1]) / 2.0
    center_y = (search_shape[0] - template_shape[0]) / 2.0
    chosen = min(tied, key=lambda item: (item.x - center_x) ** 2 + (item.y - center_y) ** 2)
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
    )
