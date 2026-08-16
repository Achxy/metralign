"""Top-K local-maximum extraction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    score: float


def local_maxima_mask(score_map: np.ndarray, nms_radius: int = 6) -> np.ndarray:
    """Return every local maximum under the configured NMS footprint."""
    values = np.asarray(score_map)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("score_map must be a non-empty two-dimensional array")
    size = max(1, 2 * int(nms_radius) + 1)
    return values == ndimage.maximum_filter(values, size=size, mode="nearest")


def top_k_candidates(score_map: np.ndarray, k: int = 32, nms_radius: int = 6) -> list[Candidate]:
    maxima = local_maxima_mask(score_map, nms_radius)
    ys, xs = np.nonzero(maxima)
    if len(ys) == 0:
        return []
    values = score_map[ys, xs]
    count = min(int(k), len(values))
    selected = np.argpartition(values, -count)[-count:]
    selected = selected[np.argsort(values[selected])[::-1]]
    return [Candidate(int(xs[i]), int(ys[i]), float(values[i])) for i in selected]
