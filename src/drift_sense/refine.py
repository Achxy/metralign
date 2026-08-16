"""Subpixel peak refinement methods."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample


def _parabolic_offset(a: float, b: float, c: float) -> float:
    denominator = a - 2.0 * b + c
    if abs(denominator) < 1e-12 or denominator >= 0:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denominator, -0.75, 0.75))


def parabolic_peak(score_map: np.ndarray, x: int, y: int) -> tuple[float, float, float]:
    if x <= 0 or y <= 0 or x >= score_map.shape[1] - 1 or y >= score_map.shape[0] - 1:
        return float(x), float(y), 0.0
    dx = _parabolic_offset(score_map[y, x - 1], score_map[y, x], score_map[y, x + 1])
    dy = _parabolic_offset(score_map[y - 1, x], score_map[y, x], score_map[y + 1, x])
    curvature = float(
        2 * score_map[y, x]
        - score_map[y, x - 1]
        - score_map[y, x + 1]
        + 2 * score_map[y, x]
        - score_map[y - 1, x]
        - score_map[y + 1, x]
    )
    return x + dx, y + dy, curvature


def dft_peak_1d(
    values: np.ndarray,
    index: int,
    radius: int = 2,
    upsample: int = 16,
) -> tuple[float, float]:
    """Refine a local score maximum by Fourier interpolation.

    This is a deliberately local DFT alternative for ablation: it upsamples a
    five-sample score neighborhood and bounds the result to the same 0.75-pixel
    capture interval as the default parabolic fit.
    """
    array = np.asarray(values, dtype=np.float64)
    if index < radius or index >= array.size - radius or upsample < 2:
        return float(index), 0.0
    patch = array[index - radius : index + radius + 1]
    interpolated = resample(patch, patch.size * upsample)
    offset = float(np.argmax(interpolated) / upsample - radius)
    offset = float(np.clip(offset, -0.75, 0.75))
    _, curvature = _parabolic_peak_1d(array, index)
    return float(index + offset), curvature


def _parabolic_peak_1d(values: np.ndarray, index: int) -> tuple[float, float]:
    if index <= 0 or index >= values.size - 1:
        return float(index), 0.0
    left, center, right = map(float, values[index - 1 : index + 2])
    denominator = left - 2.0 * center + right
    if denominator >= -1e-12:
        return float(index), 0.0
    offset = float(np.clip(0.5 * (left - right) / denominator, -0.75, 0.75))
    return float(index + offset), float(-denominator)


def dft_peak(
    score_map: np.ndarray,
    x: int,
    y: int,
    radius: int = 2,
    upsample: int = 16,
) -> tuple[float, float, float]:
    """Refine a 2-D maximum using separable local DFT interpolation."""
    if (
        x < radius
        or y < radius
        or x >= score_map.shape[1] - radius
        or y >= score_map.shape[0] - radius
        or upsample < 2
    ):
        return float(x), float(y), 0.0
    patch = np.asarray(
        score_map[y - radius : y + radius + 1, x - radius : x + radius + 1],
        dtype=np.float64,
    )
    size = patch.shape[0] * upsample
    interpolated = resample(resample(patch, size, axis=0), size, axis=1)
    peak_y, peak_x = np.unravel_index(int(np.argmax(interpolated)), interpolated.shape)
    dx = float(np.clip(peak_x / upsample - radius, -0.75, 0.75))
    dy = float(np.clip(peak_y / upsample - radius, -0.75, 0.75))
    _, _, curvature = parabolic_peak(score_map, x, y)
    return float(x + dx), float(y + dy), curvature
