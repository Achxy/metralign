"""Normalized template correlation with an OpenCV and SciPy implementation."""

from __future__ import annotations

import numpy as np
from scipy import signal

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only on broken local OpenCV installs
    cv2 = None


def zncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    search = np.ascontiguousarray(search, dtype=np.float32)
    template = np.ascontiguousarray(template, dtype=np.float32)
    if template.shape[0] > search.shape[0] or template.shape[1] > search.shape[1]:
        raise ValueError("template is larger than search image")
    if cv2 is not None:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        return np.nan_to_num(result, nan=-1.0).astype(np.float32)
    centered = template - np.mean(template)
    energy = float(np.sum(centered**2))
    if energy < 1e-12:
        return np.full(
            (search.shape[0] - template.shape[0] + 1, search.shape[1] - template.shape[1] + 1),
            -1.0,
            dtype=np.float32,
        )
    kernel = np.ones(template.shape, dtype=np.float32)
    n = float(template.size)
    sums = signal.fftconvolve(search, kernel, mode="valid")
    sums2 = signal.fftconvolve(search * search, kernel, mode="valid")
    numerator = signal.fftconvolve(search, centered[::-1, ::-1], mode="valid")
    variance = np.maximum(sums2 - sums * sums / n, 1e-12)
    return np.clip(numerator / np.sqrt(variance * energy), -1.0, 1.0).astype(np.float32)


def weighted_score_map(
    search_channels: dict[str, np.ndarray],
    template_channels: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    maps = {
        name: zncc_map(search_channels[name], template_channels[name])
        for name, weight in weights.items()
        if weight > 0
    }
    total_weight = sum(weights[name] for name in maps)
    combined = sum(weights[name] * value for name, value in maps.items()) / total_weight
    return np.asarray(combined, dtype=np.float32), maps


def balanced_residual_score_map(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse orthogonal residual evidence and retain weakest-channel support.

    The arithmetic mean ranks candidates without allowing one scan-corrupted
    direction to veto a strong match. The elementwise minimum is returned as a
    separate, candidate-local support gate so a single-channel spike cannot be
    accepted as a two-direction residual match.
    """
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("residual score maps must have equal shapes")
    fused = 0.5 * (first + second)
    support = np.minimum(first, second)
    return np.asarray(fused, dtype=np.float32), np.asarray(support, dtype=np.float32)


def candidate_supported_peak(
    fused: np.ndarray,
    support: np.ndarray,
    minimum_support: float,
) -> tuple[int, int] | None:
    """Return the fused-map maximum only when that same site has support."""
    fused = np.asarray(fused)
    support = np.asarray(support)
    if fused.shape != support.shape:
        raise ValueError("fused and support maps must have equal shapes")
    y, x = np.unravel_index(int(np.argmax(fused)), fused.shape)
    if float(support[y, x]) < minimum_support:
        return None
    return int(y), int(x)
