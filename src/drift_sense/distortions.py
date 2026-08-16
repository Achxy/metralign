"""Independently switchable SEM acquisition effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage


@dataclass
class AcquisitionParameters:
    edge_strength: float = 0.20
    psf_sigma: float = 0.75
    poisson_peak: float = 180.0
    gaussian_sigma: float = 0.018
    gain: float = 1.0
    offset: float = 0.0
    slow_intensity_drift: float = 0.035
    scan_jitter_sigma: float = 0.20
    scan_jitter_correlation: float = 6.0

    def to_dict(self) -> dict:
        return asdict(self)


def secondary_electron_edge_response(image: np.ndarray, strength: float) -> np.ndarray:
    if strength == 0:
        return image
    gx = ndimage.sobel(image, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(image, axis=0, mode="reflect") / 8.0
    edge = np.hypot(gx, gy)
    return image + strength * edge


def scan_line_shifts(height: int, sigma: float, correlation: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return np.zeros(height, dtype=np.float32)
    raw = rng.normal(0.0, sigma, size=height).astype(np.float32)
    shifts = ndimage.gaussian_filter1d(raw, max(correlation, 0.01), mode="reflect")
    # Preserve the optical-center coordinate exactly.
    center = (height - 1.0) / 2.0
    shifts -= np.interp(center, np.arange(height), shifts)
    return shifts


def apply_scan_line_shift(image: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    if not np.any(shifts):
        return image
    yy, xx = np.indices(image.shape, dtype=np.float32)
    # output(y, x) samples input(y, x - shift), so content moves right by shift.
    return ndimage.map_coordinates(image, [yy, xx - shifts[:, None]], order=1, mode="reflect")


def apply_acquisition(
    image: np.ndarray,
    params: AcquisitionParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SEM-inspired signal formation and return image plus row shifts."""
    result = secondary_electron_edge_response(image.astype(np.float32), params.edge_strength)
    if params.psf_sigma > 0:
        result = ndimage.gaussian_filter(result, params.psf_sigma, mode="reflect")
    height, width = result.shape
    yy, xx = np.indices(result.shape, dtype=np.float32)
    xn = (xx - (width - 1) / 2) / max(width / 2, 1)
    yn = (yy - (height - 1) / 2) / max(height / 2, 1)
    drift = 1.0 + params.slow_intensity_drift * (0.65 * yn + 0.35 * xn)
    result = np.clip(result * drift, 0.0, None)
    if params.poisson_peak > 0:
        result = rng.poisson(np.clip(result, 0, 1.5) * params.poisson_peak).astype(np.float32) / params.poisson_peak
    if params.gaussian_sigma > 0:
        result += rng.normal(0.0, params.gaussian_sigma, result.shape).astype(np.float32)
    result = result * params.gain + params.offset
    shifts = scan_line_shifts(height, params.scan_jitter_sigma, params.scan_jitter_correlation, rng)
    result = apply_scan_line_shift(result, shifts)
    return np.clip(result, 0.0, 1.0), shifts
