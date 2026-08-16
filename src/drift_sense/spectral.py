"""Reciprocal-space preprocessing and peak detection."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SpectralPeak:
    fy: float
    fx: float
    power: float
    symmetry: float

    @property
    def frequency(self) -> float:
        return float(np.hypot(self.fx, self.fy))

    @property
    def pitch(self) -> float:
        return 1.0 / max(self.frequency, 1e-12)


def robust_float(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    values = np.clip(values, lo, hi)
    center = np.median(values)
    scale = np.median(np.abs(values - center)) * 1.4826
    standard_deviation = float(np.std(values))
    # Sparse numerical residues can have a near-zero MAD despite nonzero FFT
    # roundoff. In that case MAD scaling would manufacture enormous outliers.
    if scale < max(1e-6, 0.02 * standard_deviation):
        scale = standard_deviation
    if scale < 1e-6:
        return np.zeros_like(values)
    return (values - center) / scale


def log_power_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Windowed FFT and log power, suitable for lattice peak analysis."""
    data = robust_float(image)
    h, w = data.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    transformed = np.fft.fftshift(np.fft.fft2(data * window))
    log_power = np.log1p(np.abs(transformed) ** 2).astype(np.float32)
    return transformed, log_power


def detect_reciprocal_peaks(
    image: np.ndarray,
    max_peaks: int = 24,
    min_frequency: float = 0.025,
    max_frequency: float = 0.24,
) -> list[SpectralPeak]:
    _, power = log_power_spectrum(image)
    h, w = power.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    fxx, fyy = np.meshgrid(fx, fy)
    radius = np.hypot(fxx, fyy)
    valid = (radius >= min_frequency) & (radius <= max_frequency)
    # Local maxima only; this prevents broad window sidelobes dominating the list.
    local_max = power == ndimage.maximum_filter(power, size=5, mode="nearest")
    floor = np.percentile(power[valid], 80.0) if np.any(valid) else np.inf
    ys, xs = np.nonzero(valid & local_max & (power >= floor))
    if len(ys) == 0:
        return []
    order = np.argsort(power[ys, xs])[::-1]
    peaks: list[SpectralPeak] = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        opposite_y = int((h - y) % h)
        opposite_x = int((w - x) % w)
        p = float(power[y, x])
        q = float(power[opposite_y, opposite_x])
        symmetry = min(p, q) / max(p, q, 1e-6)
        if symmetry < 0.55:
            continue
        candidate = SpectralPeak(float(fy[y]), float(fx[x]), p, symmetry)
        # Keep one representative from each conjugate pair.
        if candidate.fy < 0 or (abs(candidate.fy) < 1e-12 and candidate.fx < 0):
            continue
        peaks.append(candidate)
        if len(peaks) >= max_peaks:
            break
    return peaks


def smooth_notch_mask(
    shape: tuple[int, int], peaks: list[SpectralPeak], sigma_bins: float = 1.8
) -> np.ndarray:
    """Gaussian reciprocal-space mask around peaks and their conjugates."""
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    for peak in peaks:
        py = h / 2.0 + peak.fy * h
        px = w / 2.0 + peak.fx * w
        for y, x in ((py, px), (h - py, w - px)):
            d2 = (yy - y) ** 2 + (xx - x) ** 2
            mask = np.maximum(mask, np.exp(-0.5 * d2 / max(sigma_bins**2, 1e-6)))
    return np.clip(mask, 0.0, 1.0)
