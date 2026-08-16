"""Independent cross-generator capture path used only for held-out evaluation.

This module intentionally differs from :mod:`sem_render`: it uses polyphase
sensor sampling, a morphological edge response, Fourier-domain probe blur, and
a separately ordered noise pipeline. It is not used by development/training
generation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

from .architectures import render_architecture
from .distortions import AcquisitionParameters, apply_scan_line_shift, scan_line_shifts
from .geometry import CaptureGeometry, sensor_grid


def _polyphase_sample(image: np.ndarray, factor: int, window: str) -> np.ndarray:
    kernel: str | tuple[str, float] = ("kaiser", 6.5) if window == "kaiser" else "hann"
    sampled = signal.resample_poly(image, up=1, down=factor, axis=0, window=kernel)
    sampled = signal.resample_poly(sampled, up=1, down=factor, axis=1, window=kernel)
    # resample_poly anchors output[0] to supersample[0], whose sensor coordinate
    # is (1-factor)/(2*factor). Shift the decimated image forward by the exact
    # center-of-pixel phase so output[k] again represents sensor coordinate k.
    phase = (factor - 1.0) / (2.0 * factor)
    if phase:
        sampled = ndimage.shift(
            sampled,
            shift=(-phase, -phase),
            order=3,
            mode="reflect",
            prefilter=True,
        )
    return sampled.astype(np.float32)


def render_capture_alternate(
    architecture: str,
    wafer_seed: int,
    geometry: CaptureGeometry,
    acquisition: AcquisitionParameters,
    acquisition_seed: int,
    supersample: int = 2,
    geometry_variant: str = "default",
    resampler_variant: str = "kaiser",
    disabled_variations: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(acquisition_seed)
    x, y = sensor_grid(geometry.width, geometry.height, supersample)
    wx, wy = geometry.sensor_to_world(x, y)
    latent = render_architecture(
        architecture, wx, wy, wafer_seed, geometry_variant, disabled_variations
    ).astype(np.float32)
    if supersample > 1:
        latent = _polyphase_sample(latent, supersample, resampler_variant)
        latent = latent[: geometry.height, : geometry.width]

    if acquisition.edge_strength:
        high = ndimage.maximum_filter(latent, size=3, mode="mirror")
        low = ndimage.minimum_filter(latent, size=3, mode="mirror")
        latent = latent + acquisition.edge_strength * 0.55 * (high - low)
    if acquisition.psf_sigma > 0:
        spectrum = np.fft.fftn(latent)
        spectrum = ndimage.fourier_gaussian(spectrum, acquisition.psf_sigma)
        latent = np.real(np.fft.ifftn(spectrum)).astype(np.float32)

    h, w = latent.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xn = (xx - (w - 1) / 2) / max(w / 2, 1)
    yn = (yy - (h - 1) / 2) / max(h / 2, 1)
    illumination = 1.0 + acquisition.slow_intensity_drift * (0.5 * yn + 0.25 * xn + 0.25 * xn * yn)
    result = np.clip(latent * illumination, 0.0, 1.5)
    # Electronic noise is applied before shot counting in this independent path.
    if acquisition.gaussian_sigma > 0:
        result += rng.normal(0.0, acquisition.gaussian_sigma, result.shape).astype(np.float32)
    if acquisition.poisson_peak > 0:
        result = rng.poisson(np.clip(result, 0.0, 1.5) * acquisition.poisson_peak).astype(np.float32)
        result /= acquisition.poisson_peak
    result = result * acquisition.gain + acquisition.offset
    shifts = scan_line_shifts(h, acquisition.scan_jitter_sigma, acquisition.scan_jitter_correlation, rng)
    result = apply_scan_line_shift(result, shifts)
    return np.clip(result, 0.0, 1.0), shifts
