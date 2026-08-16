"""Structural, gradient, and periodic-suppressed matching channels.

The shift-difference representation is intentionally spatial.  Applying the
same lattice-period translation to both captures cancels the repeated unit
cell while retaining cell-to-cell process variation.  Unlike independently
estimated Fourier notch masks, the resulting template and search channels
therefore have exactly the same transfer function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .spectral import detect_reciprocal_peaks, robust_float, smooth_notch_mask


_MIN_PHASE_CONFIDENCE = 0.35


@dataclass(frozen=True)
class PeriodicTransformEstimate:
    """Relative capture transform and search-image lattice measurements."""

    scale: float
    rotation_deg: float
    pitch_x: float
    pitch_y: float
    x_vector: tuple[float, float]
    y_vector: tuple[float, float]
    axis_separable: bool
    confidence: float


def _quadratic_spectral_peak(
    projection: np.ndarray,
    minimum_frequency: float,
    maximum_frequency: float,
    padding: int = 16,
) -> float | None:
    """Estimate a one-dimensional peak frequency with zero-padding."""
    values = np.asarray(projection, dtype=np.float64)
    values = (values - np.mean(values)) * np.hanning(values.size)
    fft_size = 1
    while fft_size < values.size * padding:
        fft_size *= 2
    magnitude = np.abs(np.fft.rfft(values, fft_size))
    frequencies = np.fft.rfftfreq(fft_size)
    valid = np.flatnonzero(
        (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    )
    if valid.size == 0:
        return None
    index = int(valid[np.argmax(magnitude[valid])])
    if index <= 0 or index >= magnitude.size - 1 or magnitude[index] <= 1e-9:
        return float(frequencies[index])
    local = np.log(magnitude[index - 1 : index + 2] + 1e-12)
    denominator = local[0] - 2.0 * local[1] + local[2]
    offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (local[0] - local[2]) / denominator
    return float((index + np.clip(offset, -0.75, 0.75)) / fft_size)


def _phase_lattice_vector(
    normalized: np.ndarray,
    axis: str,
    frequency_range: tuple[float, float],
) -> tuple[np.ndarray, float] | None:
    """Measure a reciprocal vector from phase drift across scan lines."""
    if axis == "x":
        projection = np.mean(normalized, axis=0)
    elif axis == "y":
        projection = np.mean(normalized, axis=1)
    else:  # pragma: no cover - private call-site invariant
        raise ValueError(f"unknown axis: {axis}")
    frequency = _quadratic_spectral_peak(projection, *frequency_range)
    if frequency is None:
        return None

    if axis == "x":
        positions = np.arange(normalized.shape[1], dtype=np.float64)
        carrier = np.hanning(normalized.shape[1]) * np.exp(-2j * np.pi * frequency * positions)
        coefficients = normalized @ carrier
    else:
        positions = np.arange(normalized.shape[0], dtype=np.float64)
        carrier = np.hanning(normalized.shape[0]) * np.exp(-2j * np.pi * frequency * positions)
        coefficients = carrier @ normalized

    coordinates = np.arange(coefficients.size, dtype=np.float64)
    amplitude = np.abs(coefficients)
    phase = np.unwrap(np.angle(coefficients))
    keep = (
        (coordinates >= 0.08 * coefficients.size)
        & (coordinates <= 0.92 * coefficients.size)
        & (amplitude >= np.quantile(amplitude, 0.15))
    )
    if np.count_nonzero(keep) < 8:
        return None
    slope, intercept = np.polyfit(
        coordinates[keep], phase[keep], 1, w=np.sqrt(amplitude[keep] + 1e-9)
    )
    fitted = slope * coordinates[keep] + intercept
    phase_error = float(np.sqrt(np.average((phase[keep] - fitted) ** 2, weights=amplitude[keep])))
    cross_frequency = float(slope / (2.0 * np.pi))
    vector = (
        np.array([frequency, cross_frequency], dtype=np.float64)
        if axis == "x"
        else np.array([cross_frequency, frequency], dtype=np.float64)
    )
    confidence = float(np.exp(-min(phase_error, 8.0) / 2.5))
    return vector, confidence


def _axis_pitch_from_2d_peaks(image: np.ndarray, axis: str) -> float | None:
    """Recover an axis pitch when a 1-D projection cancels its fundamental.

    Staggered contact rows can nearly erase the x fundamental after averaging
    over y.  The 2-D spectrum retains the corresponding reciprocal peak.
    """
    peaks = detect_reciprocal_peaks(image, max_peaks=24)
    aligned = []
    for peak in peaks:
        frequency = max(float(np.hypot(peak.fx, peak.fy)), 1e-12)
        cross = abs(peak.fy) if axis == "x" else abs(peak.fx)
        along = abs(peak.fx) if axis == "x" else abs(peak.fy)
        if 5.0 <= peak.pitch <= 28.0 and along > 0.0 and cross / frequency <= 0.12:
            aligned.append(peak)
    if not aligned:
        return None
    return float(max(aligned, key=lambda peak: peak.power).pitch)


def estimate_periodic_transform(
    reference: np.ndarray,
    search: np.ndarray,
    nominal_scale: float = 0.1,
) -> PeriodicTransformEstimate:
    """Estimate scale/rotation from periodic phase without knowing location.

    The high-resolution reference has roughly one tenth the frequency of the
    search capture.  A zero-padded 1-D spectrum estimates each fundamental,
    while phase drift between scan lines resolves sub-degree orientation that
    is below a single 2-D FFT bin for a 1000-pixel image.
    """
    reference_norm = robust_float(reference).astype(np.float32)
    search_norm = robust_float(search).astype(np.float32)
    reference_range = (nominal_scale / 30.0, nominal_scale / 4.5)
    search_range = (1.0 / 30.0, 1.0 / 4.5)
    ref_x = _phase_lattice_vector(reference_norm, "x", reference_range)
    ref_y = _phase_lattice_vector(reference_norm, "y", reference_range)
    sea_x = _phase_lattice_vector(search_norm, "x", search_range)
    sea_y = _phase_lattice_vector(search_norm, "y", search_range)
    if None in (ref_x, ref_y, sea_x, sea_y):
        return PeriodicTransformEstimate(
            nominal_scale, 0.0, 10.0, 15.0, (0.1, 0.0), (0.0, 1.0 / 15.0), False, 0.0
        )
    assert ref_x is not None and ref_y is not None and sea_x is not None and sea_y is not None
    rx, ry = ref_x[0], ref_y[0]
    sx, sy = sea_x[0], sea_y[0]
    search_frequency_ratio = max(np.linalg.norm(sx), np.linalg.norm(sy)) / max(
        min(np.linalg.norm(sx), np.linalg.norm(sy)), 1e-9
    )
    # Orthogonal line arrays have two clearly separated fundamentals.  The
    # staggered contact array has similar x/y frequencies, and its x projection
    # alternates phase by row; use the stable y family for that case.
    x_confidence = min(ref_x[1], sea_x[1])
    y_confidence = min(ref_y[1], sea_y[1])
    reliable_x = x_confidence >= _MIN_PHASE_CONFIDENCE
    reliable_y = y_confidence >= _MIN_PHASE_CONFIDENCE
    axis_separable = bool(search_frequency_ratio >= 1.40 and reliable_x and reliable_y)
    if axis_separable:
        pairs = ((rx, sx, x_confidence), (ry, sy, y_confidence))
    elif reliable_y:
        pairs = ((ry, sy, y_confidence),)
    elif reliable_x:
        pairs = ((rx, sx, x_confidence),)
    else:
        pairs = ()
    scales: list[float] = []
    rotations: list[float] = []
    selected_confidences: list[float] = []
    for reference_vector, search_vector, pair_confidence in pairs:
        scales.append(float(np.linalg.norm(reference_vector) / np.linalg.norm(search_vector)))
        ref_angle = float(np.degrees(np.arctan2(reference_vector[1], reference_vector[0])))
        sea_angle = float(np.degrees(np.arctan2(search_vector[1], search_vector[0])))
        rotations.append(float((ref_angle - sea_angle + 90.0) % 180.0 - 90.0))
        selected_confidences.append(pair_confidence)
    if len(scales) == 2:
        rotation_disagreement = abs((rotations[0] - rotations[1] + 90.0) % 180.0 - 90.0)
        inconsistent = abs(scales[0] - scales[1]) > 0.012 or rotation_disagreement > 1.0
        if inconsistent:
            # Preserve the more coherent axis as a diagnostic prior, but lower
            # aggregate confidence so the localizer performs its bounded image-
            # domain transform search instead of trusting inconsistent spectra.
            selected = int(np.argmax(selected_confidences))
            scales = [scales[selected]]
            rotations = [rotations[selected]]
            selected_confidences = [0.0]
            axis_separable = False
    if not scales:
        scales = [nominal_scale]
        rotations = [0.0]
        selected_confidences = [0.0]

    pitch_x_fallback = (
        _axis_pitch_from_2d_peaks(search_norm, "x")
        if sea_x[1] < _MIN_PHASE_CONFIDENCE
        else None
    )
    pitch_y_fallback = (
        _axis_pitch_from_2d_peaks(search_norm, "y")
        if sea_y[1] < _MIN_PHASE_CONFIDENCE
        else None
    )
    pitch_x = (
        float(1.0 / max(abs(sx[0]), 1e-9))
        if sea_x[1] >= _MIN_PHASE_CONFIDENCE
        else float(pitch_x_fallback or 10.0)
    )
    pitch_y = (
        float(1.0 / max(abs(sy[1]), 1e-9))
        if sea_y[1] >= _MIN_PHASE_CONFIDENCE
        else float(pitch_y_fallback or 15.0)
    )
    return PeriodicTransformEstimate(
        scale=float(np.median(scales)),
        rotation_deg=float(np.median(rotations)),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        x_vector=(float(sx[0]), float(sx[1])),
        y_vector=(float(sy[0]), float(sy[1])),
        axis_separable=axis_separable,
        confidence=float(np.clip(np.mean(selected_confidences), 0.0, 1.0)),
    )


def periodic_difference_channels(
    image: np.ndarray,
    pitch_x: float,
    pitch_y: float,
    evidence_channel: str = "structural",
) -> dict[str, np.ndarray]:
    """Cancel the periodic backbone by differencing one lattice period apart.

    Both channels use the same symmetric crop.  Consequently a correlation-map
    coordinate still denotes the original template's top-left coordinate.
    """
    if evidence_channel == "structural":
        evidence = structural_channel(image)
    elif evidence_channel == "gradient":
        evidence = gradient_channel(image)
    elif evidence_channel == "raw":
        evidence = robust_float(image).astype(np.float32)
    else:
        raise ValueError(f"unknown periodic evidence channel: {evidence_channel}")
    shift_x = max(1, int(round(pitch_x)))
    shift_y = max(1, int(round(pitch_y)))
    if evidence.shape[1] <= 2 * shift_x + 4 or evidence.shape[0] <= 2 * shift_y + 4:
        raise ValueError("image is too small for its estimated lattice period")
    center = evidence[shift_y:-shift_y, shift_x:-shift_x]
    residual_x = center - 0.5 * (
        evidence[shift_y:-shift_y, 2 * shift_x :]
        + evidence[shift_y:-shift_y, : -2 * shift_x]
    )
    residual_y = center - 0.5 * (
        evidence[2 * shift_y :, shift_x:-shift_x]
        + evidence[: -2 * shift_y, shift_x:-shift_x]
    )
    return {
        "period_x": np.asarray(residual_x, dtype=np.float32),
        "period_y": np.asarray(residual_y, dtype=np.float32),
    }


def structural_channel(image: np.ndarray) -> np.ndarray:
    normalized = robust_float(image)
    low = ndimage.gaussian_filter(normalized, 0.65, mode="reflect")
    background = ndimage.gaussian_filter(normalized, 10.0, mode="reflect")
    return robust_float(low - 0.25 * background).astype(np.float32)


def gradient_channel(image: np.ndarray) -> np.ndarray:
    normalized = robust_float(image)
    gx = ndimage.sobel(normalized, axis=1, mode="reflect") / 8.0
    gy = ndimage.sobel(normalized, axis=0, mode="reflect") / 8.0
    return robust_float(np.hypot(gx, gy)).astype(np.float32)


def residual_channel(image: np.ndarray, max_peaks: int = 20, sigma_bins: float = 1.8) -> np.ndarray:
    """Suppress smooth reciprocal-lattice bands without a sharp ringing mask."""
    normalized = robust_float(image)
    peaks = detect_reciprocal_peaks(normalized, max_peaks=max_peaks)
    if not peaks:
        return structural_channel(normalized)
    transformed = np.fft.fftshift(np.fft.fft2(normalized))
    periodic_mask = smooth_notch_mask(normalized.shape, peaks, sigma_bins)
    # Preserve DC in neither component; slow illumination is not discriminative.
    residual_fft = transformed * (1.0 - periodic_mask)
    residual = np.real(np.fft.ifft2(np.fft.ifftshift(residual_fft)))
    residual = residual - ndimage.gaussian_filter(residual, 5.0, mode="reflect")
    if float(np.std(residual)) < 1e-6:
        return np.zeros_like(normalized, dtype=np.float32)
    # ZNCC performs its own affine normalization. Keeping the residual in this
    # scale also makes suppression measurable in unit tests and diagnostics.
    return residual.astype(np.float32)


def build_channels(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "structural": structural_channel(image),
        "gradient": gradient_channel(image),
        "residual": residual_channel(image),
    }
