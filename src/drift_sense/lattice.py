"""Robust reciprocal-lattice summaries and transform priors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .spectral import SpectralPeak, detect_reciprocal_peaks


@dataclass(frozen=True)
class LatticeEstimate:
    peaks: tuple[SpectralPeak, ...]
    pitch_primary: float | None
    pitch_secondary: float | None
    orientation_deg: float | None
    confidence: float


def estimate_lattice(image: np.ndarray) -> LatticeEstimate:
    peaks = detect_reciprocal_peaks(image)
    if not peaks:
        return LatticeEstimate((), None, None, None, 0.0)
    # Prefer fundamental peaks in the expected band to their brighter harmonics.
    fundamentals = [p for p in peaks if 5.0 <= p.pitch <= 28.0]
    if not fundamentals:
        fundamentals = list(peaks)
    fundamentals.sort(key=lambda p: (-p.symmetry, p.frequency, -p.power))
    first = fundamentals[0]
    angle1 = math.degrees(math.atan2(first.fy, first.fx))
    second = None
    for peak in fundamentals[1:]:
        angle2 = math.degrees(math.atan2(peak.fy, peak.fx))
        separation = abs(((angle2 - angle1 + 90.0) % 180.0) - 90.0)
        if separation > 28.0:
            second = peak
            break
    confidence = first.symmetry * min(1.0, len(peaks) / 6.0)
    if second is not None:
        confidence = min(1.0, 0.5 * (confidence + second.symmetry))
    return LatticeEstimate(
        tuple(peaks),
        first.pitch,
        second.pitch if second else None,
        angle1,
        float(confidence),
    )


def estimate_relative_transform(
    reference_at_nominal_scale: np.ndarray, search: np.ndarray
) -> tuple[float, float, float]:
    """Return scale correction, rotation in degrees, and confidence.

    The estimate is deliberately bounded by the caller. A poor or harmonically
    inconsistent spectrum therefore cannot silently override the nominal prior.
    """
    ref = estimate_lattice(reference_at_nominal_scale)
    sea = estimate_lattice(search)
    confidence = min(ref.confidence, sea.confidence)
    if (
        confidence < 0.35
        or ref.pitch_primary is None
        or sea.pitch_primary is None
        or ref.orientation_deg is None
        or sea.orientation_deg is None
    ):
        return 1.0, 0.0, 0.0
    correction = sea.pitch_primary / ref.pitch_primary
    rotation = ((ref.orientation_deg - sea.orientation_deg + 90.0) % 180.0) - 90.0
    if not (0.94 <= correction <= 1.06) or abs(rotation) > 4.0:
        return 1.0, 0.0, confidence * 0.2
    return float(correction), float(rotation), confidence


def reciprocal_to_real_basis(estimate: LatticeEstimate) -> np.ndarray | None:
    if len(estimate.peaks) < 2:
        return None
    vectors = []
    first = np.array([estimate.peaks[0].fx, estimate.peaks[0].fy])
    vectors.append(first)
    for peak in estimate.peaks[1:]:
        candidate = np.array([peak.fx, peak.fy])
        determinant = first[0] * candidate[1] - first[1] * candidate[0]
        if abs(determinant) > 0.001:
            vectors.append(candidate)
            break
    if len(vectors) != 2:
        return None
    reciprocal = np.vstack(vectors)
    try:
        return np.linalg.inv(reciprocal).T
    except np.linalg.LinAlgError:
        return None
