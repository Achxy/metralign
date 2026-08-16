import numpy as np

from drift_sense.lattice import estimate_lattice
from drift_sense.representations import residual_channel


def test_lattice_estimator_finds_sinusoid_pitch():
    y, x = np.mgrid[:256, :256]
    image = np.cos(2 * np.pi * x / 16) + 0.7 * np.cos(2 * np.pi * y / 20)
    estimate = estimate_lattice(image)
    pitches = [estimate.pitch_primary, estimate.pitch_secondary]
    assert estimate.confidence > 0.3
    assert any(p is not None and abs(p - 16) < 1 for p in pitches)


def test_residual_suppresses_periodic_energy():
    y, x = np.mgrid[:256, :256]
    periodic = np.cos(2 * np.pi * x / 16) + np.cos(2 * np.pi * y / 20)
    residual = residual_channel(periodic)
    assert residual.std() < periodic.std()
