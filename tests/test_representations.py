import numpy as np

import drift_sense.representations as representations
from drift_sense.representations import (
    estimate_periodic_transform,
    periodic_difference_channels,
)
from drift_sense.spectral import SpectralPeak


def test_periodic_difference_cancels_exact_tiled_backbone():
    rng = np.random.default_rng(19)
    unit_cell = rng.normal(size=(8, 10)).astype(np.float32)
    image = np.tile(unit_cell, (10, 10))

    channels = periodic_difference_channels(image, pitch_x=10, pitch_y=8)

    # Ignore the structural channel's reflected Gaussian boundary support.
    assert float(np.max(np.abs(channels["period_x"][16:-16, 16:-16]))) < 0.002
    assert float(np.max(np.abs(channels["period_y"][16:-16, 16:-16]))) < 0.002


def test_periodic_difference_retains_location_specific_change():
    x = np.arange(100, dtype=np.float32)
    y = np.arange(96, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image = np.cos(2 * np.pi * xx / 10) + np.cos(2 * np.pi * yy / 8)
    image[43:48, 52:57] += 2.0

    channels = periodic_difference_channels(image, pitch_x=10, pitch_y=8)

    assert float(np.max(np.abs(channels["period_x"]))) > 0.2
    assert float(np.max(np.abs(channels["period_y"]))) > 0.2


def test_phase_transform_recovers_scale_of_orthogonal_lattice():
    y, x = np.indices((512, 512), dtype=np.float32)
    reference = np.cos(2 * np.pi * x / 50) + 0.6 * np.cos(2 * np.pi * y / 80)
    search = np.cos(2 * np.pi * x / 5) + 0.6 * np.cos(2 * np.pi * y / 8)

    estimate = estimate_periodic_transform(reference, search, nominal_scale=0.1)

    assert estimate.axis_separable
    assert abs(estimate.scale - 0.1) < 0.002
    assert abs(estimate.rotation_deg) < 0.1
    assert abs(estimate.pitch_x - 5.0) < 0.2
    assert abs(estimate.pitch_y - 8.0) < 0.2


def test_low_confidence_axis_cannot_override_reliable_transform(monkeypatch):
    phase_results = iter(
        [
            (np.array([0.00775, 0.00199]), 0.48),
            (np.array([0.00012, 0.00665]), 0.98),
            (np.array([0.03954, -0.00067]), 0.04),
            (np.array([-0.00089, 0.06773]), 0.99),
        ]
    )
    monkeypatch.setattr(representations, "_phase_lattice_vector", lambda *args: next(phase_results))
    monkeypatch.setattr(
        representations,
        "detect_reciprocal_peaks",
        lambda *args, **kwargs: [SpectralPeak(fy=0.001, fx=1 / 12.7, power=20.0, symmetry=1.0)],
    )

    image = np.zeros((64, 64), dtype=np.float32)
    estimate = estimate_periodic_transform(image, image, nominal_scale=0.1)

    assert not estimate.axis_separable
    assert abs(estimate.scale - 0.0982) < 0.002
    assert abs(estimate.rotation_deg + 1.8) < 0.3
    assert abs(estimate.pitch_x - 12.7) < 0.2
