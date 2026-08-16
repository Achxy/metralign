import numpy as np

from drift_sense.distortions import (
    AcquisitionParameters,
    apply_acquisition,
    scan_line_shifts,
    secondary_electron_edge_response,
)


def test_every_acquisition_component_can_be_disabled():
    image = np.linspace(0.1, 0.9, 32 * 32, dtype=np.float32).reshape(32, 32)
    params = AcquisitionParameters(
        edge_strength=0,
        psf_sigma=0,
        poisson_peak=0,
        gaussian_sigma=0,
        gain=1,
        offset=0,
        slow_intensity_drift=0,
        scan_jitter_sigma=0,
    )
    result, shifts = apply_acquisition(image, params, np.random.default_rng(1))
    assert np.allclose(result, image)
    assert np.count_nonzero(shifts) == 0


def test_edge_response_changes_only_nonuniform_image():
    uniform = np.full((32, 32), 0.5, dtype=np.float32)
    assert np.allclose(secondary_electron_edge_response(uniform, 0.4), uniform)
    edge = uniform.copy()
    edge[:, 16:] = 0.8
    assert not np.allclose(secondary_electron_edge_response(edge, 0.4), edge)


def test_independent_noise_seeds_change_capture():
    image = np.full((32, 32), 0.4, dtype=np.float32)
    params = AcquisitionParameters(scan_jitter_sigma=0)
    first, _ = apply_acquisition(image, params, np.random.default_rng(2))
    second, _ = apply_acquisition(image, params, np.random.default_rng(3))
    assert not np.array_equal(first, second)


def _only(**overrides):
    values = dict(
        edge_strength=0,
        psf_sigma=0,
        poisson_peak=0,
        gaussian_sigma=0,
        gain=1,
        offset=0,
        slow_intensity_drift=0,
        scan_jitter_sigma=0,
    )
    values.update(overrides)
    return AcquisitionParameters(**values)


def test_blur_gain_offset_and_drift_are_independently_switchable():
    y, x = np.mgrid[:40, :40]
    image = (((x // 5 + y // 7) % 2) * 0.55 + 0.2).astype(np.float32)
    baseline, _ = apply_acquisition(image, _only(), np.random.default_rng(4))
    variants = [
        _only(psf_sigma=1.1),
        _only(gain=0.8),
        _only(offset=0.07),
        _only(slow_intensity_drift=0.15),
    ]
    for index, params in enumerate(variants):
        result, _ = apply_acquisition(image, params, np.random.default_rng(10 + index))
        assert not np.allclose(result, baseline)


def test_poisson_and_gaussian_noise_are_independently_switchable():
    image = np.full((48, 48), 0.45, dtype=np.float32)
    poisson, _ = apply_acquisition(image, _only(poisson_peak=80), np.random.default_rng(8))
    gaussian, _ = apply_acquisition(image, _only(gaussian_sigma=0.03), np.random.default_rng(8))
    assert poisson.std() > 0
    assert gaussian.std() > 0
    assert not np.array_equal(poisson, gaussian)


def test_scan_jitter_is_correlated_and_center_anchored():
    shifts = scan_line_shifts(101, 0.8, 5.0, np.random.default_rng(22))
    assert shifts[50] == 0
    assert np.std(shifts) > 0
    # Gaussian filtering makes adjacent rows more similar than distant rows.
    adjacent = np.mean(np.abs(np.diff(shifts)))
    distant = np.mean(np.abs(shifts[10:] - shifts[:-10]))
    assert adjacent < distant
