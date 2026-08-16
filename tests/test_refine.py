import numpy as np

from drift_sense.refine import dft_peak, parabolic_peak


def test_local_dft_refinement_recovers_fractional_gaussian_peak():
    y, x = np.indices((15, 15), dtype=np.float64)
    expected_x, expected_y = 7.35, 6.72
    score_map = np.exp(-((x - expected_x) ** 2 + (y - expected_y) ** 2) / 2.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)

    dft_x, dft_y, dft_curvature = dft_peak(score_map, int(peak_x), int(peak_y))
    parab_x, parab_y, parab_curvature = parabolic_peak(score_map, int(peak_x), int(peak_y))

    assert np.hypot(dft_x - expected_x, dft_y - expected_y) < 0.08
    assert np.hypot(parab_x - expected_x, parab_y - expected_y) < 0.12
    assert dft_curvature == parab_curvature > 0.0
