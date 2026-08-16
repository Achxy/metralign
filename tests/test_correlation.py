import numpy as np

from drift_sense.correlation import balanced_residual_score_map, candidate_supported_peak


def test_balanced_fusion_recovers_match_when_one_direction_is_corrupted():
    period_x = np.zeros((3, 3), dtype=np.float32)
    period_y = np.zeros((3, 3), dtype=np.float32)
    period_x[1, 1], period_y[1, 1] = 0.90, 0.30
    period_x[0, 0], period_y[0, 0] = 0.35, 0.35

    fused, support = balanced_residual_score_map(period_x, period_y)

    assert np.unravel_index(int(np.argmax(fused)), fused.shape) == (1, 1)
    assert support[1, 1] == np.float32(0.30)
    assert support[0, 0] == np.float32(0.35)


def test_candidate_local_support_rejects_one_channel_spurious_peak():
    period_x = np.zeros((3, 3), dtype=np.float32)
    period_y = np.zeros((3, 3), dtype=np.float32)
    period_x[1, 1], period_y[1, 1] = 1.0, -0.20
    period_x[0, 0], period_y[0, 0] = 0.30, 0.30

    fused, support = balanced_residual_score_map(period_x, period_y)
    peak = candidate_supported_peak(fused, support, minimum_support=0.25)
    assert support[1, 1] < 0.0
    assert np.unravel_index(int(np.argmax(fused)), fused.shape) == (1, 1)
    assert peak is None
