import numpy as np

from drift_sense.ambiguity import choose_candidate
from drift_sense.candidates import top_k_candidates


def test_center_tie_break_scans_beyond_truncated_top_k():
    score_map = np.ones((81, 81), dtype=np.float32)
    candidates = top_k_candidates(score_map, k=4, nms_radius=3)

    decision = choose_candidate(
        candidates,
        score_map,
        search_shape=(100, 100),
        template_shape=(20, 20),
        absolute_margin=1e-6,
        nms_radius=3,
    )

    assert decision.ambiguous
    assert decision.tied_count == score_map.size
    assert (decision.candidate.x, decision.candidate.y) == (40, 40)


def test_reliable_basis_groups_near_ties_by_integer_lattice_offset():
    score_map = np.full((31, 36), -1.0, dtype=np.float32)
    for index, (x, y) in enumerate(((5, 5), (15, 5), (25, 5), (5, 17), (15, 17))):
        score_map[y, x] = 1.0 - 0.0005 * index
    candidates = top_k_candidates(score_map, k=5, nms_radius=2)

    decision = choose_candidate(
        candidates,
        score_map,
        search_shape=(40, 45),
        template_shape=(10, 10),
        nms_radius=2,
        real_basis=np.array([[10.0, 0.0], [0.0, 12.0]]),
        residual_evidence=0.1,
    )

    assert decision.ambiguous
    assert decision.lattice_grouped
    assert decision.lattice_group_count == 5
    assert decision.lattice_group_coverage == 1.0
    assert (decision.candidate.x, decision.candidate.y) == (15, 17)


def test_score_tie_requires_secondary_ambiguity_evidence():
    score_map = np.full((21, 21), -1.0, dtype=np.float32)
    score_map[4:7, 4:7] = 0.9995
    score_map[5, 5] = 1.0
    score_map[4:7, 14:17] = 0.9975
    score_map[5, 15] = 0.998
    candidates = top_k_candidates(score_map, k=2, nms_radius=2)

    resolved = choose_candidate(
        candidates,
        score_map,
        search_shape=(33, 33),
        template_shape=(5, 5),
        nms_radius=2,
        residual_evidence=0.8,
        transform_stability=0.9,
    )
    insufficient_residual = choose_candidate(
        candidates,
        score_map,
        search_shape=(33, 33),
        template_shape=(5, 5),
        nms_radius=2,
        residual_evidence=0.1,
        transform_stability=0.9,
    )

    assert resolved.score_tied and not resolved.secondary_evidence
    assert not resolved.ambiguous
    assert insufficient_residual.score_tied and insufficient_residual.secondary_evidence
    assert insufficient_residual.ambiguous
    assert insufficient_residual.candidate.x == 15
