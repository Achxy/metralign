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
        residual_evidence=0.0,
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


def test_symmetric_center_boundary_returns_an_actual_tied_maximum():
    score_map = np.full((22, 22), -1.0, dtype=np.float32)
    score_map[10, 6] = 1.0
    score_map[11, 15] = 0.999
    candidates = top_k_candidates(score_map, k=2, nms_radius=2)

    decision = choose_candidate(
        candidates,
        score_map,
        search_shape=(30, 30),
        template_shape=(9, 9),
        nms_radius=2,
        residual_evidence=0.01,
    )

    assert decision.ambiguous
    assert (decision.candidate.x, decision.candidate.y) in {(6, 10), (15, 11)}
    assert score_map[decision.candidate.y, decision.candidate.x] >= 0.999


def test_periodic_development_band_includes_closest_valid_peak():
    score_map = np.full((31, 31), -1.0, dtype=np.float32)
    score_map[4, 4] = 1.0
    score_map[15, 15] = 1.0 - 0.07129
    candidates = top_k_candidates(score_map, k=2, nms_radius=2)

    decision = choose_candidate(
        candidates,
        score_map,
        search_shape=(41, 41),
        template_shape=(11, 11),
        absolute_margin=0.075,
        nms_radius=2,
        residual_evidence=0.01,
    )

    assert decision.ambiguous
    assert (decision.candidate.x, decision.candidate.y) == (15, 15)


def test_user_prior_replaces_image_center_only_for_ambiguous_selection():
    score_map = np.full((31, 31), -1.0, dtype=np.float32)
    score_map[5, 5] = 1.0
    score_map[15, 15] = 1.0
    candidates = top_k_candidates(score_map, k=2, nms_radius=2)

    default = choose_candidate(
        candidates,
        score_map,
        search_shape=(41, 41),
        template_shape=(11, 11),
        residual_evidence=0.01,
        nms_radius=2,
    )
    with_prior = choose_candidate(
        candidates,
        score_map,
        search_shape=(41, 41),
        template_shape=(11, 11),
        residual_evidence=0.01,
        nms_radius=2,
        prior_center=(10.0, 10.0),
    )

    assert (default.candidate.x, default.candidate.y) == (15, 15)
    assert default.selection_prior_source == "image_center_default"
    assert (with_prior.candidate.x, with_prior.candidate.y) == (5, 5)
    assert with_prior.selection_prior_center == (10.0, 10.0)
    assert with_prior.selection_prior_source == "user_supplied"
    assert len(with_prior.hypotheses) == 2


def test_local_peak_perturbation_alone_does_not_authorize_center_fallback():
    score_map = np.full((31, 31), -1.0, dtype=np.float32)
    score_map[4:7, 4:7] = 0.98
    score_map[5, 5] = 1.0
    score_map[14:17, 14:17] = 0.979
    score_map[15, 15] = 0.999
    candidates = top_k_candidates(score_map, k=2, nms_radius=2)

    decision = choose_candidate(
        candidates,
        score_map,
        search_shape=(41, 41),
        template_shape=(11, 11),
        nms_radius=2,
    )

    assert decision.score_tied
    assert decision.local_perturbation_support
    assert not decision.transform_instability_support
    assert not decision.low_residual_support
    assert not decision.secondary_evidence
    assert not decision.ambiguous
    assert (decision.candidate.x, decision.candidate.y) == (5, 5)
