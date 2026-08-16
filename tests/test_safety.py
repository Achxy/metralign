from pathlib import Path

import pytest

from audit_safety_policy import (
    _compact_rows,
    _mechanical_development_threshold,
    _mechanical_transform_threshold,
)

from drift_sense.safety import (
    AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD,
    assess_absolute_site,
)


def test_selective_review_policy_separates_match_and_absolute_site_confidence():
    assessment = assess_absolute_site(
        match_confidence=0.99,
        ambiguous=True,
        residual_evidence=AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD,
        transform_stability=0.98,
    )

    assert assessment.status == "review"
    assert assessment.review_recommended
    assert assessment.conservative_abstention_recommended
    assert assessment.absolute_site_confidence == 0.0
    assert assessment.reasons == ("ambiguous_high_residual",)


def test_resolved_site_retains_bounded_match_confidence():
    assessment = assess_absolute_site(
        match_confidence=1.4,
        ambiguous=False,
        residual_evidence=0.8,
        transform_stability=0.99,
    )

    assert assessment.status == "resolved"
    assert not assessment.review_recommended
    assert not assessment.conservative_abstention_recommended
    assert assessment.absolute_site_confidence == 1.0


def test_safety_assessment_rejects_nonfinite_evidence():
    with pytest.raises(ValueError, match="finite"):
        assess_absolute_site(
            match_confidence=float("nan"),
            ambiguous=False,
            residual_evidence=0.8,
            transform_stability=0.99,
        )


def test_checked_in_development_diagnostics_reproduce_policy_bounds():
    rows, source_hashes = _compact_rows(
        Path("evidence/safety/development-diagnostics.json")
    )

    assert len(rows) == 700
    assert len(source_hashes) == 7
    assert _mechanical_development_threshold(rows) == (
        0.1382027417421341,
        AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD,
    )
    assert _mechanical_transform_threshold(rows) == (
        0.9544444166429555,
        0.95,
    )
