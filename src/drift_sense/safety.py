"""Decision support for absolute-site ambiguity.

The localizer's existing ``confidence`` describes the sharpness and strength of
the selected match.  It must not be interpreted as confidence that the correct
member of a repeated lattice family was selected.  This module keeps those two
questions separate and provides a versioned review policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


POLICY_ID = "metralign-absolute-site-review-v1"
# Mechanically selected from the archived 700-pair development split by rounding
# the maximum ambiguous residual evidence (0.1382027417) outward to the next 0.05.
# The reporting split was not used to compute this threshold.
AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD = 0.15
# Archived development minimum (all cases) is 0.9544444166; round down to 0.95.
TRANSFORM_STABILITY_REVIEW_THRESHOLD = 0.95


@dataclass(frozen=True)
class SafetyAssessment:
    """Machine-readable recommendation; scores are diagnostics, not probabilities."""

    policy_id: str
    status: str
    review_recommended: bool
    conservative_abstention_recommended: bool
    absolute_site_confidence: float
    reasons: tuple[str, ...]
    calibration_scope: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def assess_absolute_site(
    *,
    match_confidence: float,
    ambiguous: bool,
    residual_evidence: float,
    transform_stability: float,
) -> SafetyAssessment:
    """Assess whether a forced absolute-site coordinate should be reviewed.

    ``review_recommended`` is the selective, development-derived policy.
    ``conservative_abstention_recommended`` is true for every detected tie and
    is appropriate when a wrong absolute lattice site has high consequence.
    """
    values = (match_confidence, residual_evidence, transform_stability)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("safety evidence must be finite")

    reasons: list[str] = []
    if ambiguous and residual_evidence >= AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD:
        reasons.append("ambiguous_high_residual")
    if ambiguous and transform_stability < TRANSFORM_STABILITY_REVIEW_THRESHOLD:
        reasons.append("ambiguous_transform_instability")

    review = bool(reasons)
    if review:
        status = "review"
    elif ambiguous:
        status = "ambiguous"
    else:
        status = "resolved"

    # A tied lattice family contains no evidence for a unique absolute member,
    # even when the selected local peak is sharp.  Preserve the ordinary match
    # score only for resolved cases; do not present this value as a probability.
    absolute_site_confidence = 0.0 if ambiguous else float(
        min(1.0, max(0.0, match_confidence))
    )
    return SafetyAssessment(
        policy_id=POLICY_ID,
        status=status,
        review_recommended=review,
        conservative_abstention_recommended=bool(ambiguous),
        absolute_site_confidence=absolute_site_confidence,
        reasons=tuple(reasons),
        calibration_scope=(
            "thresholds derived from the archived 700-pair development split; "
            "diagnostic score is not an empirical probability"
        ),
    )
