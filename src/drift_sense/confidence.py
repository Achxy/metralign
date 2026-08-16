"""Transparent confidence summary for internal evaluation."""

from __future__ import annotations

import math


def confidence_from_evidence(best_score: float, margin: float, curvature: float) -> float:
    score_term = 1.0 / (1.0 + math.exp(-8.0 * (best_score - 0.25)))
    margin_term = 1.0 - math.exp(-max(margin, 0.0) * 30.0)
    curvature_term = 1.0 - math.exp(-max(curvature, 0.0) * 10.0)
    return float(0.55 * score_term + 0.30 * margin_term + 0.15 * curvature_term)
