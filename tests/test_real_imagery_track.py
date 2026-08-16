"""Expose the isolated real-imagery protocol tests to the default test suite."""

from real_imagery.tests.test_protocol import *  # noqa: F401,F403

import json
from pathlib import Path

import pytest

from real_imagery.verify_report import (
    _verify_exact_coverage,
    _verify_fallback_counts,
)


def test_registered_tem_dataset_id_names_the_actual_publisher_track() -> None:
    source = json.loads(Path("real_imagery/paired_tem_source.json").read_text())

    assert source["dataset"]["id"] == "wieslander_minitem_registered_test_v1"
    assert "boiko" not in source["dataset"]["id"]


def test_real_report_coverage_guard_requires_every_unique_case_method_pair() -> None:
    methods = {"full", "baseline0"}
    records = [
        {"case_id": case, "method": method}
        for case in ("case-a", "case-b")
        for method in methods
    ]

    _verify_exact_coverage(records, "case_id", {"case-a", "case-b"}, methods, "fixture")

    with pytest.raises(ValueError, match="coverage mismatch"):
        _verify_exact_coverage(
            records[:-1], "case_id", {"case-a", "case-b"}, methods, "fixture"
        )
    with pytest.raises(ValueError, match="duplicate"):
        _verify_exact_coverage(
            [*records, records[0]],
            "case_id",
            {"case-a", "case-b"},
            methods,
            "fixture",
        )


def test_real_report_fallback_count_is_recomputed_from_record_diagnostics() -> None:
    records = [
        {
            "case_id": "a",
            "method": "full",
            "prediction": [1.0, 2.0],
            "diagnostics": {
                "pipeline_stages": {
                    "fallback": "baseline0_small_periodic_template"
                }
            },
        }
    ]

    _verify_fallback_counts(records, {"full": {"fallback_count": 1}}, {"full"})
    with pytest.raises(ValueError, match="fallback count mismatch"):
        _verify_fallback_counts(
            records, {"full": {"fallback_count": 0}}, {"full"}
        )
