import json
from pathlib import Path

import numpy as np
import pytest

from compare_xfeat import select_records, summarize
from drift_sense.xfeat_baseline import (
    OFFICIAL_CODE_BUNDLE_SHA256,
    OFFICIAL_COMMIT,
    OFFICIAL_WEIGHTS_SHA256,
    XFeatBaselineConfig,
    estimate_reference_center,
    prepare_xfeat_inputs,
)


def test_locked_protocol_matches_adapter_identity():
    protocol = json.loads(
        Path("evidence/external/xfeat-predeclared-protocol.json").read_text(
            encoding="utf-8"
        )
    )
    upstream = protocol["upstream"]
    assert upstream["official_commit"] == OFFICIAL_COMMIT
    assert upstream["inference_code_bundle_sha256"] == OFFICIAL_CODE_BUNDLE_SHA256
    assert upstream["weights_sha256"] == OFFICIAL_WEIGHTS_SHA256
    assert protocol["fixed_adapter"]["matcher_top_k"] == 8000


def test_xfeat_preprocessing_applies_only_nominal_scale_and_channel_repeat():
    reference = np.arange(100 * 120, dtype=np.uint8).reshape(100, 120)
    search = np.arange(200 * 220, dtype=np.uint8).reshape(200, 220)

    reference_nominal, search_rgb = prepare_xfeat_inputs(reference, search, 0.8)

    assert reference_nominal.shape == (80, 96, 3)
    assert search_rgb.shape == (200, 220, 3)
    np.testing.assert_array_equal(search_rgb[..., 0], search)
    np.testing.assert_array_equal(search_rgb[..., 1], search)
    np.testing.assert_array_equal(search_rgb[..., 2], search)


def test_official_homography_projection_recovers_reference_center():
    points0 = np.asarray(
        [[0, 0], [99, 0], [99, 99], [0, 99], [50, 30], [20, 70]],
        dtype=np.float32,
    )
    translation = np.asarray([125.25, 231.5], dtype=np.float32)
    points1 = points0 + translation

    result = estimate_reference_center(
        points0,
        points1,
        (100, 100),
        XFeatBaselineConfig(),
    )

    assert result.status == "resolved"
    assert result.x == pytest.approx(174.75, abs=1e-3)
    assert result.y == pytest.approx(281.0, abs=1e-3)
    assert result.diagnostics["inlier_count"] == 6


def test_official_homography_adapter_abstains_with_too_few_matches():
    points = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

    result = estimate_reference_center(
        points,
        points,
        (100, 100),
        XFeatBaselineConfig(),
    )

    assert result.status == "unresolved"
    assert result.diagnostics == {"reason": "insufficient_matches", "match_count": 3}


def test_hash_stratified_selection_is_balanced_and_order_preserving():
    records = [
        {
            "id": f"{index:03d}",
            "architecture": "dram" if index % 2 == 0 else "finfet",
        }
        for index in range(100)
    ]

    first = select_records(records, "a" * 64, "hash280")
    second = select_records(records, "a" * 64, "hash280")

    assert first == second
    assert len(first) == 40
    assert sum(row["architecture"] == "dram" for row in first) == 20
    assert sum(row["architecture"] == "finfet" for row in first) == 20
    original_positions = [records.index(row) for row in first]
    assert original_positions == sorted(original_positions)


def test_xfeat_summary_counts_unresolved_against_success_rate():
    rows = [
        {"error": 0.2, "runtime_ms": 10.0},
        {"error": None, "runtime_ms": 20.0},
    ]

    metrics = summarize(rows)

    assert metrics["coverage"] == 0.5
    assert metrics["success_le_0.5px"] == 0.5
    assert metrics["resolved_success_le_0.5px"] == 1.0
