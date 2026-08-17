from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "video/tools/export_evidence.py"
EVIDENCE = ROOT / "video/evidence/exported"


def load_exporter():
    spec = importlib.util.spec_from_file_location("metralign_film_evidence_exporter", EXPORTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_film_evidence_bundle_is_hash_bound_and_resolvable() -> None:
    exporter = load_exporter()
    index = exporter.verify_export(EVIDENCE)

    required_pointers = {
        "pair_count",
        "suite_count",
        "within_1px_count",
        "outside_1px_count",
        "within_1px_rate",
        "primary_threshold_px",
        "median_error_px",
        "p95_error_px",
        "mean_runtime_ms",
        "failure_gt_5px_count",
    }
    assert required_pointers <= set(index["metrics"]["frozen"])
    frozen = index["metrics"]["frozen"]
    assert frozen["outside_1px_count"] == frozen["pair_count"] - frozen[
        "within_1px_count"
    ]
    assert frozen["outside_1px_count_binding"]["display_value"] == frozen[
        "outside_1px_count"
    ]
    assert isinstance(index["metrics"]["frozen"]["primary_threshold_px"], int)
    assert isinstance(
        index["metrics"]["independent_renderer"]["primary_threshold_px"], int
    )
    for family in ("frozen", "independent_renderer"):
        threshold = index["metrics"][family]["primary_threshold_px"]
        binding = index["metrics"][family]["primary_threshold_px_binding"]
        assert binding["display_value"] == threshold
        assert binding["exact_value"] == float(threshold)
    assert index["metrics"]["external_baselines"][
        "classic_best_within_1px_rate_binding"
    ]["display_value"] == index["metrics"]["external_baselines"][
        "classic_best_within_1px_rate"
    ]
    external = index["metrics"]["external_baselines"]
    assert external["classic_adapter_count_binding"]["display_value"] == external[
        "classic_adapter_count"
    ]
    assert external["classic_best_name_binding"]["display_value"] == external[
        "classic_best_name"
    ]
    assert external["xfeat_coverage_rate_binding"]["display_value"] == external[
        "xfeat_coverage_rate"
    ]
    assert external["xfeat_within_5px_rate_binding"]["display_value"] == external[
        "xfeat_within_5px_rate"
    ]

    for asset in index["assets"].values():
        assert asset["file"].startswith("video/evidence/exported/")
        assert (ROOT / asset["file"]).is_file()
        assert len(asset["sha256"]) == 64
        assert asset["source_keys"]

    for sample_name in ("success_iid", "failure_scan"):
        sample = index["samples"][sample_name]
        assert len(sample["ground_truth"]) == 2
        assert len(sample["prediction"]) == 2
        assert isinstance(sample["review_recommended"], bool)
        assert "ambiguity_flag" in sample["diagnostics"]
        assert "tied_count" in sample["diagnostics"]


def test_exported_arrays_capture_real_pipeline_values_and_live_output() -> None:
    index = json.loads((EVIDENCE / "evidence_index.json").read_text(encoding="utf-8"))
    score = np.load(EVIDENCE / "candidate_score_map.npy", allow_pickle=False)
    residual_reference = np.load(
        EVIDENCE / "period_difference_reference.npy", allow_pickle=False
    )
    residual_search = np.load(
        EVIDENCE / "period_difference_search_crop.npy", allow_pickle=False
    )
    assert score.ndim == 2 and float(np.ptp(score)) > 0.0
    assert residual_reference.shape == residual_search.shape
    assert float(np.ptp(residual_reference)) > 0.0
    assert float(np.ptp(residual_search)) > 0.0

    phase = json.loads((EVIDENCE / "phase_transform.json").read_text(encoding="utf-8"))
    success = index["samples"]["success_iid"]
    assert phase["estimate"]["scale"] == success["diagnostics"]["selected_scale"]
    assert phase["estimate"]["rotation_deg"] == success["diagnostics"][
        "selected_rotation_deg"
    ]
    assert phase["estimate"]["confidence"] == success["diagnostics"][
        "spectral_confidence"
    ]
    for capture in ("reference", "search"):
        fft_asset = index["assets"][f"fft_{capture}.png"]
        assert fft_asset["marker_count_including_conjugates"] == 2 * len(
            phase["reciprocal_peaks"][capture]
        )
        assert fft_asset["peak_markers"] == phase["reciprocal_peaks"][capture]

    candidates = json.loads((EVIDENCE / "candidates.json").read_text(encoding="utf-8"))
    baseline = candidates["baseline0"]
    assert baseline == success["baseline0"]
    assert baseline == index["assets"]["baseline_candidate_overlay.png"]["baseline0"]
    score = np.load(EVIDENCE / "baseline_score_map.npy", allow_pickle=False)
    peak_y, peak_x = np.unravel_index(int(np.argmax(score)), score.shape)
    assert baseline["integer_argmax_top_left_xy"] == [int(peak_x), int(peak_y)]
    expected_error = float(
        np.hypot(
            baseline["prediction_xy"][0] - baseline["ground_truth_xy"][0],
            baseline["prediction_xy"][1] - baseline["ground_truth_xy"][1],
        )
    )
    assert baseline["error_px"] == expected_error

    live = index["samples"]["live_inference"]
    assert live["return_code"] == 0
    assert [float(value) for value in live["stdout"].split()] == [
        round(float(value), 6) for value in success["prediction"]
    ]
    sanitized = json.loads(
        (EVIDENCE / "live_stderr_sanitized.json").read_text(encoding="utf-8")
    )
    assert "runtime_ms" not in sanitized
    assert sanitized["x"] == success["prediction"][0]
    assert sanitized["y"] == success["prediction"][1]

    for name in (
        "fft_reference.png",
        "fft_search.png",
        "baseline_score_map.png",
        "candidate_score_map.png",
        "external_comparison_table.png",
        "terminal_capture.png",
    ):
        with Image.open(EVIDENCE / name) as image:
            pixels = np.asarray(image)
        assert pixels.size > 0 and float(np.ptp(pixels)) > 0.0


def test_display_metrics_recompute_from_bound_reports(tmp_path: Path) -> None:
    exporter = load_exporter()
    writer = exporter.Writer(tmp_path)
    sources: dict[str, object] = {}
    metrics, _ = exporter.frozen_metrics(writer, sources)
    metrics.update(exporter.comparison_metrics(sources))
    canonical_index = json.loads(
        (EVIDENCE / "evidence_index.json").read_text(encoding="utf-8")
    )
    canonical = canonical_index["metrics"]
    assert metrics["pair_count"] == canonical["frozen"]["pair_count"]
    assert metrics["within_1px_count"] == canonical["frozen"]["within_1px_count"]
    assert metrics["median_error_px_binding"] == canonical["frozen"][
        "median_error_px_binding"
    ]
    assert metrics["p95_error_px_binding"] == canonical["frozen"]["p95_error_px_binding"]
    assert metrics["external_baselines"] == canonical["external_baselines"]
    assert metrics["independent_renderer"] == canonical["independent_renderer"]
    assert metrics["safety"] == canonical["safety"]

    plot_data = json.loads(
        (EVIDENCE / "benchmark_plot_data.json").read_text(encoding="utf-8")
    )
    distribution = canonical_index["assets"]["evaluation_error_distribution.png"]
    assert distribution["overflow_rule"] == "error values above 1 px enter the final bin"
    assert distribution["displayed_pair_count"] == plot_data["aggregate"]["pair_count"]
    assert distribution["displayed_median_error_px"] == round(
        plot_data["aggregate"]["median_error_px"], 3
    )
    assert distribution["displayed_p95_error_px"] == round(
        plot_data["aggregate"]["p95_error_px"], 3
    )

    suite_strip = canonical_index["assets"]["evaluation_suite_strip.png"]
    for displayed, source in zip(
        suite_strip["displayed_rows"], plot_data["suites"], strict=True
    ):
        assert displayed["suite"] == source["suite"]
        assert displayed["within_1px_count"] == source["within_1px_count"]
        assert displayed["pair_count"] == source["pair_count"]
        assert displayed["p95_error_px_exact"] == source["p95_error_px"]
        assert displayed["p95_error_px_display"] == round(source["p95_error_px"], 3)

    classic_source = json.loads(
        (ROOT / "results/comparisons/external-registration-frozen.json").read_text(
            encoding="utf-8"
        )
    )
    classic_methods = {
        name: record["metrics"]
        for name, record in classic_source["methods"].items()
        if name != "metralign_archived"
    }
    external = canonical["external_baselines"]
    assert external["classic_adapter_count"] == len(classic_methods)
    assert [row["method_id"] for row in external["classic_adapters"]] == list(
        classic_methods
    )
    for row in external["classic_adapters"]:
        exact = classic_methods[row["method_id"]]["success_le_1px"]
        assert row["within_1px_rate_exact_fraction"] == exact
        assert row["within_1px_rate_display"] == round(100.0 * exact, 2)

    xfeat_source = json.loads(
        (ROOT / "results/comparisons/xfeat-frozen-all1400-development.json").read_text(
            encoding="utf-8"
        )
    )
    xfeat = xfeat_source["pooled"]["metrics"]
    assert external["xfeat_population_count"] == xfeat["count"]
    assert external["xfeat_coverage_count"] == xfeat["resolved_count"]
    assert external["xfeat_coverage_rate"] == round(100.0 * xfeat["coverage"], 2)
    assert external["xfeat_within_5px_count"] == round(
        xfeat["count"] * xfeat["success_le_5px"]
    )
    table = canonical_index["assets"]["external_comparison_table.png"]
    assert table["classic_adapters"] == external["classic_adapters"]
    assert table["xfeat_development"]["claim_boundary"] == xfeat_source[
        "claim_boundary"
    ]
