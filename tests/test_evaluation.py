from argparse import Namespace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import evaluate


def test_metrics_use_inclusive_euclidean_thresholds_and_linear_percentiles():
    rows = [
        {
            "error": error,
            "runtime_ms": runtime,
            "image_io_ms": 1.0,
            "localizer_runtime_ms": runtime - 0.25,
            "sample_wall_ms": runtime + 1.0,
        }
        for error, runtime in zip((0.5, 1.0, 2.0, 3.0, 5.0, 6.0), (10, 20, 30, 40, 50, 60))
    ]

    result = evaluate.metrics(rows)

    assert result["success_le_0.5px"] == pytest.approx(1 / 6)
    assert result["success_le_1px"] == pytest.approx(2 / 6)
    assert result["success_le_2px"] == pytest.approx(3 / 6)
    assert result["success_le_3px"] == pytest.approx(4 / 6)
    assert result["success_le_5px"] == pytest.approx(5 / 6)
    assert result["failure_gt_5px_count"] == 1
    assert result["median_error_px"] == 2.5
    assert result["p95_error_px"] == 5.75
    assert result["mean_runtime_ms"] == 35.0
    assert result["mean_image_io_ms"] == 1.0
    assert result["mean_localizer_runtime_ms"] == 34.75


def test_metrics_reject_empty_or_nonfinite_input():
    with pytest.raises(ValueError, match="empty"):
        evaluate.metrics([])
    with pytest.raises(ValueError, match="non-finite"):
        evaluate.metrics([{"error": math.inf, "runtime_ms": 1.0}])


def test_evaluator_reports_observed_and_internal_timing_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    image = np.zeros((32, 32), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "reference.png")
    Image.fromarray(image).save(tmp_path / "search.png")
    record = {
        "id": "sample",
        "architecture": "dram",
        "suite": "iid",
        "difficulty": "medium",
        "seed": 7,
        "reference": "reference.png",
        "search": "search.png",
        "center_x": 4.0,
        "center_y": 5.0,
    }

    prediction = SimpleNamespace(x=4.0, y=5.0, runtime_ms=0.25)
    prediction.to_dict = lambda: {"runtime_ms": 0.25}
    observed = {}

    def fake_localize(reference, search, config):
        observed["config"] = config
        return prediction

    monkeypatch.setattr(evaluate, "localize", fake_localize)
    args = Namespace(
        top_k=1,
        scale_range=0.0,
        rotation_range=0.0,
        phase_calibration=False,
        evidence_channel="gradient",
        spatial_residual=False,
        lattice_grouping=False,
        ambiguity_rule=False,
        subpixel_refinement="none",
        quiet=True,
    )

    result = evaluate.evaluate_method(tmp_path, [record], "baseline0", args)
    row = result["samples"][0]

    assert row["error"] == 0.0
    assert row["runtime_ms"] == row["inference_wall_ms"]
    assert row["localizer_runtime_ms"] == 0.25
    assert row["image_io_ms"] >= 0.0
    assert row["sample_wall_ms"] >= row["runtime_ms"]
    assert result["metrics"]["all"]["success_le_0.5px"] == 1.0
    assert observed["config"].enable_phase_calibration is False
    assert observed["config"].periodic_evidence_channel == "gradient"
    assert observed["config"].enable_spatial_residual is False
    assert observed["config"].enable_lattice_grouping is False
    assert observed["config"].enable_ambiguity_rule is False
    assert observed["config"].subpixel_refinement == "none"


def test_evaluator_pipeline_cli_defaults_and_overrides():
    defaults = evaluate.parse_args(["--data-dir", "dataset"])
    assert evaluate.pipeline_configuration(defaults) == {
        "enable_phase_calibration": True,
        "periodic_evidence_channel": "structural",
        "enable_spatial_residual": True,
        "enable_lattice_grouping": True,
        "enable_ambiguity_rule": True,
        "subpixel_refinement": "parabolic",
    }

    ablated = evaluate.parse_args(
        [
            "--data-dir",
            "dataset",
            "--no-phase-calibration",
            "--evidence-channel",
            "raw",
            "--no-spatial-residual",
            "--no-lattice-grouping",
            "--no-ambiguity-rule",
            "--subpixel-refinement",
            "dft",
        ]
    )
    assert evaluate.pipeline_configuration(ablated) == {
        "enable_phase_calibration": False,
        "periodic_evidence_channel": "raw",
        "enable_spatial_residual": False,
        "enable_lattice_grouping": False,
        "enable_ambiguity_rule": False,
        "subpixel_refinement": "dft",
    }


def test_manifest_validation_rejects_duplicate_ids(tmp_path: Path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "image.png")
    record = {
        "id": "same",
        "architecture": "dram",
        "suite": "iid",
        "reference": "image.png",
        "search": "image.png",
        "center_x": 1.0,
        "center_y": 1.0,
        "actual_scale": 0.1,
        "rotation_deg": 0.0,
        "search_geometry": {"width": 8, "height": 8},
    }
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.validate_records(tmp_path, [record, dict(record)])


def test_manifest_validation_requires_failure_taxonomy_fields(tmp_path: Path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "image.png")
    record = {
        "id": "sample",
        "architecture": "dram",
        "suite": "iid",
        "reference": "image.png",
        "search": "image.png",
        "center_x": 1.0,
        "center_y": 1.0,
    }

    with pytest.raises(ValueError, match="actual_scale, rotation_deg, search_geometry"):
        evaluate.validate_records(tmp_path, [record])


def test_artifact_binding_detects_input_image_tampering(tmp_path: Path):
    reference = tmp_path / "reference.png"
    search = tmp_path / "search.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(reference)
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(search)
    record = {
        "id": "sample",
        "architecture": "dram",
        "suite": "iid",
        "reference": reference.name,
        "search": search.name,
        "center_x": 1.0,
        "center_y": 1.0,
        "actual_scale": 0.1,
        "rotation_deg": 0.0,
        "search_geometry": {"width": 8, "height": 8},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    binding = evaluate.build_artifact_binding(
        manifest,
        [record],
        Path(evaluate.__file__).resolve().parent,
    )
    report = {"manifest_sha256": binding["manifest_sha256"], "artifact_binding": binding}

    evaluate.verify_artifact_binding(
        report,
        manifest,
        [record],
        Path(evaluate.__file__).resolve().parent,
        require_current_code=True,
    )
    Image.fromarray(np.full((8, 8), 17, dtype=np.uint8)).save(search)
    with pytest.raises(ValueError, match="input_images_sha256|dataset_sha256"):
        evaluate.verify_artifact_binding(
            report,
            manifest,
            [record],
            Path(evaluate.__file__).resolve().parent,
            require_current_code=True,
        )


def test_artifact_binding_rejects_wrong_implementation_fingerprint(tmp_path: Path):
    image = tmp_path / "image.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image)
    record = {
        "id": "sample",
        "architecture": "dram",
        "suite": "iid",
        "reference": image.name,
        "search": image.name,
        "center_x": 1.0,
        "center_y": 1.0,
        "actual_scale": 0.1,
        "rotation_deg": 0.0,
        "search_geometry": {"width": 8, "height": 8},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    repo_root = Path(evaluate.__file__).resolve().parent
    binding = evaluate.build_artifact_binding(manifest, [record], repo_root)
    binding["implementation_sha256"] = "0" * 64
    report = {"manifest_sha256": binding["manifest_sha256"], "artifact_binding": binding}

    with pytest.raises(ValueError, match="implementation_sha256"):
        evaluate.verify_artifact_binding(
            report,
            manifest,
            [record],
            repo_root,
            require_current_code=True,
        )
