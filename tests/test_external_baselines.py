import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from aggregate_external_comparison import main as aggregate_main
from compare_external_baselines import _archived_metralign_reference, summarize
from drift_sense.external_baselines import BaselineConfig, METHODS, run_baseline


def test_opencv_template_adapter_returns_reference_center():
    rng = np.random.default_rng(41)
    reference = rng.normal(size=(128, 128)).astype(np.float32)
    template = cv2.resize(reference, (32, 32), interpolation=cv2.INTER_AREA)
    search = rng.normal(scale=0.01, size=(128, 128)).astype(np.float32)
    search[50:82, 40:72] = template

    result = run_baseline(
        "opencv_template",
        reference,
        search,
        BaselineConfig(nominal_scale=0.25),
    )

    assert result.status == "resolved"
    assert result.x == pytest.approx(55.5)
    assert result.y == pytest.approx(65.5)
    assert result.score == pytest.approx(1.0)


def test_scikit_image_adapter_is_independent_and_subpixel_capable():
    pytest.importorskip("skimage")
    rng = np.random.default_rng(52)
    reference = rng.normal(size=(128, 128)).astype(np.float32)
    template = cv2.resize(reference, (32, 32), interpolation=cv2.INTER_AREA)
    search = rng.normal(scale=0.01, size=(128, 128)).astype(np.float32)
    search[47:79, 61:93] = template

    result = run_baseline(
        "skimage_template_phase",
        reference,
        search,
        BaselineConfig(nominal_scale=0.25),
    )

    assert result.status == "resolved"
    assert result.x == pytest.approx(76.5, abs=0.1)
    assert result.y == pytest.approx(62.5, abs=0.1)


def test_comparison_metrics_count_unresolved_estimates_against_success():
    rows = [
        {"error": 0.25, "runtime_ms": 1.0},
        {"error": None, "runtime_ms": 2.0},
    ]

    metrics = summarize(rows)

    assert metrics["coverage"] == 0.5
    assert metrics["success_le_0.5px"] == 0.5
    assert metrics["resolved_success_le_0.5px"] == 1.0


def test_external_baseline_rejects_unknown_method():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    with pytest.raises(ValueError, match="unknown"):
        run_baseline("not-a-method", image, image)


def _method_result() -> dict:
    rows = [
        {"id": "a", "error": 0.25, "runtime_ms": 1.0},
        {"id": "b", "error": 1.25, "runtime_ms": 2.0},
    ]
    return {"metrics": summarize(rows), "samples": rows}


def _comparison_report(dataset_methods: dict[str, list[str]], archived: dict | None = None) -> dict:
    return {
        "schema_version": 2,
        "study": "external-registration-baseline-comparison",
        "method_metadata": {
            method: {"category": "fixture"}
            for methods in dataset_methods.values()
            for method in methods
        },
        "external_software": {},
        "datasets": {
            dataset: {
                "manifest": f"fixtures/{dataset}/manifest.jsonl",
                "artifact_binding": {
                    "dataset_sha256": f"dataset-{dataset}",
                    "manifest_sha256": f"manifest-{dataset}",
                },
                "dataset_record_count": 2,
                "evaluated_record_count": 2,
                "archived_metralign": archived,
                "methods": {method: _method_result() for method in methods},
            }
            for dataset, methods in dataset_methods.items()
        },
    }


def test_archived_metralign_reference_is_compact_portable_and_hash_bound(
    tmp_path: Path,
):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    source = {
        "manifest_sha256": "manifest",
        "artifact_binding": {"git_commit": "abc123"},
        "methods": {
            "full": {
                "samples": [
                    {
                        "id": "a",
                        "architecture": "dram",
                        "suite": "suite",
                        "ground_truth": [1.0, 2.0],
                        "prediction": [1.1, 2.1],
                        "error": 0.14,
                        "runtime_ms": 3.0,
                    },
                    {
                        "id": "b",
                        "architecture": "finfet",
                        "suite": "suite",
                        "ground_truth": [3.0, 4.0],
                        "prediction": [3.2, 4.2],
                        "error": 0.28,
                        "runtime_ms": 4.0,
                    },
                ]
            }
        },
    }
    (report_dir / "suite.json").write_text(json.dumps(source), encoding="utf-8")

    reference = _archived_metralign_reference(
        report_dir, "suite", "manifest", ["b"]
    )

    assert reference is not None
    assert not Path(reference["source_report"]).is_absolute()
    assert reference["evaluated_record_count"] == 1
    assert [row["id"] for row in reference["samples"]] == ["b"]
    assert reference["metrics"] == summarize(reference["samples"])
    assert len(reference["report_sha256"]) == 64


def test_aggregate_rejects_a_method_missing_from_one_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    report = _comparison_report(
        {
            "complete": list(METHODS),
            "incomplete": list(METHODS[:-1]),
        }
    )
    (input_dir / "comparison.json").write_text(json.dumps(report), encoding="utf-8")

    result = aggregate_main(
        [
            "--input-dir",
            str(input_dir),
            "--label",
            "fixture",
            "--output",
            str(tmp_path / "aggregate.json"),
        ]
    )

    assert result == 2
    assert "method set mismatch" in capsys.readouterr().err


def test_aggregate_uses_embedded_archived_rows_without_source_path(
    tmp_path: Path,
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    archived_samples = [
        {"id": "a", "error": 0.1, "runtime_ms": 3.0},
        {"id": "b", "error": 0.2, "runtime_ms": 4.0},
    ]
    archived = {
        "source_report": "unavailable/source.json",
        "report_sha256": "a" * 64,
        "algorithm_git_commit": "abc123",
        "evaluated_record_count": 2,
        "metrics": summarize(archived_samples),
        "samples": archived_samples,
    }
    report = _comparison_report({"suite": list(METHODS)}, archived=archived)
    (input_dir / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "aggregate.json"

    result = aggregate_main(
        [
            "--input-dir",
            str(input_dir),
            "--label",
            "fixture",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["schema_version"] == 2
    assert set(aggregate["expected_methods"]) == set(METHODS)
    assert aggregate["methods"]["metralign_archived"]["metrics"] == summarize(
        archived_samples
    )
