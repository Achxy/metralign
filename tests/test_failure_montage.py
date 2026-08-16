import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import evaluate
import make_failure_montage


def test_representative_selection_balances_failure_categories():
    rows = [
        {"id": "a1", "error": 20.0, "failure_category": "A"},
        {"id": "a2", "error": 19.0, "failure_category": "A"},
        {"id": "b1", "error": 18.0, "failure_category": "B"},
        {"id": "c1", "error": 17.0, "failure_category": "C"},
        {"id": "ok", "error": 5.0, "failure_category": None},
    ]

    selected = make_failure_montage.select_failures(rows, 5.0, 3)

    assert {row["failure_category"] for row in selected} == {"A", "B", "C"}
    assert all(row["error"] > 5.0 for row in selected)


def test_montage_cli_renders_only_evaluated_failure(tmp_path: Path):
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    Image.fromarray(image).save(tmp_path / "reference.png")
    Image.fromarray(image).save(tmp_path / "search.png")
    record = {
        "id": "sample",
        "architecture": "dram",
        "suite": "scan_distortion",
        "seed": 101,
        "reference": "reference.png",
        "search": "search.png",
        "center_x": 20.0,
        "center_y": 20.0,
        "actual_scale": 0.1,
        "rotation_deg": 0.0,
        "search_geometry": {"width": 64, "height": 64},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    binding = evaluate.build_artifact_binding(
        manifest,
        [record],
        Path(evaluate.__file__).resolve().parent,
    )
    report = {
        "manifest": str(manifest),
        "manifest_sha256": binding["manifest_sha256"],
        "artifact_binding": binding,
        "methods": {
            "full": {
                "samples": [
                    {
                        "id": "sample",
                        "architecture": "dram",
                        "suite": "scan_distortion",
                        "seed": 101,
                        "ground_truth": [20.0, 20.0],
                        "prediction": [40.0, 40.0],
                        "error": math.hypot(20.0, 20.0),
                        "failure_category": "scan distortion",
                    }
                ]
            }
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "montage.png"

    assert make_failure_montage.main(
        ["--report", str(report_path), "--output", str(output)]
    ) == 0
    with Image.open(output) as montage:
        assert montage.width == 440
        assert montage.height > 440

    report["methods"]["full"]["samples"][0]["ground_truth"][0] += 1e-13
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="ground truth disagrees"):
        make_failure_montage.main(
            ["--report", str(report_path), "--output", str(tmp_path / "bad-gt.png")]
        )

    report["methods"]["full"]["samples"][0]["ground_truth"] = [20.0, 20.0]
    report["methods"]["full"]["samples"][0]["error"] += 1.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="error is inconsistent"):
        make_failure_montage.main(
            ["--report", str(report_path), "--output", str(tmp_path / "bad-error.png")]
        )

    report["methods"]["full"]["samples"][0]["error"] = math.hypot(20.0, 20.0)
    report["manifest_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit, match="top-level manifest hash"):
        make_failure_montage.main(
            ["--report", str(report_path), "--output", str(tmp_path / "bad-manifest.png")]
        )

    report["manifest_sha256"] = binding["manifest_sha256"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(tmp_path / "search.png")
    with pytest.raises(SystemExit, match="input_images_sha256|dataset_sha256"):
        make_failure_montage.main(
            ["--report", str(report_path), "--output", str(tmp_path / "bad-image.png")]
        )
