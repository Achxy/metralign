from __future__ import annotations

import json
from zipfile import ZipFile
from pathlib import Path

import numpy as np

from real_imagery.evaluate_real_imagery import (
    carinthia_selection,
    summarize_carinthia,
    summarize_digital,
    summarize_native,
)
from real_imagery.evaluate_registered_tem import discover_test_pairs, summarize
from real_imagery.make_real_imagery_evidence_plate import (
    carinthia_images,
    select_success_record,
)
from real_imagery.make_real_imagery_plate import tem_images
from real_imagery.protocol import digital_crop_pair, error_metrics, informative_crop_positions
from real_imagery.verify_report import verify_record_errors


ROOT = Path(__file__).resolve().parents[2]


def test_digital_crop_pair_has_exact_center_and_deterministic_pixels() -> None:
    search = np.arange(400 * 500, dtype=np.uint32).reshape(400, 500).astype(np.uint8)
    first, center, details = digital_crop_pair(search, (0.5, 0.5), nominal_scale=0.1)
    second, repeated_center, repeated_details = digital_crop_pair(
        search, (0.5, 0.5), nominal_scale=0.1
    )
    x0, y0, width, height = details["crop_box_xywh"]
    assert first.shape == search.shape
    assert np.array_equal(first, second)
    assert center == repeated_center
    assert details == repeated_details
    assert center == (x0 + (width - 1) / 2.0, y0 + (height - 1) / 2.0)
    assert 0 <= x0 < search.shape[1] - width
    assert 0 <= y0 < search.shape[0] - height


def test_short_registered_frame_uses_isotropic_nominal_crop() -> None:
    search = np.zeros((204, 929), dtype=np.uint8)
    _, _, details = digital_crop_pair(
        search, (0.5, 0.5), nominal_scale=0.1, minimum_crop_size=8
    )
    _, _, width, height = details["crop_box_xywh"]
    assert width == round(929 * 0.1)
    assert height == round(204 * 0.1)
    assert details["minimum_crop_size_px"] == 8


def test_informative_positions_are_deterministic_nonoverlapping_and_gt_only() -> None:
    image = np.zeros((240, 1000), dtype=np.uint16)
    rng = np.random.default_rng(7)
    image[:, 500:] = rng.integers(0, 4096, size=(240, 500), dtype=np.uint16)
    first = informative_crop_positions(image)
    second = informative_crop_positions(image.copy())
    assert first == second
    assert len(first) == 5
    assert all(item["center_xy"][0] >= 500 for item in first)
    for index, item in enumerate(first):
        ax, ay, aw, ah = item["crop_box_xywh"]
        for prior in first[:index]:
            bx, by, bw, bh = prior["crop_box_xywh"]
            assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay


def test_error_metrics_does_not_invent_statistics_for_empty_input() -> None:
    assert error_metrics([], thresholds=(1.0, 5.0)) == {"count": 0}


def test_summaries_surface_all_execution_errors() -> None:
    digital = [
        {
            "method": "full",
            "category": "ordered",
            "error_px": 2.0,
            "repeat_delta_px": 0.0,
            "diagnostics": {"runtime_ms": 2.0, "pipeline_stages": {}},
        },
        {"method": "full", "category": "disordered", "error": "failed"},
    ]
    digital_summary = summarize_digital(digital, ["full"])["full"]
    assert digital_summary["attempted_count"] == 2
    assert digital_summary["completed_count"] == 1
    assert digital_summary["error_count"] == 1

    native = [
        {
            "pair_id": "p1",
            "method": "full",
            "proxy_accepted": True,
            "proxy_disagreement_px": 3.0,
            "repeat_delta_px": 0.0,
            "prediction": [1.0, 2.0],
            "diagnostics": {"runtime_ms": 3.0, "pipeline_stages": {}},
        },
        {"pair_id": "p2", "method": "full", "proxy_accepted": True, "error": "failed"},
    ]
    native_summary = summarize_native(native, ["full"])["methods"]["full"]
    assert native_summary["attempted_count"] == 2
    assert native_summary["completed_count"] == 1
    assert native_summary["error_count"] == 1
    assert native_summary["fallback_count"] == 0
    assert native_summary["localizer_runtime_ms"]["total"] == 3.0


def test_registered_tem_summary_counts_fallbacks_and_runtime() -> None:
    rows = [
        {
            "method": "full",
            "sample": "kidney",
            "position_index": 0,
            "error_px": 0.5,
            "repeat_delta_px": 0.0,
            "diagnostics": {
                "runtime_ms": 4.0,
                "pipeline_stages": {"fallback": "baseline0_small_periodic_template"},
            },
        },
        {
            "method": "full",
            "sample": "kidney",
            "position_index": 1,
            "error_px": 1.5,
            "repeat_delta_px": 0.0,
            "diagnostics": {"runtime_ms": 6.0, "pipeline_stages": {}},
        },
    ]

    result = summarize(rows, ["full"])["full"]

    assert result["fallback_count"] == 1
    assert result["localizer_runtime_ms"]["count"] == 2
    assert result["localizer_runtime_ms"]["total"] == 10.0
    assert result["localizer_runtime_ms"]["median"] == 5.0

    carinthia_rows = [
        {**row, "class_label": str(index + 1)} for index, row in enumerate(rows)
    ]
    carinthia_result = summarize_carinthia(carinthia_rows, ["full"])["full"]
    assert carinthia_result["attempted_count"] == 2
    assert carinthia_result["fallback_count"] == 1
    assert set(carinthia_result["by_class_label"]) == {"1", "2"}


def test_sem_source_manifest_is_small_pinned_and_unambiguous() -> None:
    manifest = json.loads((ROOT / "real_imagery" / "sources.json").read_text())
    records = [
        (dataset["id"], area["area"], record)
        for dataset in manifest["datasets"]
        for area in dataset["areas"]
        for record in area["files"]
    ]
    assert len(records) == 18
    assert len({record[2]["file_id"] for record in records}) == 18
    assert {record[2]["magnification_k"] for record in records} == {50, 100, 200}
    assert all(len(record[2]["md5"]) == 32 for record in records)
    assert sum(record[2]["size"] for record in records) == 23_595_156


def test_paired_tem_source_is_pinned_and_claim_limited() -> None:
    manifest = json.loads(
        (ROOT / "real_imagery" / "paired_tem_source.json").read_text()
    )
    assert manifest["archive"] == {
        "name": "TrainTestVal.zip",
        "url": "https://zenodo.org/api/records/4113244/files/TrainTestVal.zip/content",
        "bytes": 629_335_440,
        "md5": "2128033df6437e5d9bcdc0d4796a7b94",
    }
    assert manifest["dataset"]["license"] == "CC BY 4.0"
    assert manifest["dataset"]["publisher_split"] == "Test"
    boundary = manifest["claim_boundary"].lower()
    assert "not microscope-stage ground truth" in boundary
    assert "not" in boundary and "cross-magnification" in boundary


def test_carinthia_source_is_pinned_balanced_and_claim_limited() -> None:
    manifest = json.loads((ROOT / "real_imagery" / "carinthia_source.json").read_text())
    assert manifest["archive"] == {
        "name": "data.zip",
        "url": "https://zenodo.org/api/records/10715190/files/data.zip/content",
        "bytes": 133_840_870,
        "md5": "457011cf9063e5a49751f33ea468309d",
        "sha256": "02436de8c2d6b0c7eabdcdcf133ab5f17e59a0e3de56ebffc4cb6b2acb771490",
    }
    assert manifest["dataset"]["record_image_count"] == 4591
    assert manifest["dataset"]["license"] == "CC BY 4.0"
    assert manifest["selection"]["images_per_class"] == 4
    assert "same-acquisition digital self-consistency" in manifest["claim_boundary"]


def test_carinthia_selection_is_balanced_path_bound_and_deterministic(tmp_path: Path) -> None:
    from hashlib import md5, sha256
    from io import BytesIO
    from PIL import Image

    archive_path = tmp_path / "carinthia.zip"
    csv_lines = ["image_path;file_name;label"]
    encoded_by_path = {}
    for label in range(1, 7):
        for index in range(4):
            member = f"data/images/label{label}-{index}.jpg"
            pixels = np.full((128, 128), label * 20 + index, dtype=np.uint8)
            encoded = BytesIO()
            Image.fromarray(pixels).save(encoded, format="JPEG")
            encoded_by_path[member] = encoded.getvalue()
            csv_lines.append(f"{member};label{label}-{index}.jpg;{label}")
    index_content = ("\n".join(csv_lines) + "\n").encode()
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("data/carinthia.csv", index_content)
        for member, content in encoded_by_path.items():
            archive.writestr(member, content)
    archive_content = archive_path.read_bytes()
    manifest = {
        "dataset": {"record_image_count": 24, "class_labels": list("123456")},
        "archive": {
            "name": "carinthia.zip",
            "bytes": len(archive_content),
            "md5": md5(archive_content).hexdigest(),
            "sha256": sha256(archive_content).hexdigest(),
        },
        "index": {
            "member": "data/carinthia.csv",
            "bytes": len(index_content),
            "sha256": sha256(index_content).hexdigest(),
            "delimiter": ";",
            "columns": ["image_path", "file_name", "label"],
        },
        "selection": {
            "images_per_class": 4,
            "positions": [[0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7]],
        },
    }

    first_binding, first = carinthia_selection(archive_path, manifest)
    second_binding, second = carinthia_selection(archive_path, manifest)

    assert first_binding == second_binding
    assert [{k: v for k, v in row.items() if k != "pixels"} for row in first] == [
        {k: v for k, v in row.items() if k != "pixels"} for row in second
    ]
    assert len(first) == 24
    assert {label: sum(row["class_label"] == label for row in first) for label in "123456"} == {
        label: 4 for label in "123456"
    }
    shown = {
        **first[0],
        "construction": {"position_fraction": first[0]["position_fraction"]},
    }
    with ZipFile(archive_path) as archive:
        reference, search = carinthia_images(shown, archive)
    assert reference.shape == search.shape == (128, 128)
    assert reference.dtype == search.dtype == np.uint8


def test_registered_tem_discovery_uses_only_test_gt_low_pairs(tmp_path: Path) -> None:
    archive_path = tmp_path / "fixture.zip"
    image = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256).astype(np.uint8)
    from io import BytesIO
    from PIL import Image

    encoded = BytesIO()
    Image.fromarray(image).save(encoded, format="TIFF")
    payload = encoded.getvalue()
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("TrainTestVal/Kidney/Test/GT/frame01.tif", payload)
        for index in range(5):
            archive.writestr(
                f"TrainTestVal/Kidney/Test/Low/frame01/lq{index}.tif", payload
            )
        archive.writestr("TrainTestVal/Kidney/Train/GT/ignored.tif", payload)
        archive.writestr("TrainTestVal/Kidney/Test/GT_hr/ignored.tif", payload)
    with ZipFile(archive_path) as archive:
        pairs = discover_test_pairs(archive)
    assert len(pairs) == 5
    assert {row["sample"] for row in pairs} == {"kidney"}
    assert [row["low_index_within_gt"] for row in pairs] == list(range(5))
    assert [row["position_index"] for row in pairs] == list(range(5))


def test_tem_plate_loader_preserves_native_uint16_pixels(tmp_path: Path) -> None:
    from hashlib import sha256
    from io import BytesIO
    from PIL import Image

    pixels = np.arange(96 * 128, dtype=np.uint16).reshape(96, 128) * 3
    encoded = BytesIO()
    Image.fromarray(pixels).save(encoded, format="TIFF")
    payload = encoded.getvalue()
    archive_path = tmp_path / "uint16.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("gt.tif", payload)
        archive.writestr("low.tif", payload)
    record = {
        "gt_member": "gt.tif",
        "low_member": "low.tif",
        "gt_sha256": sha256(payload).hexdigest(),
        "low_sha256": sha256(payload).hexdigest(),
        "position_fraction": [0.5, 0.5],
        "construction": {"minimum_crop_size_px": 8},
    }

    with ZipFile(archive_path) as archive:
        reference, search = tem_images(record, archive)

    assert reference.dtype == np.uint16
    assert search.dtype == np.uint16
    assert int(search.max()) > 255


def test_evidence_plate_selection_is_statistic_bound_and_deterministic() -> None:
    rows = [
        {
            "case_id": case_id,
            "method": "full",
            "category": "ordered",
            "error_px": error,
            "prediction": [1.0, 2.0],
        }
        for case_id, error in (("d", 4.0), ("c", 3.0), ("b", 2.0), ("a", 1.0))
    ]
    median, median_target = select_success_record(
        rows, group_field="category", group="ordered", statistic="median"
    )
    p95, p95_target = select_success_record(
        rows, group_field="category", group="ordered", statistic="p95"
    )

    assert median_target == 2.5
    assert median["case_id"] == "b"
    assert p95_target == 3.8499999999999996
    assert p95["case_id"] == "d"


def test_report_verifier_recomputes_errors_and_requires_full_configuration() -> None:
    configuration = {
        "method": "full",
        "nominal_scale": 0.1,
        "scale_range": 0.006,
        "rotation_range": 3.0,
        "enable_phase_calibration": True,
        "periodic_evidence_channel": "structural",
        "enable_spatial_residual": True,
        "enable_lattice_grouping": True,
        "enable_ambiguity_rule": True,
        "subpixel_refinement": "parabolic",
    }
    record = {
        "case_id": "case",
        "prediction": [4.0, 6.0],
        "ground_truth": [1.0, 2.0],
        "error_px": 5.0,
        "repeat_delta_px": 0.0,
        "configuration": configuration,
    }
    verify_record_errors([record])
    record["error_px"] = 4.9
    import pytest

    with pytest.raises(ValueError, match="error arithmetic mismatch"):
        verify_record_errors([record])
