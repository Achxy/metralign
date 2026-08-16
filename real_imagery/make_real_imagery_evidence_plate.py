#!/usr/bin/env python3
"""Build a report-bound, success-focused SEM/TEM evidence contact sheet."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from real_imagery.make_real_imagery_plate import (
    cross,
    digest_file,
    font,
    paste_contained,
    sem_images,
    tem_images,
)
from real_imagery.protocol import digital_crop_pair


SUCCESS_THRESHOLD_PX = 5.0
EVIDENCE_SPECIFICATIONS = (
    ("carinthia", None, "median", "Production-wafer SEM (Carinthia)", "real SEM search capture"),
    ("carinthia", None, "p95", "Production-wafer SEM (Carinthia)", "real SEM search capture"),
    ("sem", "ordered", "median", "Ordered real SEM", "real SEM search capture"),
    ("sem", "disordered", "p95", "Disordered real SEM", "real SEM search capture"),
    ("tem", "calibration_grid", "median", "Registered calibration-grid TEM", "registered Low capture"),
    ("tem", "kidney", "p95", "Registered kidney TEM", "registered Low capture"),
)


def select_success_record(
    records: list[dict],
    *,
    group_field: str | None,
    group: str | None,
    statistic: str,
) -> tuple[dict, float]:
    """Select the completed successful row nearest a declared group statistic."""
    eligible = [
        row
        for row in records
        if row.get("method") == "full"
        and (group_field is None or row.get(group_field) == group)
        and isinstance(row.get("error_px"), (int, float))
        and float(row["error_px"]) <= SUCCESS_THRESHOLD_PX
        and isinstance(row.get("prediction"), list)
    ]
    if not eligible:
        raise ValueError(f"no successful full records for {group_field}={group}")
    values = np.asarray([float(row["error_px"]) for row in eligible], dtype=np.float64)
    if statistic == "median":
        target = float(np.median(values))
    elif statistic == "p95":
        target = float(np.percentile(values, 95, method="linear"))
    else:
        raise ValueError(f"unknown selection statistic: {statistic}")
    selected = min(
        eligible,
        key=lambda row: (abs(float(row["error_px"]) - target), str(row["case_id"])),
    )
    return selected, target


def carinthia_images(record: dict, archive: ZipFile) -> tuple[np.ndarray, np.ndarray]:
    content = archive.read(record["member"])
    if sha256(content).hexdigest() != record["member_sha256"]:
        raise ValueError(f"Carinthia member hash differs from report: {record['member']}")
    with Image.open(BytesIO(content)) as image:
        if image.mode != "L":
            raise ValueError("Carinthia plate input must be native grayscale")
        search = np.array(image, copy=True)
    reference, _, _ = digital_crop_pair(
        search, tuple(record["construction"]["position_fraction"])
    )
    return reference, search


def _overlay_search(
    canvas: Image.Image,
    search: np.ndarray,
    bounds: tuple[int, int, int, int],
    record: dict,
) -> tuple[int, int, int, int]:
    search_box = paste_contained(canvas, search, bounds)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(search_box, outline="#222222", width=1)
    scale_x = (search_box[2] - search_box[0]) / search.shape[1]
    scale_y = (search_box[3] - search_box[1]) / search.shape[0]

    def transform(point: list[float]) -> tuple[float, float]:
        return (
            search_box[0] + float(point[0]) * scale_x,
            search_box[1] + float(point[1]) * scale_y,
        )

    crop_x, crop_y, crop_width, crop_height = record["construction"]["crop_box_xywh"]
    draw.rectangle(
        (
            search_box[0] + crop_x * scale_x,
            search_box[1] + crop_y * scale_y,
            search_box[0] + (crop_x + crop_width) * scale_x,
            search_box[1] + (crop_y + crop_height) * scale_y,
        ),
        outline="#0072b2",
        width=3,
    )
    truth = transform(record["ground_truth"])
    prediction = transform(record["prediction"])
    draw.line((*truth, *prediction), fill="#111111", width=2)
    cross(draw, truth, "#0072b2", radius=8)
    cross(draw, prediction, "#d55e00", radius=8)
    return search_box


def render_panel(
    canvas: Image.Image,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    letter: str,
    title: str,
    selection_label: str,
    reference: np.ndarray,
    search: np.ndarray,
    record: dict,
    search_label: str,
) -> None:
    x, y = origin
    width, height = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), letter, fill="#111111", font=font(24, bold=True))
    draw.text((x + 38, y), title, fill="#111111", font=font(20, bold=True))
    draw.text(
        (x + width, y + 2),
        f"{selection_label} · {float(record['error_px']):.2f} px",
        fill="#333333",
        font=font(16),
        anchor="ra",
    )
    draw.line((x, y + 37, x + width, y + 37), fill="#444444", width=1)

    reference_box = paste_contained(canvas, reference, (x, y + 55, x + 230, y + height - 65))
    draw.rectangle(reference_box, outline="#222222", width=1)
    search_box = _overlay_search(
        canvas,
        search,
        (x + 255, y + 55, x + width, y + height - 65),
        record,
    )
    draw.text(
        (reference_box[0], reference_box[3] + 7),
        "digital reference",
        fill="#444444",
        font=font(14),
    )
    draw.text(
        (search_box[0], search_box[3] + 7),
        search_label,
        fill="#444444",
        font=font(14),
    )
    fallback = record["diagnostics"]["pipeline_stages"].get("fallback")
    if fallback:
        draw.text(
            (x, y + height - 20),
            "Coordinate from declared baseline0 fallback; periodic residual unsupported at this query size.",
            fill="#444444",
            font=font(13),
        )


def _sidecar_record(
    track: str,
    group: str | None,
    statistic: str,
    target: float,
    row: dict,
) -> dict:
    result = {
        "track": track,
        "group": group,
        "case_id": row["case_id"],
        "method": row["method"],
        "selection": f"nearest_{statistic}_among_completed_error_le_{SUCCESS_THRESHOLD_PX:g}px",
        "selection_target_px": target,
        "ground_truth_xy": row["ground_truth"],
        "prediction_xy": row["prediction"],
        "error_px": row["error_px"],
        "crop_box_xywh": row["construction"]["crop_box_xywh"],
        "pipeline_stages": row["diagnostics"]["pipeline_stages"],
    }
    if track == "sem":
        result.update(
            {
                "dataset": row["dataset"],
                "area": row["area"],
                "source_sha256": row["source_sha256"],
            }
        )
    elif track == "carinthia":
        result.update(
            {
                "dataset": row["dataset"],
                "class_label": row["class_label"],
                "member": row["member"],
                "member_sha256": row["member_sha256"],
            }
        )
    else:
        result.update(
            {
                "gt_member": row["gt_member"],
                "gt_sha256": row["gt_sha256"],
                "low_member": row["low_member"],
                "low_sha256": row["low_sha256"],
            }
        )
    return result


def evidence_selections(sem_report: dict, tem_report: dict) -> list[tuple]:
    sem_rows = sem_report["digital_crop_self_consistency"]["records"]
    carinthia_rows = sem_report["carinthia_semiconductor_sem_self_consistency"]["records"]
    tem_rows = tem_report["records"]
    selected = []
    for track, group, statistic, title, search_label in EVIDENCE_SPECIFICATIONS:
        rows = (
            carinthia_rows
            if track == "carinthia"
            else sem_rows if track == "sem" else tem_rows
        )
        field = None if track == "carinthia" else "category" if track == "sem" else "sample"
        row, target = select_success_record(
            rows, group_field=field, group=group, statistic=statistic
        )
        selected.append((track, group, statistic, title, search_label, row, target))
    return selected


def verify_evidence_plate(
    sem_report_path: Path,
    tem_report_path: Path,
    plate_path: Path,
) -> dict:
    sidecar_path = plate_path.with_suffix(".json")
    sem_bytes = sem_report_path.read_bytes()
    tem_bytes = tem_report_path.read_bytes()
    sem_report = json.loads(sem_bytes)
    tem_report = json.loads(tem_bytes)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sem_sha256 = sha256(sem_bytes).hexdigest()
    tem_sha256 = sha256(tem_bytes).hexdigest()
    if sidecar["reports"]["sem"]["sha256"] != sem_sha256:
        raise ValueError("evidence plate SEM report hash mismatch")
    if sidecar["reports"]["tem"]["sha256"] != tem_sha256:
        raise ValueError("evidence plate TEM report hash mismatch")
    if sidecar["plate_sha256"] != digest_file(plate_path):
        raise ValueError("evidence plate PNG hash mismatch")
    with Image.open(plate_path) as image:
        if image.info.get("sem_report_sha256") != sem_sha256:
            raise ValueError("embedded SEM report hash mismatch")
        if image.info.get("tem_report_sha256") != tem_sha256:
            raise ValueError("embedded TEM report hash mismatch")
        if image.info.get("license") != "CC BY 4.0":
            raise ValueError("embedded evidence plate license mismatch")
    selected = evidence_selections(sem_report, tem_report)
    expected = [
        _sidecar_record(track, group, statistic, target, row)
        for track, group, statistic, _title, _label, row, target in selected
    ]
    if sidecar["selected_records"] != expected:
        raise ValueError("evidence plate representative selection mismatch")
    for row in expected:
        error = float(
            np.hypot(
                float(row["prediction_xy"][0]) - float(row["ground_truth_xy"][0]),
                float(row["prediction_xy"][1]) - float(row["ground_truth_xy"][1]),
            )
        )
        if not np.isclose(error, float(row["error_px"]), rtol=0.0, atol=1e-12):
            raise ValueError(f"evidence plate coordinate/error mismatch: {row['case_id']}")
    if len({row["case_id"] for row in expected}) != len(expected):
        raise ValueError("evidence plate contains duplicate cases")
    return {
        "verified": True,
        "plate_sha256": sidecar["plate_sha256"],
        "sem_report_sha256": sem_sha256,
        "tem_report_sha256": tem_sha256,
        "selected_case_count": len(expected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-report", type=Path, required=True)
    parser.add_argument("--tem-report", type=Path, required=True)
    parser.add_argument("--sem-data-dir", type=Path, required=True)
    parser.add_argument("--tem-archive", type=Path, required=True)
    parser.add_argument("--carinthia-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar_path = args.output.with_suffix(".json")
    if args.output.exists() or sidecar_path.exists():
        raise FileExistsError(f"refusing to replace existing evidence plate: {args.output}")

    sem_bytes = args.sem_report.read_bytes()
    tem_bytes = args.tem_report.read_bytes()
    sem_report = json.loads(sem_bytes)
    tem_report = json.loads(tem_bytes)
    if sem_report.get("protocol") != "real-sem-self-consistency-v1":
        raise ValueError("unexpected SEM report protocol")
    if tem_report.get("protocol") != "registered-real-minitem-crop-localization-v1":
        raise ValueError("unexpected TEM report protocol")
    archive_binding = tem_report["archive"]
    if args.tem_archive.stat().st_size != int(archive_binding["bytes"]):
        raise ValueError("TEM archive size differs from report")
    if digest_file(args.tem_archive) != archive_binding["sha256"]:
        raise ValueError("TEM archive SHA-256 differs from report")
    carinthia_section = sem_report.get("carinthia_semiconductor_sem_self_consistency")
    if not isinstance(carinthia_section, dict):
        raise ValueError("SEM report omits the Carinthia supplement")
    carinthia_binding = carinthia_section["archive"]
    if args.carinthia_archive.stat().st_size != int(carinthia_binding["bytes"]):
        raise ValueError("Carinthia archive size differs from report")
    if digest_file(args.carinthia_archive) != carinthia_binding["sha256"]:
        raise ValueError("Carinthia archive SHA-256 differs from report")

    selected = evidence_selections(sem_report, tem_report)

    sem_manifest = json.loads(
        (Path(__file__).resolve().parent / "sources.json").read_text(encoding="utf-8")
    )
    canvas = Image.new("RGB", (1800, 1780), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 38), "Measured localization on real microscopy", fill="#111111", font=font(36, bold=True))
    draw.text(
        (70, 88),
        "Mechanically selected successful cases · blue = registered/digital truth · orange = prediction",
        fill="#333333",
        font=font(19),
    )
    draw.line((70, 130, 1730, 130), fill="#222222", width=2)
    draw.line((900, 150, 900, 1660), fill="#b5b5b5", width=1)
    draw.line((70, 655, 1730, 655), fill="#b5b5b5", width=1)
    draw.line((70, 1165, 1730, 1165), fill="#b5b5b5", width=1)

    positions = (
        (70, 160),
        (930, 160),
        (70, 675),
        (930, 675),
        (70, 1185),
        (930, 1185),
    )
    with ZipFile(args.tem_archive) as archive, ZipFile(args.carinthia_archive) as carinthia_archive:
        for index, (selection, position) in enumerate(zip(selected, positions, strict=True)):
            track, group, statistic, title, search_label, row, _target = selection
            if track == "sem":
                reference, search = sem_images(row, args.sem_data_dir, sem_manifest)
            elif track == "carinthia":
                reference, search = carinthia_images(row, carinthia_archive)
            else:
                reference, search = tem_images(row, archive)
            render_panel(
                canvas,
                origin=position,
                size=(800, 455),
                letter=chr(ord("A") + index),
                title=title,
                selection_label=(
                    f"balanced-set {statistic.upper() if statistic == 'p95' else statistic}"
                    if track == "carinthia"
                    else "group median" if statistic == "median" else "nearest group P95"
                ),
                reference=reference,
                search=search,
                record=row,
                search_label=search_label,
            )

    draw.line((70, 1670, 1730, 1670), fill="#222222", width=1)
    draw.text(
        (70, 1687),
        "Sources: 10.5281/zenodo.10715190, 10.6084/m9.figshare.11783661.v1, 10.6084/m9.figshare.11783667.v1, 10.5281/zenodo.4113244 · CC BY 4.0",
        fill="#333333",
        font=font(15),
    )
    draw.text(
        (70, 1717),
        "Coordinates/errors are in source pixels; display transforms are visualization-only. Each fallback is disclosed in its panel.",
        fill="#333333",
        font=font(15),
    )

    sem_sha256 = sha256(sem_bytes).hexdigest()
    tem_sha256 = sha256(tem_bytes).hexdigest()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("sem_report_sha256", sem_sha256)
    metadata.add_text("tem_report_sha256", tem_sha256)
    metadata.add_text("license", "CC BY 4.0")
    metadata.add_text("selection", "group median / nearest group P95 among completed <=5 px cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, pnginfo=metadata, optimize=True)

    sidecar = {
        "schema_version": 1,
        "plate": args.output.name,
        "plate_sha256": digest_file(args.output),
        "success_threshold_px": SUCCESS_THRESHOLD_PX,
        "reports": {
            "sem": {"path": str(args.sem_report), "sha256": sem_sha256},
            "tem": {"path": str(args.tem_report), "sha256": tem_sha256},
        },
        "selection_policy": (
            "Prespecified group/statistic mapping; select the completed full-method row "
            "with error <=5 px nearest the group's median or NumPy-linear P95, with "
            "lexicographic case_id tie-breaking."
        ),
        "selected_records": [
            _sidecar_record(track, group, statistic, target, row)
            for track, group, statistic, _title, _label, row, target in selected
        ],
        "sources": {
            "carinthia_doi": "10.5281/zenodo.10715190",
            "sem_ordered_doi": "10.6084/m9.figshare.11783661.v1",
            "sem_disordered_doi": "10.6084/m9.figshare.11783667.v1",
            "tem_doi": "10.5281/zenodo.4113244",
            "license": "CC BY 4.0",
        },
        "claim_boundary": (
            "Carinthia and other SEM coordinates are digital-crop truth within one acquisition. TEM coordinates "
            "are digital-crop truth in publisher-registered Low/GT pairs; all shown TEM full "
            "requests used the declared baseline0 fallback because the periodic residual was "
            "unsupported at the transformed query size."
        ),
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verification = verify_evidence_plate(args.sem_report, args.tem_report, args.output)
    print(json.dumps({"artifact": sidecar, "verification": verification}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
