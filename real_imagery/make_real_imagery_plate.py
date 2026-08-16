#!/usr/bin/env python3
"""Build a report-bound raster plate from real microscopy and measured overlays."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from real_imagery.protocol import digest_file, digital_crop_pair, read_sem_content


HERE = Path(__file__).resolve().parent
SEM_SOURCE = HERE / "sources.json"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def completed_full_records(report: dict) -> list[dict]:
    if report["protocol"] == "real-sem-self-consistency-v1":
        records = report["digital_crop_self_consistency"]["records"]
    elif report["protocol"] == "registered-real-minitem-crop-localization-v1":
        records = report["records"]
    else:
        raise ValueError(f"unsupported report protocol: {report.get('protocol')}")
    return [row for row in records if row["method"] == "full" and "error_px" in row]


def representative_records(report: dict) -> list[dict]:
    records = completed_full_records(report)
    group_key = "category" if report["protocol"] == "real-sem-self-consistency-v1" else "sample"
    selected = []
    for group in sorted({row[group_key] for row in records}):
        rows = sorted((row for row in records if row[group_key] == group), key=lambda row: row["error_px"])
        if not rows:
            continue
        selected.extend([rows[len(rows) // 2], rows[-1]])
    if len(selected) != 4:
        raise ValueError(f"expected two non-empty groups and four representatives, got {len(selected)}")
    return selected


def sem_images(record: dict, data_dir: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    dataset = next(item for item in manifest["datasets"] if item["id"] == record["dataset"])
    area = next(item for item in dataset["areas"] if item["area"] == record["area"])
    source = next(item for item in area["files"] if item["magnification_k"] == 50)
    path = data_dir / dataset["id"] / area["area"] / source["name"]
    if digest_file(path) != record["source_sha256"]:
        raise ValueError(f"source hash differs from report: {path}")
    search = read_sem_content(path, manifest["content_crop"])
    reference, _, _ = digital_crop_pair(search, tuple(record["construction"]["position_fraction"]))
    return reference, search


def tem_images(record: dict, archive: ZipFile) -> tuple[np.ndarray, np.ndarray]:
    gt_content = archive.read(record["gt_member"])
    low_content = archive.read(record["low_member"])
    import hashlib

    if hashlib.sha256(gt_content).hexdigest() != record["gt_sha256"]:
        raise ValueError(f"GT member hash differs from report: {record['gt_member']}")
    if hashlib.sha256(low_content).hexdigest() != record["low_sha256"]:
        raise ValueError(f"Low member hash differs from report: {record['low_member']}")
    with Image.open(BytesIO(gt_content)) as image:
        gt = np.array(image, copy=True)
    with Image.open(BytesIO(low_content)) as image:
        low = np.array(image, copy=True)
    if gt.ndim != 2 or low.ndim != 2:
        raise ValueError("TEM plate inputs must be single-channel images")
    reference, _, _ = digital_crop_pair(
        gt,
        tuple(record["position_fraction"]),
        minimum_crop_size=int(record["construction"]["minimum_crop_size_px"]),
    )
    return reference, low


def cross(draw: ImageDraw.ImageDraw, point: tuple[float, float], color: str, radius: int = 9) -> None:
    x, y = point
    draw.line((x - radius, y, x + radius, y), fill=color, width=3)
    draw.line((x, y - radius, x, y + radius), fill=color, width=3)


def paste_contained(
    canvas: Image.Image, pixels: np.ndarray, bounds: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Paste without changing the microscopy image's aspect ratio."""
    x0, y0, x1, y1 = bounds
    maximum_width = x1 - x0
    maximum_height = y1 - y0
    scale = min(maximum_width / pixels.shape[1], maximum_height / pixels.shape[0])
    width = max(1, int(round(pixels.shape[1] * scale)))
    height = max(1, int(round(pixels.shape[0] * scale)))
    left = x0 + (maximum_width - width) // 2
    top = y0
    display = display_uint8(pixels)
    image = Image.fromarray(display).resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, (left, top))
    return left, top, left + width, top + height


def display_uint8(pixels: np.ndarray) -> np.ndarray:
    """Deterministic robust display window; evaluation retains native values."""
    if pixels.dtype == np.uint8:
        return pixels
    values = np.asarray(pixels, dtype=np.float64)
    low, high = np.percentile(values, (0.5, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (np.clip(values, low, high) - low) * (255.0 / (high - low))
    return np.rint(scaled).astype(np.uint8)


def render_card(
    canvas: Image.Image,
    origin: tuple[int, int],
    size: tuple[int, int],
    reference: np.ndarray,
    search: np.ndarray,
    record: dict,
    label: str,
    panel_letter: str,
) -> None:
    x, y = origin
    width, height = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y + 2), panel_letter, fill="#111111", font=font(24, bold=True))
    draw.text((x + 40, y + 2), label, fill="#111111", font=font(20, bold=True))
    draw.text(
        (x + width, y + 4),
        f"error {record['error_px']:.2f} px",
        fill="#333333",
        font=font(17),
        anchor="ra",
    )

    draw.line((x, y + 38, x + width, y + 38), fill="#222222", width=2)
    ref_box = (x, y + 54, x + 238, y + height - 36)
    search_box = (x + 260, y + 54, x + width, y + height - 36)
    ref_box = paste_contained(canvas, reference, ref_box)
    search_box = paste_contained(canvas, search, search_box)
    draw.rectangle(ref_box, outline="#111111", width=1)
    draw.rectangle(search_box, outline="#111111", width=1)
    draw.text((ref_box[0], ref_box[3] + 7), "reference", fill="#333333", font=font(15))
    draw.text((search_box[0], search_box[3] + 7), "real search capture", fill="#333333", font=font(15))

    sx = (search_box[2] - search_box[0]) / search.shape[1]
    sy = (search_box[3] - search_box[1]) / search.shape[0]
    transform = lambda point: (search_box[0] + point[0] * sx, search_box[1] + point[1] * sy)
    truth = transform(record["ground_truth"])
    prediction = transform(record["prediction"])
    crop_x, crop_y, crop_w, crop_h = record["construction"]["crop_box_xywh"]
    crop_box = (
        search_box[0] + crop_x * sx,
        search_box[1] + crop_y * sy,
        search_box[0] + (crop_x + crop_w) * sx,
        search_box[1] + (crop_y + crop_h) * sy,
    )
    draw.rectangle(crop_box, outline="#38d6c6", width=3)
    draw.line((*truth, *prediction), fill="#ffffff", width=2)
    cross(draw, truth, "#38d6c6")
    cross(draw, prediction, "#ffb347")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sem-data-dir", type=Path)
    source.add_argument("--tem-archive", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to replace existing plate artifact: {args.output}")

    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    selected = representative_records(report)
    if args.sem_data_dir and report["protocol"] != "real-sem-self-consistency-v1":
        raise ValueError("SEM data directory provided for a non-SEM report")
    if args.tem_archive and report["protocol"] != "registered-real-minitem-crop-localization-v1":
        raise ValueError("TEM archive provided for a non-TEM report")
    if args.tem_archive:
        archive_binding = report["archive"]
        if args.tem_archive.stat().st_size != archive_binding["bytes"]:
            raise ValueError("TEM archive size differs from report")
        if digest_file(args.tem_archive) != archive_binding["sha256"]:
            raise ValueError("TEM archive SHA-256 differs from report")

    canvas = Image.new("RGB", (1800, 1240), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title = "Registered real TEM pairs" if args.tem_archive else "Real SEM crop self-consistency"
    draw.text((70, 42), title, fill="#111111", font=font(38, bold=True))
    draw.text(
        (70, 92),
        "Measured full-pipeline overlays · teal = registered/digital truth · amber = prediction",
        fill="#333333",
        font=font(20),
    )

    sem_manifest = json.loads(SEM_SOURCE.read_text()) if args.sem_data_dir else None
    archive = ZipFile(args.tem_archive) if args.tem_archive else None
    try:
        positions = ((70, 145), (930, 145), (70, 665), (930, 665))
        for index, (record, position) in enumerate(zip(selected, positions, strict=True)):
            if archive:
                reference, search = tem_images(record, archive)
                group = record["sample"].replace("_", " ")
            else:
                reference, search = sem_images(record, args.sem_data_dir, sem_manifest)
                group = record["category"]
            rank = "median" if index % 2 == 0 else "largest error"
            render_card(
                canvas,
                position,
                (800, 475),
                reference,
                search,
                record,
                f"{group} · {rank}",
                chr(ord("A") + index),
            )
    finally:
        if archive:
            archive.close()

    doi = report["source"]["doi"] if args.tem_archive else ", ".join(item["doi"] for item in report["sources"])
    draw.text(
        (70, 1190),
        f"Source: {doi} · CC BY 4.0 · displays robust-windowed/resized; errors use source pixels · report-bound selection",
        fill="#444444",
        font=font(16),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_sha256 = sha256_bytes(report_bytes)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("report_sha256", report_sha256)
    metadata.add_text("protocol", report["protocol"])
    metadata.add_text("source_doi", doi)
    metadata.add_text("license", "CC BY 4.0")
    metadata.add_text(
        "description",
        "Real microscopy pixels with measured coordinate overlays; representative selection is defined in the JSON sidecar.",
    )
    canvas.save(args.output, pnginfo=metadata, optimize=True)
    sidecar = {
        "schema_version": 1,
        "plate": args.output.name,
        "plate_sha256": digest_file(args.output),
        "report": args.report.name,
        "report_sha256": report_sha256,
        "protocol": report["protocol"],
        "selection": "median and maximum full-pipeline error within each of two prespecified source groups",
        "selected_records": [
            {
                "case_id": row["case_id"],
                "method": row["method"],
                "error_px": row["error_px"],
            }
            for row in selected
        ],
        "source_doi": doi,
        "license": "CC BY 4.0",
    }
    sidecar_path = args.output.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sidecar, indent=2, sort_keys=True))
    return 0


def sha256_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
