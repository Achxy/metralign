#!/usr/bin/env python3
"""Create a categorized montage from genuine evaluated failures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from drift_sense.dataset import load_manifest
from evaluate import validate_records, verify_artifact_binding


def _marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], radius: int) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)


def _panel(root: Path, record: dict, row: dict, width: int = 440) -> Image.Image:
    with Image.open(root / record["search"]) as source:
        search_source = source.convert("RGB")
    scale = width / search_source.width
    search = search_source.resize(
        (width, round(search_source.height * scale)), Image.Resampling.LANCZOS
    )
    draw = ImageDraw.Draw(search)
    gx, gy = (float(value) for value in row["ground_truth"])
    px, py = (float(value) for value in row["prediction"])
    radius = 7
    _marker(draw, gx * scale, gy * scale, (35, 210, 85), radius)
    draw.line(
        (px * scale - radius, py * scale, px * scale + radius, py * scale),
        fill=(245, 65, 60),
        width=3,
    )
    draw.line(
        (px * scale, py * scale - radius, px * scale, py * scale + radius),
        fill=(245, 65, 60),
        width=3,
    )

    footer_height = 92
    footer = Image.new("RGB", (width, footer_height), "white")
    footer_draw = ImageDraw.Draw(footer)
    with Image.open(root / record["reference"]) as source:
        reference = source.convert("RGB")
    reference.thumbnail((76, 76), Image.Resampling.LANCZOS)
    footer.paste(reference, (8, 8))
    category = row.get("failure_category") or "unclassified > threshold"
    seed = row.get("seed", record.get("seed", "not recorded"))
    caption = (
        f"{row['id']} | {row.get('architecture', record.get('architecture', '?'))} | "
        f"{row.get('suite', record.get('suite', '?'))}\n"
        f"error={float(row['error']):.3f}px | seed={seed}\n"
        f"{category}\n"
        "green circle=ground truth | red cross=prediction"
    )
    footer_draw.multiline_text(
        (94, 8), caption, fill="black", font=ImageFont.load_default(), spacing=4
    )
    panel = Image.new("RGB", (width, search.height + footer.height), "white")
    panel.paste(search, (0, 0))
    panel.paste(footer, (0, search.height))
    return panel


def select_failures(
    rows: list[dict],
    threshold: float,
    max_failures: int,
    categories: set[str] | None = None,
    selection: str = "representative",
) -> list[dict]:
    """Select only measured failures, balanced across categories by default."""
    failures = []
    for row in rows:
        error = float(row["error"])
        if not math.isfinite(error):
            raise ValueError(f"non-finite error for sample {row.get('id', '?')}")
        category = row.get("failure_category") or "unclassified > threshold"
        if error > threshold and (categories is None or category in categories):
            failures.append(row)
    failures.sort(key=lambda row: (-float(row["error"]), str(row["id"])))
    if selection == "worst":
        return failures[:max_failures]
    if selection != "representative":
        raise ValueError(f"unknown selection mode: {selection}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in failures:
        grouped[row.get("failure_category") or "unclassified > threshold"].append(row)
    selected = []
    category_names = sorted(
        grouped,
        key=lambda category: (-float(grouped[category][0]["error"]), category),
    )
    depth = 0
    while len(selected) < max_failures:
        added = False
        for category in category_names:
            if depth < len(grouped[category]):
                selected.append(grouped[category][depth])
                added = True
                if len(selected) == max_failures:
                    break
        if not added:
            break
        depth += 1
    return selected


def _validate_rows(rows: list[dict], records: dict[str, dict]) -> None:
    required = {"id", "ground_truth", "prediction", "error"}
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"report sample {index} is missing: {', '.join(sorted(missing))}")
        if row["id"] not in records:
            raise ValueError(f"report sample is absent from manifest: {row['id']}")
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate report sample id: {row['id']}")
        seen_ids.add(row["id"])
        if len(row["ground_truth"]) != 2 or len(row["prediction"]) != 2:
            raise ValueError(f"sample {row['id']} does not contain coordinate pairs")
        record = records[row["id"]]
        ground_truth = tuple(float(value) for value in row["ground_truth"])
        prediction = tuple(float(value) for value in row["prediction"])
        if not all(math.isfinite(value) for value in (*ground_truth, *prediction)):
            raise ValueError(f"sample {row['id']} contains non-finite coordinates")
        expected_ground_truth = (float(record["center_x"]), float(record["center_y"]))
        if ground_truth != expected_ground_truth:
            raise ValueError(f"report ground truth disagrees with manifest for sample {row['id']}")
        reported_error = float(row["error"])
        recomputed_error = math.hypot(
            prediction[0] - expected_ground_truth[0],
            prediction[1] - expected_ground_truth[1],
        )
        if not math.isfinite(reported_error) or not math.isclose(
            reported_error,
            recomputed_error,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(f"report error is inconsistent for sample {row['id']}")
        for field in ("architecture", "suite", "seed"):
            if field in row and row[field] != record.get(field):
                raise ValueError(f"report {field} disagrees with manifest for sample {row['id']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--method", default="full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="override the manifest location recorded in the report")
    parser.add_argument("--max-failures", type=int, default=6)
    parser.add_argument("--failure-threshold", type=float, default=5.0)
    parser.add_argument("--category", dest="categories", action="append")
    parser.add_argument("--selection", choices=("representative", "worst"), default="representative")
    args = parser.parse_args(argv)
    if args.max_failures < 1:
        parser.error("--max-failures must be positive")
    if args.failure_threshold < 0:
        parser.error("--failure-threshold must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if args.method not in report.get("methods", {}):
            raise ValueError(f"method {args.method!r} is absent from the report")
        rows = report["methods"][args.method]["samples"]
        manifest_path = args.data_dir if args.data_dir else Path(report["manifest"])
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.jsonl"
        root = manifest_path.parent
        manifest_records = load_manifest(manifest_path)
        validate_records(root, manifest_records)
        records = {record["id"]: record for record in manifest_records}
        if len(records) != len(manifest_records):
            raise ValueError("manifest contains duplicate sample ids")
        verify_artifact_binding(
            report,
            manifest_path,
            manifest_records,
            Path(__file__).resolve().parent,
            require_current_code=False,
        )
        _validate_rows(rows, records)
        selected = select_failures(
            rows,
            args.failure_threshold,
            args.max_failures,
            set(args.categories) if args.categories else None,
            args.selection,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    if not selected:
        category_text = ", ".join(args.categories) if args.categories else "all categories"
        raise SystemExit(
            f"report has no genuine failures above {args.failure_threshold:g}px for {category_text}"
        )
    panels = [_panel(root, records[row["id"]], row) for row in selected]
    columns = min(3, len(panels))
    rows_count = (len(panels) + columns - 1) // columns
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    header_h = 34
    montage = Image.new(
        "RGB", (columns * cell_w, header_h + rows_count * cell_h), (232, 232, 232)
    )
    title = (
        f"Measured failures | method={args.method} | error>{args.failure_threshold:g}px | "
        f"selection={args.selection}"
    )
    ImageDraw.Draw(montage).text((10, 10), title, fill="black", font=ImageFont.load_default())
    for index, panel in enumerate(panels):
        montage.paste(
            panel,
            ((index % columns) * cell_w, header_h + (index // columns) * cell_h),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
