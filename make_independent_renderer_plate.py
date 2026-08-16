#!/usr/bin/env python3
"""Build report-bound academic evidence plates for the independent renderer."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from drift_sense.dataset import load_manifest
from drift_sense.independent_renderer import verify_independent_suite
from evaluate import verify_artifact_binding


ARCHITECTURES = ("dram", "finfet")
SUCCESS_THRESHOLD_PX = 1.0


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest(rows: list[dict[str, Any]], target: float) -> dict[str, Any]:
    # Quantize only the comparison distance so mathematically equal distances
    # do not acquire platform-dependent ordering from a final binary ULP.
    return min(
        rows,
        key=lambda row: (
            round(abs(float(row["error"]) - target), 15),
            str(row["id"]),
        ),
    )


def select_cases(
    samples: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    """Select four cases mechanically from the measured error distribution.

    ``success`` uses the successful (error <= 1 px) subset independently for
    each architecture. ``audit`` uses every measured case. Linear NumPy
    percentiles match the main evaluator.
    """
    if mode not in {"success", "audit"}:
        raise ValueError(f"unknown plate mode: {mode}")
    by_architecture: dict[str, list[dict[str, Any]]] = {}
    for architecture in ARCHITECTURES:
        rows = [row for row in samples if row.get("architecture") == architecture]
        if not rows:
            raise ValueError(f"report has no {architecture} samples")
        if any(row.get("error") is None or not np.isfinite(float(row["error"])) for row in rows):
            raise ValueError(f"report has non-finite {architecture} errors")
        by_architecture[architecture] = rows

    selected: list[dict[str, Any]] = []
    if mode == "success":
        statistics = (("median successful", 50.0), ("P95 successful", 95.0))
        for label, percentile in statistics:
            for architecture in ARCHITECTURES:
                successful = [
                    row
                    for row in by_architecture[architecture]
                    if float(row["error"]) <= SUCCESS_THRESHOLD_PX
                ]
                if not successful:
                    raise ValueError(f"report has no successful {architecture} samples")
                target = float(
                    np.percentile(
                        [float(row["error"]) for row in successful],
                        percentile,
                        method="linear",
                    )
                )
                selected.append(
                    {
                        "selection": label,
                        "selection_population": "architecture samples with error <= 1 px",
                        "selection_statistic_error_px": target,
                        "sample": _nearest(successful, target),
                    }
                )
    else:
        for label in ("median", "largest error"):
            for architecture in ARCHITECTURES:
                rows = by_architecture[architecture]
                if label == "median":
                    target = float(
                        np.percentile(
                            [float(row["error"]) for row in rows],
                            50.0,
                            method="linear",
                        )
                    )
                    sample = _nearest(rows, target)
                else:
                    target = max(float(row["error"]) for row in rows)
                    sample = min(
                        (row for row in rows if float(row["error"]) == target),
                        key=lambda row: str(row["id"]),
                    )
                selected.append(
                    {
                        "selection": label,
                        "selection_population": "all architecture samples",
                        "selection_statistic_error_px": target,
                        "sample": sample,
                    }
                )
    return selected


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow bundles this font, so rendering does not depend on host fonts.
    return ImageFont.load_default(size=size)


def _render_plate(
    data_dir: Path,
    records_by_id: dict[str, dict[str, Any]],
    selections: list[dict[str, Any]],
    metadata: dict[str, Any],
    report: dict[str, Any],
    mode: str,
    output: Path,
) -> list[dict[str, Any]]:
    canvas_width, canvas_height = 2480, 1840
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(44)
    subtitle_font = _font(25)
    panel_font = _font(27)
    label_font = _font(21)
    detail_font = _font(19)
    footer_font = _font(16)
    black = "#161616"
    grey = "#5B5B5B"
    rule = "#A8A8A8"
    blue = "#0072B2"
    vermilion = "#D55E00"

    if mode == "success":
        title = "Independent renderer - representative successful localizations"
        subtitle_scope = "successful-subset median and P95 cases (error <= 1 px)"
    else:
        title = "Independent renderer - distribution and largest-error audit"
        subtitle_scope = "architecture median and largest-error cases"
    seed = min(int(item["sample"]["seed"]) for item in selections)
    # Every sample seed is base_seed + index * stride; the first manifest row
    # therefore carries the declared base seed.
    seed = int(report["methods"]["full"]["samples"][0]["seed"])
    draw.text((60, 35), title, fill=black, font=title_font)
    draw.text(
        (60, 92),
        f"Predeclared {metadata['record_count']}-pair development transfer set | "
        f"{subtitle_scope} | base seed {seed}",
        fill=grey,
        font=subtitle_font,
    )
    draw.line((60, 132, canvas_width - 60, 132), fill=black, width=2)

    panel_origins = ((60, 160), (1250, 160), (60, 910), (1250, 910))
    image_size = 500
    sidecar_cases: list[dict[str, Any]] = []
    for letter, origin, selection in zip("abcd", panel_origins, selections, strict=True):
        row = selection["sample"]
        architecture = str(row["architecture"])
        record = records_by_id.get(str(row["id"]))
        if record is None:
            raise ValueError(f"report sample is absent from manifest: {row['id']}")
        ox, oy = origin
        draw.text(
            (ox, oy),
            f"({letter}) {architecture.upper()} - {selection['selection']}",
            fill=black,
            font=panel_font,
        )
        reference_name = str(record["reference"])
        search_name = str(record["search"])
        with Image.open(data_dir / reference_name) as image:
            reference = image.convert("L").resize(
                (image_size, image_size), Image.Resampling.LANCZOS
            ).convert("RGB")
        with Image.open(data_dir / search_name) as image:
            search = image.convert("L").resize(
                (image_size, image_size), Image.Resampling.LANCZOS
            ).convert("RGB")
        reference_xy = (ox, oy + 58)
        search_xy = (ox + 550, oy + 58)
        canvas.paste(reference, reference_xy)
        canvas.paste(search, search_xy)
        draw.rectangle(
            (*reference_xy, reference_xy[0] + image_size - 1, reference_xy[1] + image_size - 1),
            outline=rule,
            width=1,
        )
        draw.rectangle(
            (*search_xy, search_xy[0] + image_size - 1, search_xy[1] + image_size - 1),
            outline=rule,
            width=1,
        )

        source_width = float(record["search_geometry"]["width"])
        source_height = float(record["search_geometry"]["height"])
        gt_x = search_xy[0] + float(row["ground_truth"][0]) * image_size / source_width
        gt_y = search_xy[1] + float(row["ground_truth"][1]) * image_size / source_height
        prediction_x = search_xy[0] + float(row["prediction"][0]) * image_size / source_width
        prediction_y = search_xy[1] + float(row["prediction"][1]) * image_size / source_height
        radius = 12
        draw.ellipse(
            (gt_x - radius, gt_y - radius, gt_x + radius, gt_y + radius),
            outline=blue,
            width=5,
        )
        arm = 13
        draw.line(
            (
                prediction_x - arm,
                prediction_y - arm,
                prediction_x + arm,
                prediction_y + arm,
            ),
            fill=vermilion,
            width=5,
        )
        draw.line(
            (
                prediction_x - arm,
                prediction_y + arm,
                prediction_x + arm,
                prediction_y - arm,
            ),
            fill=vermilion,
            width=5,
        )
        label_y = oy + 569
        draw.text((ox, label_y), "reference | finer physical sampling", fill=grey, font=label_font)
        draw.text(
            (ox + 550, label_y),
            "search | blue circle: GT | orange cross: prediction",
            fill=grey,
            font=label_font,
        )
        detail_y = label_y + 34
        status = str(row["diagnostics"]["decision_support"]["status"])
        draw.text(
            (ox, detail_y),
            f"{row['id']} | seed {row['seed']} | error {float(row['error']):.4f} px | "
            f"decision {status}",
            fill=black,
            font=detail_font,
        )
        draw.line((ox, detail_y + 33, ox + 1110, detail_y + 33), fill=rule, width=1)
        sidecar_cases.append(
            {
                "panel": letter,
                "selection": selection["selection"],
                "selection_population": selection["selection_population"],
                "selection_statistic_error_px": selection[
                    "selection_statistic_error_px"
                ],
                "architecture": architecture,
                "id": row["id"],
                "seed": row["seed"],
                "error_px": row["error"],
                "ground_truth": row["ground_truth"],
                "prediction": row["prediction"],
                "decision_status": status,
                "reference": reference_name,
                "search": search_name,
                "reference_sha256": metadata["input_images_sha256"][reference_name],
                "search_sha256": metadata["input_images_sha256"][search_name],
            }
        )

    draw.line((1215, 155, 1215, 1650), fill=rule, width=1)
    draw.line((60, 880, canvas_width - 60, 880), fill=rule, width=1)
    draw.line((60, 1660, canvas_width - 60, 1660), fill=black, width=2)
    footer = (
        f"generator source SHA-256  {metadata['generator_source_sha256']}",
        f"generator dataset SHA-256 {metadata['dataset_sha256']}",
        f"evaluator dataset SHA-256 {report['artifact_binding']['dataset_sha256']}",
    )
    for index, line in enumerate(footer):
        draw.text((60, 1680 + index * 30), line, fill=black, font=footer_font)
    draw.text(
        (60, 1780),
        "Unretouched generator outputs; plate operations are resize, report coordinates, text, and rules.",
        fill=grey,
        font=footer_font,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    return sidecar_cases


def verify_plate_binding(
    sidecar_path: Path, report_path: Path, plate_path: Path
) -> dict[str, Any]:
    """Verify the two non-circular bindings stored in a plate sidecar."""
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("report_sha256") != _sha256(report_path):
        raise ValueError("contact-sheet report hash mismatch")
    if sidecar.get("plate_sha256") != _sha256(plate_path):
        raise ValueError("contact-sheet plate hash mismatch")
    return sidecar


def build_plate(
    data_dir: Path,
    report_path: Path,
    output_path: Path,
    sidecar_path: Path,
    mode: str,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    report_path = report_path.resolve()
    output_path = output_path.resolve()
    sidecar_path = sidecar_path.resolve()
    manifest_path = data_dir / "manifest.jsonl"
    records = load_manifest(manifest_path)
    records_by_id = {str(record["id"]): record for record in records}
    metadata = verify_independent_suite(data_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parent
    verify_artifact_binding(
        report,
        manifest_path,
        records,
        repository_root,
        require_current_code=True,
    )
    method = report.get("methods", {}).get("full")
    if not isinstance(method, dict) or not isinstance(method.get("samples"), list):
        raise ValueError("report has no full-method samples")
    selections = select_cases(method["samples"], mode)
    cases = _render_plate(
        data_dir,
        records_by_id,
        selections,
        metadata,
        report,
        mode,
        output_path,
    )
    sidecar = {
        "schema_version": 1,
        "title": f"Independent renderer {mode} evidence plate",
        "protocol": "predeclared development transfer verification",
        "mode": mode,
        "selection_policy": (
            "Per-architecture successful-subset linear percentile; mechanically nearest "
            "error with ID tie-break."
            if mode == "success"
            else "Per-architecture all-sample median or maximum; ID tie-break."
        ),
        "success_threshold_px": SUCCESS_THRESHOLD_PX if mode == "success" else None,
        "configuration": metadata["configuration"],
        "record_count": metadata["record_count"],
        "implementation_sha256": report["artifact_binding"]["implementation_sha256"],
        "generator_source_sha256": metadata["generator_source_sha256"],
        "generator_configuration_sha256": metadata["configuration_sha256"],
        "generator_manifest_sha256": metadata["manifest_sha256"],
        "generator_dataset_sha256": metadata["dataset_sha256"],
        "evaluator_manifest_sha256": report["artifact_binding"]["manifest_sha256"],
        "evaluator_dataset_sha256": report["artifact_binding"]["dataset_sha256"],
        "report_sha256": _sha256(report_path),
        "plate_sha256": _sha256(output_path),
        "plate_operations": [
            "Pillow LANCZOS resize",
            "ground-truth circle overlay",
            "prediction cross overlay",
            "text and rules",
        ],
        "cases": cases,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return verify_plate_binding(sidecar_path, report_path, output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode", choices=("success", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sidecar = build_plate(
            args.data_dir,
            args.report,
            args.output,
            args.sidecar,
            args.mode,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        json.dumps(
            {
                "mode": sidecar["mode"],
                "output": str(args.output),
                "plate_sha256": sidecar["plate_sha256"],
                "sidecar": str(args.sidecar),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
