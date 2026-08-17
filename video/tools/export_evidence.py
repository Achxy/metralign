#!/usr/bin/env python3
"""Export hash-bound numerical and raster evidence for the Metralign film.

The exporter never generates microscopy or benchmark observations.  Its
canonical bundle reads only checked-in frozen records; the film's checked-in,
hash-bound acquired-microscopy plate and sidecar remain separate inputs.
Derived rasters are deterministic views of source bytes or arrays produced by
the shipping localization implementation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO, StringIO
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Any
from zipfile import ZipFile

# Make the repository implementation importable when this file is executed
# directly from ``video/tools``; no installed Metralign distribution is used.
ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import scipy

from drift_sense.candidates import top_k_candidates
from drift_sense.correlation import balanced_residual_score_map, candidate_supported_peak, zncc_map
from drift_sense.localizer import (
    LocalizationConfig,
    _choose_candidate_with_evidence,
    _projection_energy_fraction,
    _refine_2d,
    _resize,
    _transform_template,
    localize,
)
from drift_sense.representations import (
    _phase_lattice_vector,
    _quadratic_spectral_peak,
    estimate_periodic_transform,
    periodic_difference_channels,
)
from drift_sense.spectral import detect_reciprocal_peaks, log_power_spectrum, robust_float
from real_imagery.protocol import digital_crop_pair


DEFAULT_OUTPUT = ROOT / "video" / "evidence" / "exported"
SUITES = (
    "iid",
    "high_noise",
    "geometry_ood",
    "transform_ood",
    "periodic_ambiguity",
    "scan_distortion",
    "cross_generator",
)

CLASSIC_DISPLAY_NAMES = {
    "opencv_ecc_affine": "OpenCV ECC affine",
    "opencv_grid_template": "OpenCV grid template",
    "opencv_sift_ransac": "OpenCV SIFT + RANSAC",
    "opencv_template": "OpenCV template",
    "opencv_template_phase": "OpenCV template + phase",
    "skimage_template_phase": "scikit-image template + phase",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def array_binding(array: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return {
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "array_sha256": digest.hexdigest(),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.assets: dict[str, dict[str, Any]] = {}

    def _finish(
        self,
        relative: str,
        *,
        kind: str,
        sources: list[str],
        derivation: str,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.root / relative
        record: dict[str, Any] = {
            "file": f"video/evidence/exported/{relative}",
            "bundle_relative_file": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "kind": kind,
            "source_keys": sources,
            "derivation": derivation,
        }
        if extra:
            record.update(extra)
        self.assets[relative] = record
        return path

    def text(self, relative: str, value: str, **metadata: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return self._finish(relative, **metadata)

    def json(self, relative: str, value: Any, **metadata: Any) -> Path:
        return self.text(relative, canonical_json(value), **metadata)

    def bytes(self, relative: str, value: bytes, **metadata: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return self._finish(relative, **metadata)

    def npy(self, relative: str, array: np.ndarray, **metadata: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        return self._finish(relative, extra={"array": array_binding(array)}, **metadata)

    def image(self, relative: str, image: Image.Image, **metadata: Any) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG", optimize=False)
        supplied = metadata.pop("extra", {})
        image_metadata = {"pixel_mode": image.mode, "pixel_size_wh": list(image.size)}
        image_metadata.update(supplied)
        return self._finish(
            relative,
            extra=image_metadata,
            **metadata,
        )


def source_record(path: Path) -> dict[str, Any]:
    return {
        "repo_path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def gray_image(array: np.ndarray) -> Image.Image:
    values = np.asarray(array)
    if values.dtype != np.uint8:
        values = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return Image.fromarray(values, mode="L")


def scalar_u16(array: np.ndarray, low: float, high: float) -> Image.Image:
    values = np.asarray(array, dtype=np.float64)
    if not high > low:
        raise ValueError("display interval must be nonempty")
    mapped = np.rint(np.clip((values - low) / (high - low), 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    return Image.fromarray(mapped, mode="L")


def score_image(array: np.ndarray) -> Image.Image:
    return scalar_u16(array, -1.0, 1.0)


def _spectral_peak_records(peaks: list[Any], shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Bind detected reciprocal frequencies to their displayed FFT pixels."""
    height, width = shape
    records: list[dict[str, Any]] = []
    for peak in peaks:
        representative = [
            width / 2.0 + float(peak.fx) * width,
            height / 2.0 + float(peak.fy) * height,
        ]
        conjugate = [
            width / 2.0 - float(peak.fx) * width,
            height / 2.0 - float(peak.fy) * height,
        ]
        records.append(
            {
                **asdict(peak),
                "representative_pixel_xy": representative,
                "conjugate_pixel_xy": conjugate,
            }
        )
    return records


def _spectral_peak_image(
    array: np.ndarray,
    high: float,
    records: list[dict[str, Any]],
) -> Image.Image:
    image = scalar_u16(array, 0.0, high).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (126, 166, 201)
    for record in records:
        representative = record["representative_pixel_xy"]
        conjugate = record["conjugate_pixel_xy"]
        for x, y in (representative, conjugate):
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
            draw.line((x - radius, y, x + radius, y), fill=color, width=1)
            draw.line((x, y - radius, x, y + radius), fill=color, width=1)
    return image


def signed_pair_images(first: np.ndarray, second: np.ndarray) -> tuple[Image.Image, Image.Image, float]:
    limit = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))), 1e-12)
    return scalar_u16(first, -limit, limit), scalar_u16(second, -limit, limit), limit


def overlay(
    pixels: np.ndarray,
    ground_truth: tuple[float, float],
    prediction: tuple[float, float],
    candidates: list[tuple[float, float]] | None = None,
) -> Image.Image:
    image = gray_image(pixels).convert("RGB")
    draw = ImageDraw.Draw(image)
    if candidates:
        for x, y in candidates:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(235, 232, 224), width=2)
    gx, gy = ground_truth
    px, py = prediction
    draw.line((gx, gy, px, py), fill=(241, 240, 236), width=2)
    draw.line((gx - 9, gy, gx + 9, gy), fill=(126, 166, 201), width=3)
    draw.line((gx, gy - 9, gx, gy + 9), fill=(126, 166, 201), width=3)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), outline=(214, 90, 58), width=3)
    return image


def _line_plot(
    width: int,
    height: int,
    series: list[tuple[list[float], tuple[int, int, int]]],
    *,
    y_min: float,
    y_max: float,
) -> Image.Image:
    """A plain evidence raster: axes and source-valued polylines, no decoration."""
    canvas = Image.new("RGB", (width, height), (9, 9, 9))
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = 70, 36, width - 28, height - 54
    draw.line((left, bottom, right, bottom), fill=(154, 153, 148), width=2)
    draw.line((left, top, left, bottom), fill=(154, 153, 148), width=2)
    span = max(y_max - y_min, 1e-12)
    for values, color in series:
        if len(values) < 2:
            continue
        points = [
            (
                left + index * (right - left) / (len(values) - 1),
                bottom - (float(value) - y_min) * (bottom - top) / span,
            )
            for index, value in enumerate(values)
        ]
        draw.line(points, fill=color, width=3)
    return canvas


def _distribution_plot(errors: np.ndarray) -> Image.Image:
    width, height = 1400, 560
    canvas = Image.new("RGB", (width, height), (9, 9, 9))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else font
    left, top, right, bottom = 84, 72, width - 42, height - 104
    capped = np.minimum(np.asarray(errors, dtype=np.float64), 1.0)
    counts, edges = np.histogram(capped, bins=np.linspace(0.0, 1.0, 51))
    maximum = max(int(np.max(counts)), 1)
    draw.line((left, bottom, right, bottom), fill=(154, 153, 148), width=2)
    draw.line((left, top, left, bottom), fill=(154, 153, 148), width=2)
    for index, count in enumerate(counts):
        x0 = left + index * (right - left) / len(counts)
        x1 = left + (index + 1) * (right - left) / len(counts) - 1
        y0 = bottom - int(count) * (bottom - top) / maximum
        draw.rectangle((x0, y0, x1, bottom), fill=(241, 240, 236))

    for value, label, anchor in ((0.0, "0", "la"), (0.5, ".5", "ma"), (1.0, "1", "ra")):
        x = left + value * (right - left)
        draw.line((x, bottom, x, bottom + 9), fill=(154, 153, 148), width=2)
        draw.text((x, bottom + 16), label, fill=(241, 240, 236), font=font, anchor=anchor)
    draw.text((right, bottom + 48), "PX", fill=(154, 153, 148), font=small_font, anchor="ra")
    draw.text(
        (right, bottom + 76),
        "FINAL BIN INCLUDES ERRORS > 1 PX",
        fill=(154, 153, 148),
        font=small_font,
        anchor="ra",
    )

    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95, method="linear"))
    for value, color in ((median, (126, 166, 201)), (p95, (214, 90, 58))):
        x = left + min(max(value, 0.0), 1.0) * (right - left)
        draw.line((x, top, x, bottom), fill=color, width=3)
    draw.text(
        (left, 24),
        f"MEDIAN {median:.3f} PX",
        fill=(126, 166, 201),
        font=font,
        anchor="la",
    )
    draw.text(
        (left + 330, 24),
        f"P95 {p95:.3f} PX",
        fill=(214, 90, 58),
        font=font,
        anchor="la",
    )
    draw.text(
        (right, 24),
        f"N = {len(errors):,}",
        fill=(154, 153, 148),
        font=font,
        anchor="ra",
    )
    return canvas


def _suite_strip(rows: list[dict[str, Any]]) -> Image.Image:
    width, height = 1400, 600
    canvas = Image.new("RGB", (width, height), (9, 9, 9))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    font = ImageFont.truetype(str(font_path), 24) if font_path.is_file() else ImageFont.load_default()
    header_font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else font
    left, right = 68, width - 54
    suite_x, count_x, p95_x = left, 860, 1290
    header_y, first_y, row_height = 38, 102, 66
    rule = (70, 70, 68)
    muted = (154, 153, 148)
    paper = (241, 240, 236)
    accent = (126, 166, 201)

    draw.text((suite_x, header_y), "SUITE", fill=muted, font=header_font, anchor="la")
    draw.text((count_x, header_y), "WITHIN 1 PX", fill=muted, font=header_font, anchor="ra")
    draw.text((p95_x, header_y), "P95 ERROR", fill=muted, font=header_font, anchor="ra")
    draw.line((left, 74, right, 74), fill=muted, width=1)
    for index, row in enumerate(rows):
        y = first_y + index * row_height
        pair_count = int(row["pair_count"])
        within = int(row["within_1px_count"])
        p95 = float(row["p95_error_px"])
        draw.text(
            (suite_x, y),
            str(row["suite"]).replace("_", " ").upper(),
            fill=paper,
            font=font,
            anchor="lm",
        )
        draw.text(
            (count_x, y),
            f"{within} / {pair_count}",
            fill=accent if within == pair_count else (214, 90, 58),
            font=font,
            anchor="rm",
        )
        draw.text(
            (p95_x, y),
            f"{p95:.3f} PX",
            fill=paper,
            font=font,
            anchor="rm",
        )
        draw.line((left, y + 32, right, y + 32), fill=rule, width=1)
    return canvas


def _external_comparison_table(
    metralign: dict[str, Any],
    classic_rows: list[dict[str, Any]],
    xfeat: dict[str, Any],
) -> Image.Image:
    """Render a flat audit table from source-bound comparison rows."""
    width, height = 1600, 900
    canvas = Image.new("RGB", (width, height), (9, 9, 9))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    row_font = ImageFont.truetype(str(font_path), 23) if font_path.is_file() else ImageFont.load_default()
    header_font = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else row_font
    note_font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else row_font
    left, right = 68, width - 68
    coverage_x, result_x = 1110, right
    paper = (241, 240, 236)
    muted = (154, 153, 148)
    rule = (70, 70, 68)
    blue = (126, 166, 201)
    orange = (214, 90, 58)

    draw.text(
        (left, 36),
        "FIXED EXTERNAL REGISTRATION · SAME 1,400 SEALED IMAGE PAIRS",
        fill=muted,
        font=header_font,
        anchor="la",
    )
    draw.text((left, 76), "METHOD", fill=muted, font=header_font, anchor="la")
    draw.text((coverage_x, 76), "COVERAGE", fill=muted, font=header_font, anchor="ra")
    draw.text((result_x, 76), "≤ 1 PX · ALL-PAIR RATE", fill=muted, font=header_font, anchor="ra")
    draw.line((left, 100, right, 100), fill=muted, width=1)

    draw.text((left, 134), "METRALIGN · FROZEN", fill=blue, font=row_font, anchor="lm")
    draw.text(
        (coverage_x, 134),
        f"{int(metralign['resolved_count']):,} / {int(metralign['count']):,}",
        fill=paper,
        font=row_font,
        anchor="rm",
    )
    draw.text(
        (result_x, 134),
        f"{int(metralign['within_1px_count']):,} / {int(metralign['count']):,} · {float(metralign['within_1px_rate_display']):.2f}%",
        fill=blue,
        font=row_font,
        anchor="rm",
    )
    draw.line((left, 166, right, 166), fill=rule, width=1)

    draw.text(
        (left, 197),
        f"PREREGISTERED CLASSIC ADAPTERS · {len(classic_rows)} FIXED METHODS",
        fill=muted,
        font=header_font,
        anchor="la",
    )
    first_y, row_height = 237, 62
    for index, row in enumerate(classic_rows):
        y = first_y + index * row_height
        is_best = bool(row["is_best"])
        draw.text((left, y), row["display_name"].upper(), fill=paper, font=row_font, anchor="lm")
        draw.text(
            (coverage_x, y),
            f"{int(row['resolved_count']):,} / {int(row['count']):,}",
            fill=paper,
            font=row_font,
            anchor="rm",
        )
        result = f"{float(row['within_1px_rate_display']):.2f}%"
        if is_best:
            result += " · BEST CLASSIC"
        draw.text(
            (result_x, y),
            result,
            fill=orange if is_best else paper,
            font=row_font,
            anchor="rm",
        )
        draw.line((left, y + 30, right, y + 30), fill=rule, width=1)

    draw.line((left, 635, right, 635), fill=muted, width=2)
    draw.text(
        (left, 666),
        "RETROSPECTIVE DEVELOPMENT TASK-MISMATCH CONTROL",
        fill=muted,
        font=header_font,
        anchor="la",
    )
    draw.text((left, 708), "OFFICIAL XFEAT* + USAC_MAGSAC", fill=paper, font=row_font, anchor="lm")
    draw.text(
        (coverage_x, 708),
        f"{int(xfeat['resolved_count']):,} / {int(xfeat['count']):,} · {float(xfeat['coverage_rate_display']):.2f}%",
        fill=paper,
        font=row_font,
        anchor="rm",
    )
    draw.text(
        (result_x, 708),
        f"{int(xfeat['within_5px_count']):,} / {int(xfeat['count']):,} ≤ 5 PX",
        fill=orange,
        font=row_font,
        anchor="rm",
    )
    draw.line((left, 740, right, 740), fill=muted, width=1)
    draw.text(
        (left, 786),
        "POST-FREEZE DEVELOPMENT EVIDENCE · NOT PART OF THE FROZEN CLAIM",
        fill=muted,
        font=note_font,
        anchor="la",
    )
    draw.text(
        (left, 822),
        "NOT A SOTA OR INTENDED-BENCHMARK VERDICT ON XFEAT",
        fill=muted,
        font=note_font,
        anchor="la",
    )
    return canvas


def _terminal_png(command: str, stdout: str) -> Image.Image:
    image = Image.new("RGB", (1600, 520), (9, 9, 9))
    draw = ImageDraw.Draw(image)
    font_path = Path("/System/Library/Fonts/Menlo.ttc")
    font = ImageFont.truetype(str(font_path), 26) if font_path.is_file() else ImageFont.load_default()
    output_font = ImageFont.truetype(str(font_path), 38) if font_path.is_file() else font
    wrapped = textwrap.wrap("$ " + command, width=92, subsequent_indent="  ", break_long_words=False)
    draw.multiline_text((48, 48), "\n".join(wrapped), fill=(241, 240, 236), font=font, spacing=12)
    draw.text((48, 350), stdout.strip(), fill=(214, 90, 58), font=output_font)
    return image


def _phase_trace(image: np.ndarray, axis: str, frequency_range: tuple[float, float]) -> dict[str, Any]:
    normalized = robust_float(image).astype(np.float32)
    projection = np.mean(normalized, axis=0 if axis == "x" else 1)
    frequency = _quadratic_spectral_peak(projection, *frequency_range)
    if frequency is None:
        raise ValueError(f"phase trace unavailable for axis {axis}")
    if axis == "x":
        positions = np.arange(normalized.shape[1], dtype=np.float64)
        carrier = np.hanning(normalized.shape[1]) * np.exp(-2j * np.pi * frequency * positions)
        coefficients = normalized @ carrier
    else:
        positions = np.arange(normalized.shape[0], dtype=np.float64)
        carrier = np.hanning(normalized.shape[0]) * np.exp(-2j * np.pi * frequency * positions)
        coefficients = carrier @ normalized
    coordinates = np.arange(coefficients.size, dtype=np.float64)
    amplitude = np.abs(coefficients)
    phase = np.unwrap(np.angle(coefficients))
    keep = (
        (coordinates >= 0.08 * coefficients.size)
        & (coordinates <= 0.92 * coefficients.size)
        & (amplitude >= np.quantile(amplitude, 0.15))
    )
    slope, intercept = np.polyfit(
        coordinates[keep], phase[keep], 1, w=np.sqrt(amplitude[keep] + 1e-9)
    )
    fitted = slope * coordinates + intercept
    phase_error = float(
        np.sqrt(np.average((phase[keep] - fitted[keep]) ** 2, weights=amplitude[keep]))
    )
    cross_frequency = float(slope / (2.0 * np.pi))
    vector = (
        [float(frequency), cross_frequency]
        if axis == "x"
        else [cross_frequency, float(frequency)]
    )
    confidence = float(np.exp(-min(phase_error, 8.0) / 2.5))
    official = _phase_lattice_vector(normalized, axis, frequency_range)
    if official is None or not np.allclose(official[0], vector, rtol=0.0, atol=1e-12):
        raise AssertionError("phase trace diverges from shipping implementation")
    return {
        "summary": {
            "axis": axis,
            "frequency_range_cycles_per_px": list(frequency_range),
            "carrier_frequency_cycles_per_px": float(frequency),
            "cross_frequency_cycles_per_px": cross_frequency,
            "reciprocal_vector": vector,
            "fit_slope_rad_per_px": float(slope),
            "fit_intercept_rad": float(intercept),
            "phase_rmse_rad": phase_error,
            "confidence": confidence,
            "kept_count": int(np.count_nonzero(keep)),
            "sample_count": int(coordinates.size),
        },
        "coordinate": coordinates,
        "amplitude": amplitude,
        "phase": phase,
        "fitted": fitted,
        "keep": keep,
    }


def _phase_csv(trace: dict[str, Any]) -> str:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("scan_coordinate_px", "carrier_amplitude", "unwrapped_phase_rad", "fitted_phase_rad", "fit_used"))
    for row in zip(
        trace["coordinate"], trace["amplitude"], trace["phase"], trace["fitted"], trace["keep"], strict=True
    ):
        writer.writerow((format(float(row[0]), ".17g"), format(float(row[1]), ".17g"), format(float(row[2]), ".17g"), format(float(row[3]), ".17g"), int(row[4])))
    return stream.getvalue()


def _rounded_binding(exact: float, display: float, rule: str, source: str) -> dict[str, Any]:
    return {
        "exact_value": exact,
        "display_value": display,
        "rounding_rule": rule,
        "source": source,
    }


def frozen_metrics(writer: Writer, sources: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    all_samples: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    input_hash_count = 0
    for suite in SUITES:
        path = ROOT / "results" / "frozen" / "reports" / f"{suite}.json"
        key = f"frozen_report_{suite}"
        sources[key] = source_record(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = report["methods"]["full"]["samples"]
        if len(samples) != 200:
            raise AssertionError(f"unexpected sample count for {suite}")
        all_samples.extend(samples)
        input_hash_count += len(report["artifact_binding"]["input_images_sha256"])
        errors = np.asarray([row["error"] for row in samples], dtype=np.float64)
        suite_rows.append(
            {
                "suite": suite,
                "pair_count": int(errors.size),
                "within_1px_count": int(np.count_nonzero(errors <= 1.0)),
                "within_1px_rate": float(np.mean(errors <= 1.0)),
                "median_error_px": float(np.median(errors)),
                "p95_error_px": float(np.percentile(errors, 95, method="linear")),
                "maximum_error_px": float(np.max(errors)),
            }
        )
    errors = np.asarray([row["error"] for row in all_samples], dtype=np.float64)
    runtimes = np.asarray([row["runtime_ms"] for row in all_samples], dtype=np.float64)
    exact_median = float(np.median(errors))
    exact_p95 = float(np.percentile(errors, 95, method="linear"))
    exact_runtime = float(np.mean(runtimes))
    count = int(errors.size)
    within = int(np.count_nonzero(errors <= 1.0))
    failed = int(np.count_nonzero(errors > 5.0))
    if (count, within, failed) != (1400, 1398, 2):
        raise AssertionError("frozen aggregate differs from the sealed result")

    rows_stream = StringIO()
    csv_writer = csv.DictWriter(rows_stream, fieldnames=list(suite_rows[0]))
    csv_writer.writeheader()
    csv_writer.writerows(suite_rows)
    report_keys = [f"frozen_report_{suite}" for suite in SUITES]
    writer.text(
        "benchmark_suites.csv",
        rows_stream.getvalue(),
        kind="plot_data",
        sources=report_keys,
        derivation="Per-suite counts and NumPy-linear percentiles recomputed from archived full-method sample rows.",
    )
    plot_payload = {
        "schema_version": 1,
        "threshold_px": 1.0,
        "percentile_method": "NumPy linear",
        "aggregate": {
            "pair_count": count,
            "within_1px_count": within,
            "outside_1px_count": count - within,
            "within_1px_rate": within / count,
            "median_error_px": exact_median,
            "p95_error_px": exact_p95,
            "mean_runtime_ms": exact_runtime,
            "failure_gt_5px_count": failed,
        },
        "suites": suite_rows,
    }
    writer.json(
        "benchmark_plot_data.json",
        plot_payload,
        kind="plot_data",
        sources=report_keys,
        derivation="Exact aggregate and suite values recomputed from all 1,400 archived sample records.",
    )
    writer.image(
        "evaluation_error_distribution.png",
        _distribution_plot(errors),
        kind="derived_plot_raster",
        sources=report_keys,
        derivation="Fifty fixed-width bins over errors clipped at the documented 1 px primary threshold, with source-derived median and linear-P95 markers; raw values remain in the reports and benchmark_plot_data.json.",
        extra={
            "x_domain_px": [0.0, 1.0],
            "bin_count": 50,
            "overflow_rule": "error values above 1 px enter the final bin",
            "displayed_pair_count": count,
            "displayed_median_error_px": round(exact_median, 3),
            "displayed_p95_error_px": round(exact_p95, 3),
            "display_rounding_rule": "round-half-even to 3 decimal places",
        },
    )
    writer.image(
        "evaluation_suite_strip.png",
        _suite_strip(suite_rows),
        kind="derived_plot_raster",
        sources=report_keys,
        derivation="Flat, directly labeled suite rows showing exact within-1-px count over pair count and source-derived linear P95 error from benchmark_suites.csv.",
        extra={
            "displayed_rows": [
                {
                    "suite": row["suite"],
                    "within_1px_count": row["within_1px_count"],
                    "pair_count": row["pair_count"],
                    "p95_error_px_exact": row["p95_error_px"],
                    "p95_error_px_display": round(float(row["p95_error_px"]), 3),
                }
                for row in suite_rows
            ],
            "p95_display_rounding_rule": "round-half-even to 3 decimal places",
        },
    )
    return (
        {
            "pair_count": count,
            "suite_count": len(SUITES),
            "within_1px_count": within,
            "outside_1px_count": count - within,
            "outside_1px_count_binding": {
                "exact_value": count - within,
                "display_value": count - within,
                "rounding_rule": "source pair_count minus source within_1px_count; exact integer, no rounding",
                "source": "benchmark_plot_data.json#/aggregate/outside_1px_count",
            },
            "within_1px_rate": 100.0 * within / count,
            "within_1px_rate_binding": _rounded_binding(
                within / count,
                100.0 * within / count,
                "exact fraction multiplied by 100; no decimal rounding",
                "benchmark_plot_data.json#/aggregate/within_1px_rate",
            ),
            "threshold_px": float(plot_payload["threshold_px"]),
            "primary_threshold_px": int(plot_payload["threshold_px"]),
            "primary_threshold_px_binding": _rounded_binding(
                float(plot_payload["threshold_px"]),
                int(plot_payload["threshold_px"]),
                "exact whole-pixel value rendered as an integer; no numerical rounding",
                "benchmark_plot_data.json#/threshold_px",
            ),
            "failure_gt_5px_count": failed,
            "input_image_hash_count": input_hash_count,
            "median_error_px": round(exact_median, 3),
            "median_error_px_binding": _rounded_binding(
                exact_median, round(exact_median, 3), "round-half-even to 3 decimal places", "benchmark_plot_data.json#/aggregate/median_error_px"
            ),
            "p95_error_px": round(exact_p95, 3),
            "p95_error_px_binding": _rounded_binding(
                exact_p95, round(exact_p95, 3), "round-half-even to 3 decimal places", "benchmark_plot_data.json#/aggregate/p95_error_px"
            ),
            "mean_runtime_ms": round(exact_runtime),
            "mean_runtime_ms_binding": _rounded_binding(
                exact_runtime, round(exact_runtime), "round-half-even to nearest millisecond", "benchmark_plot_data.json#/aggregate/mean_runtime_ms"
            ),
        },
        errors,
    )


def comparison_metrics(
    sources: dict[str, Any], writer: Writer | None = None
) -> dict[str, Any]:
    independent_path = ROOT / "results/comparisons/independent-renderer-final-100.json"
    classic_path = ROOT / "results/comparisons/external-registration-frozen.json"
    xfeat_path = ROOT / "results/comparisons/xfeat-frozen-all1400-development.json"
    safety_path = ROOT / "results/comparisons/safety-audit.json"
    sources["independent_renderer_report"] = source_record(independent_path)
    sources["classic_external_frozen_report"] = source_record(classic_path)
    sources["xfeat_frozen_development_report"] = source_record(xfeat_path)
    sources["safety_audit_report"] = source_record(safety_path)
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    independent_all = independent["methods"]["full"]["metrics"]["all"]
    independent_count = int(independent_all["count"])
    independent_rate = float(independent_all["success_le_1px"])
    classic = json.loads(classic_path.read_text(encoding="utf-8"))
    eligible = {
        name: row["metrics"]
        for name, row in classic["methods"].items()
        if name != "metralign_archived"
    }
    if set(eligible) != set(CLASSIC_DISPLAY_NAMES):
        raise AssertionError("classic comparison method inventory changed")
    best_name, best = max(
        eligible.items(), key=lambda item: (float(item[1]["success_le_1px"]), item[0])
    )
    best_fraction = float(best["success_le_1px"])
    best_display_percent = round(best_fraction * 100.0, 2)
    classic_rows = [
        {
            "method_id": name,
            "display_name": CLASSIC_DISPLAY_NAMES[name],
            "count": int(metrics["count"]),
            "resolved_count": int(metrics["resolved_count"]),
            "coverage_exact_fraction": float(metrics["coverage"]),
            "within_1px_rate_exact_fraction": float(metrics["success_le_1px"]),
            "within_1px_rate_display": round(
                100.0 * float(metrics["success_le_1px"]), 2
            ),
            "is_best": name == best_name,
            "source": f"results/comparisons/external-registration-frozen.json#/methods/{name}/metrics",
        }
        for name, metrics in eligible.items()
    ]

    metralign_metrics = classic["methods"]["metralign_archived"]["metrics"]
    metralign_count = int(metralign_metrics["count"])
    metralign_rate = float(metralign_metrics["success_le_1px"])
    metralign_success_product = metralign_count * metralign_rate
    if not math.isclose(
        metralign_success_product,
        round(metralign_success_product),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssertionError("Metralign comparison fraction does not map to a count")
    metralign_row = {
        "method_id": "metralign_archived",
        "display_name": "Metralign · frozen",
        "count": metralign_count,
        "resolved_count": int(metralign_metrics["resolved_count"]),
        "within_1px_count": int(round(metralign_success_product)),
        "within_1px_rate_exact_fraction": metralign_rate,
        "within_1px_rate_display": round(100.0 * metralign_rate, 2),
        "source": "results/comparisons/external-registration-frozen.json#/methods/metralign_archived/metrics",
    }

    xfeat = json.loads(xfeat_path.read_text(encoding="utf-8"))
    xfeat_metrics = xfeat["pooled"]["metrics"]
    xfeat_count = int(xfeat_metrics["count"])
    xfeat_coverage = float(xfeat_metrics["coverage"])
    xfeat_success_rate = float(xfeat_metrics["success_le_5px"])
    xfeat_success_product = xfeat_count * xfeat_success_rate
    if not math.isclose(
        xfeat_success_product,
        round(xfeat_success_product),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise AssertionError("XFeat success fraction does not map to a count")
    xfeat_row = {
        "method_id": str(xfeat["method"]),
        "display_name": "Official XFeat* + USAC_MAGSAC",
        "count": xfeat_count,
        "resolved_count": int(xfeat_metrics["resolved_count"]),
        "coverage_rate_exact_fraction": xfeat_coverage,
        "coverage_rate_display": round(100.0 * xfeat_coverage, 2),
        "within_5px_count": int(round(xfeat_success_product)),
        "within_5px_rate_exact_fraction": xfeat_success_rate,
        "within_5px_rate_display": round(100.0 * xfeat_success_rate, 2),
        "claim_boundary": str(xfeat["claim_boundary"]),
        "upstream_commit": str(xfeat["external_software"]["xfeat"]["commit"]),
        "upstream_license": str(xfeat["external_software"]["xfeat"]["license"]),
        "source": "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics",
    }
    if writer is not None:
        writer.image(
            "external_comparison_table.png",
            _external_comparison_table(metralign_row, classic_rows, xfeat_row),
            kind="derived_comparison_table",
            sources=[
                "classic_external_frozen_report",
                "xfeat_frozen_development_report",
            ],
            derivation="Flat table of source-named methods, all-sample classic within-1-px rates, and the separately scoped official-XFeat retrospective task-mismatch result.",
            extra={
                "metralign": metralign_row,
                "classic_adapters": classic_rows,
                "xfeat_development": xfeat_row,
                "rate_display_rounding_rule": "exact all-sample fraction multiplied by 100, then round-half-even to 2 decimal places",
                "xfeat_scope_note": "Retrospective post-freeze development task-mismatch control; not part of the frozen claim and not a SOTA or intended-benchmark verdict on XFeat.",
            },
        )
    safety = json.loads(safety_path.read_text(encoding="utf-8"))["frozen_evaluation"]["selective"]
    return {
        "independent_renderer": {
            "pair_count": independent_count,
            "within_1px_count": int(round(independent_count * independent_rate)),
            "within_1px_rate": 100.0 * independent_rate,
            "threshold_px": 1.0,
            "primary_threshold_px": 1,
            "primary_threshold_px_binding": _rounded_binding(
                1.0,
                1,
                "exact whole-pixel value rendered as an integer; no numerical rounding",
                "results/comparisons/independent-renderer-final-100.json#/methods/full/metrics/all/success_le_1px",
            ),
            "threshold_binding": {
                "source_metric": "success_le_1px",
                "comparison": "error <= threshold",
                "source": "results/comparisons/independent-renderer-final-100.json#/metric_definition",
            },
        },
        "external_baselines": {
            "classic_adapter_count": len(classic_rows),
            "classic_adapter_count_binding": {
                "exact_value": len(classic_rows),
                "display_value": len(classic_rows),
                "rounding_rule": "count of source method keys excluding metralign_archived; no rounding",
                "source": "results/comparisons/external-registration-frozen.json#/methods",
            },
            "classic_best_method": best_name,
            "classic_best_name": CLASSIC_DISPLAY_NAMES[best_name],
            "classic_best_name_binding": {
                "exact_value": best_name,
                "display_value": CLASSIC_DISPLAY_NAMES[best_name],
                "rounding_rule": "source method identifier mapped to its audit-table display label",
                "source": f"results/comparisons/external-registration-frozen.json#/methods/{best_name}",
            },
            "classic_best_within_1px_rate": best_display_percent,
            "classic_best_within_1px_rate_binding": _rounded_binding(
                best_fraction,
                best_display_percent,
                "exact all-sample fraction multiplied by 100, then round-half-even to 2 decimal places",
                f"results/comparisons/external-registration-frozen.json#/methods/{best_name}/metrics/success_le_1px",
            ),
            "classic_adapters": classic_rows,
            "xfeat_population_count": xfeat_count,
            "xfeat_population_count_binding": {
                "exact_value": xfeat_count,
                "display_value": xfeat_count,
                "rounding_rule": "integer copied from source; no rounding",
                "source": "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/count",
            },
            "xfeat_coverage_count": int(xfeat_metrics["resolved_count"]),
            "xfeat_coverage_count_binding": {
                "exact_value": int(xfeat_metrics["resolved_count"]),
                "display_value": int(xfeat_metrics["resolved_count"]),
                "rounding_rule": "integer copied from source; no rounding",
                "source": "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/resolved_count",
            },
            "xfeat_coverage_rate": xfeat_row["coverage_rate_display"],
            "xfeat_coverage_rate_binding": _rounded_binding(
                xfeat_coverage,
                xfeat_row["coverage_rate_display"],
                "exact all-sample fraction multiplied by 100, then round-half-even to 2 decimal places",
                "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/coverage",
            ),
            "xfeat_success_threshold_px": 5,
            "xfeat_success_threshold_px_binding": {
                "exact_value": 5.0,
                "display_value": 5,
                "rounding_rule": "whole-pixel threshold encoded by the source metric name success_le_5px; no numerical rounding",
                "source": "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/success_le_5px",
            },
            "xfeat_within_5px_count": xfeat_row["within_5px_count"],
            "xfeat_within_5px_count_binding": {
                "exact_value": xfeat_row["within_5px_count"],
                "display_value": xfeat_row["within_5px_count"],
                "rounding_rule": "source all-sample fraction multiplied by source population count; exact integer result",
                "source": "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/success_le_5px",
            },
            "xfeat_within_5px_rate": xfeat_row["within_5px_rate_display"],
            "xfeat_within_5px_rate_binding": _rounded_binding(
                xfeat_success_rate,
                xfeat_row["within_5px_rate_display"],
                "exact all-sample fraction multiplied by 100, then round-half-even to 2 decimal places",
                "results/comparisons/xfeat-frozen-all1400-development.json#/pooled/metrics/success_le_5px",
            ),
            "xfeat_claim_boundary": str(xfeat["claim_boundary"]),
            "scope": "Fixed preregistered classic adapters on the same sealed image bytes; not a universal or SOTA claim.",
        },
        "safety": {
            "review_count": int(safety["reviewed_count"]),
            "total_count": int(safety["count"]),
            "large_error_count": int(safety["large_error_count"]),
            "large_error_recall": float(safety["large_error_recall"]),
            "policy": "selective",
        },
    }


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.uint8, copy=True)


def _frozen_record(suite: str, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = json.loads(
        (ROOT / "results/frozen/reports" / f"{suite}.json").read_text(encoding="utf-8")
    )
    record = next(row for row in report["methods"]["full"]["samples"] if row["id"] == sample_id)
    return report, record


def algorithm_trace(writer: Writer, sources: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    suite, sample_id = "iid", "000002_dram"
    case_dir = ROOT / "results/frozen/cases/success_iid_000002_dram"
    reference_path = case_dir / f"{sample_id}_reference.png"
    search_path = case_dir / f"{sample_id}_search.png"
    case_path = case_dir / f"{sample_id}.json"
    for key, path in (
        ("success_reference", reference_path),
        ("success_search", search_path),
        ("success_case_manifest", case_path),
    ):
        sources[key] = source_record(path)
    report, archived = _frozen_record(suite, sample_id)
    reference = _load_gray(reference_path)
    search = _load_gray(search_path)
    for path in (reference_path, search_path):
        expected = report["artifact_binding"]["input_images_sha256"][path.name]
        if file_sha256(path) != expected:
            raise AssertionError(f"frozen input hash mismatch: {path}")
    manifest = json.loads(case_path.read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "results/frozen/benchmark_report.json").read_text(encoding="utf-8"))["plan"]["configuration"]
    cfg = LocalizationConfig(
        method="full",
        nominal_scale=float(manifest["nominal_scale"]),
        scale_range=float(plan["scale_range"]),
        rotation_range=float(plan["rotation_range"]),
        top_k=int(plan["top_k"]),
        enable_phase_calibration=bool(plan["enable_phase_calibration"]),
        periodic_evidence_channel=str(plan["periodic_evidence_channel"]),
        enable_spatial_residual=bool(plan["enable_spatial_residual"]),
        enable_lattice_grouping=bool(plan["enable_lattice_grouping"]),
        enable_ambiguity_rule=bool(plan["enable_ambiguity_rule"]),
        subpixel_refinement=str(plan["subpixel_refinement"]),
    )
    prediction = localize(reference, search, cfg)
    archived_xy = np.asarray(archived["prediction"], dtype=np.float64)
    if not np.allclose([prediction.x, prediction.y], archived_xy, rtol=0.0, atol=1e-12):
        raise AssertionError("current inference does not reproduce frozen success record")

    estimate = estimate_periodic_transform(reference, search, cfg.nominal_scale)
    pitch_x = float(np.clip(estimate.pitch_x, 4.0, 30.0))
    pitch_y = float(np.clip(estimate.pitch_y, 4.0, 30.0))
    transformed = _transform_template(reference, prediction.selected_scale, prediction.selected_rotation_deg)
    search_channels = periodic_difference_channels(search, pitch_x, pitch_y, cfg.periodic_evidence_channel)
    template_channels = periodic_difference_channels(transformed, pitch_x, pitch_y, cfg.periodic_evidence_channel)
    use_axis_projection = bool(
        estimate.axis_separable
        and min(
            _projection_energy_fraction(template_channels["period_x"], axis=0),
            _projection_energy_fraction(template_channels["period_y"], axis=1),
        )
        >= 0.55
    )
    if use_axis_projection:
        raise AssertionError("selected evidence sample unexpectedly uses axis projection")
    x_map = zncc_map(search_channels["period_x"], template_channels["period_x"])
    y_map = zncc_map(search_channels["period_y"], template_channels["period_y"])
    score_map, support_map = balanced_residual_score_map(x_map, y_map)
    supported = candidate_supported_peak(score_map, support_map, cfg.residual_evidence_floor)
    if supported is None:
        raise AssertionError("selected evidence sample has no supported residual peak")
    peak_y, peak_x = supported
    residual_evidence = float(support_map[peak_y, peak_x])
    candidates = top_k_candidates(score_map, cfg.top_k, cfg.nms_radius)
    decision, basis = _choose_candidate_with_evidence(
        candidates,
        score_map,
        search.shape,
        transformed.shape,
        cfg.ambiguity_margin,
        cfg.nms_radius,
        search,
        residual_evidence,
        float(estimate.confidence),
        cfg.residual_evidence_floor,
        cfg.enable_lattice_grouping,
        cfg.enable_ambiguity_rule,
        None,
    )
    if basis is not None:
        raise AssertionError("unambiguous selected sample unexpectedly formed a lattice basis")
    chosen = decision.candidate
    refined_x, refined_y, _ = _refine_2d(score_map, chosen.x, chosen.y, cfg.subpixel_refinement)
    reproduced = [
        refined_x + (transformed.shape[1] - 1.0) / 2.0,
        refined_y + (transformed.shape[0] - 1.0) / 2.0,
    ]
    if not np.allclose(reproduced, archived_xy, rtol=0.0, atol=1e-12):
        raise AssertionError("exported residual trace does not reproduce archived coordinate")

    _, fft_reference = log_power_spectrum(reference)
    _, fft_search = log_power_spectrum(search)
    fft_limit = max(float(np.max(fft_reference)), float(np.max(fft_search)))
    spectral_peak_objects = {
        capture: detect_reciprocal_peaks(image, max_peaks=24)
        for capture, image in (("reference", reference), ("search", search))
    }
    spectral_peaks = {
        capture: _spectral_peak_records(spectral_peak_objects[capture], array.shape)
        for capture, array in (("reference", fft_reference), ("search", fft_search))
    }
    for name, array in (("reference", fft_reference), ("search", fft_search)):
        writer.npy(
            f"fft_{name}.npy", array, kind="exact_numeric_array", sources=[f"success_{name}"], derivation="drift_sense.spectral.log_power_spectrum"
        )
        writer.image(
            f"fft_{name}.png",
            _spectral_peak_image(array, fft_limit, spectral_peaks[name]),
            kind="derived_evidence_raster",
            sources=[f"success_{name}"],
            derivation="8-bit linear display of the exact windowed log-power array with restrained markers at every detected reciprocal peak and its conjugate; shared pair maximum.",
            extra={
                "display_interval": [0.0, fft_limit],
                "marker_count_including_conjugates": 2 * len(spectral_peaks[name]),
                "marker_coordinate_derivation": "representative x = width/2 + fx*width, y = height/2 + fy*height; conjugate uses -fx,-fy",
                "peak_markers": spectral_peaks[name],
            },
        )

    phase_summaries: dict[str, Any] = {}
    for capture, image, ranges in (
        ("reference", reference, (cfg.nominal_scale / 30.0, cfg.nominal_scale / 4.5)),
        ("search", search, (1.0 / 30.0, 1.0 / 4.5)),
    ):
        for axis in ("x", "y"):
            trace = _phase_trace(image, axis, ranges)
            phase_summaries[f"{capture}_{axis}"] = trace["summary"]
            writer.text(
                f"phase_drift_{capture}_{axis}.csv", _phase_csv(trace), kind="exact_numeric_table", sources=[f"success_{capture}"], derivation="Exact carrier amplitude, unwrapped phase, weighted fit, and keep mask from the shipping phase-lattice estimator."
            )
    writer.json(
        "phase_transform.json",
        {
            "estimate": asdict(estimate),
            "clipped_pitch_xy_px": [pitch_x, pitch_y],
            "phase_traces": phase_summaries,
            "reciprocal_peaks": spectral_peaks,
            "reciprocal_peak_pixel_derivation": "FFT-shifted pixel x = width/2 + fx*width and y = height/2 + fy*height; conjugate uses -fx,-fy.",
        },
        kind="exact_numeric_record",
        sources=["success_reference", "success_search"],
        derivation="Shipping periodic transform estimator plus its exact phase-fit inputs.",
    )

    baseline_template = robust_float(_resize(reference, cfg.nominal_scale)).astype(np.float32)
    baseline_map = zncc_map(robust_float(search).astype(np.float32), baseline_template)
    writer.npy("baseline_score_map.npy", baseline_map, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation="ZNCC of robust-normalized search and nominal-scale reference.")
    writer.image("baseline_score_map.png", score_image(baseline_map), kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation="Fixed [-1,1] to 8-bit mapping of baseline_score_map.npy.", extra={"display_interval": [-1.0, 1.0]})
    baseline_candidates = top_k_candidates(baseline_map, cfg.top_k, cfg.nms_radius)
    baseline_half_x = (baseline_template.shape[1] - 1.0) / 2.0
    baseline_half_y = (baseline_template.shape[0] - 1.0) / 2.0
    baseline_cfg = LocalizationConfig(**{**asdict(cfg), "method": "baseline0"})
    baseline_prediction = localize(reference, search, baseline_cfg)
    baseline_peak_y, baseline_peak_x = np.unravel_index(
        int(np.argmax(baseline_map)), baseline_map.shape
    )
    baseline_refined_x, baseline_refined_y, _ = _refine_2d(
        baseline_map, int(baseline_peak_x), int(baseline_peak_y), "parabolic"
    )
    baseline_reproduced = [
        baseline_refined_x + baseline_half_x,
        baseline_refined_y + baseline_half_y,
    ]
    if not np.allclose(
        baseline_reproduced,
        [baseline_prediction.x, baseline_prediction.y],
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("baseline score-map trace does not reproduce shipping baseline0")
    baseline_error = float(
        np.hypot(
            baseline_prediction.x - float(archived["ground_truth"][0]),
            baseline_prediction.y - float(archived["ground_truth"][1]),
        )
    )
    baseline_binding = {
        "method": "baseline0",
        "ground_truth_xy": archived["ground_truth"],
        "prediction_xy": [baseline_prediction.x, baseline_prediction.y],
        "error_px": baseline_error,
        "integer_argmax_top_left_xy": [int(baseline_peak_x), int(baseline_peak_y)],
        "refined_top_left_xy": [baseline_refined_x, baseline_refined_y],
        "template_shape_yx": list(baseline_template.shape),
        "prediction_without_runtime": {
            key: value
            for key, value in baseline_prediction.to_dict().items()
            if key != "runtime_ms"
        },
        "verification": "prediction_xy exactly reproduced from baseline_score_map.npy argmax, parabolic refinement, and template-center offset within 1e-12 px",
    }
    baseline_centers = [
        (item.x + baseline_half_x, item.y + baseline_half_y)
        for item in baseline_candidates[:12]
    ]
    writer.image(
        "baseline_candidate_overlay.png",
        overlay(
            search,
            tuple(archived["ground_truth"]),
            (baseline_prediction.x, baseline_prediction.y),
            baseline_centers,
        ),
        kind="coordinate_overlay",
        sources=["success_reference", "success_search", "frozen_report_iid"],
        derivation="Top twelve shipping baseline ZNCC candidate centers plus construction truth and the shipping baseline0 prediction over exact search pixels.",
        extra={"baseline0": baseline_binding},
    )

    writer.npy("candidate_score_map.npy", score_map, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation="Equal-weight fused ZNCC of the exact period-x and period-y residual maps.")
    writer.image("candidate_score_map.png", score_image(score_map), kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation="Fixed [-1,1] to 8-bit mapping of candidate_score_map.npy.", extra={"display_interval": [-1.0, 1.0]})
    writer.npy("candidate_support_map.npy", support_map, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation="Elementwise minimum of the two exact residual-direction ZNCC maps.")
    writer.npy("matched_reference_template.npy", transformed, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation="Shipping scale-and-rotation transform of the selected reference.")
    writer.image("matched_reference_template.png", gray_image(transformed), kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation="8-bit display of matched_reference_template.npy; values rounded and clipped to the source grayscale domain.")

    half_x = (transformed.shape[1] - 1.0) / 2.0
    half_y = (transformed.shape[0] - 1.0) / 2.0
    candidate_rows = [
        {"rank": rank, "top_left_xy": [item.x, item.y], "center_xy": [item.x + half_x, item.y + half_y], "score": item.score, "selected": bool(item == chosen)}
        for rank, item in enumerate(candidates, 1)
    ]
    writer.json(
        "candidates.json",
        {
            "baseline0": baseline_binding,
            "candidates": candidate_rows,
            "decision": {
                "ambiguous": bool(decision.ambiguous),
                "score_tied": bool(decision.score_tied),
                "tied_count": int(decision.tied_count),
                "threshold": float(decision.threshold),
            },
        },
        kind="exact_numeric_record",
        sources=["success_reference", "success_search", "frozen_report_iid"],
        derivation="Shipping baseline0 binding plus top-K NMS and ambiguity decision on candidate_score_map.npy.",
    )
    candidate_centers = [tuple(row["center_xy"]) for row in candidate_rows[:12]]
    candidate_overlay = overlay(search, tuple(archived["ground_truth"]), tuple(archived["prediction"]), candidate_centers)
    writer.image("candidate_overlay.png", candidate_overlay, kind="coordinate_overlay", sources=["success_search", "frozen_report_iid"], derivation="Exact top candidate centers, archived ground truth, and archived prediction drawn over the bound search pixels.")

    chosen_x, chosen_y = chosen.x, chosen.y
    for axis, expected_name in (("period_x", ""), ("period_y", "_y")):
        ref_channel = template_channels[axis]
        sea_full = search_channels[axis]
        sea_crop = sea_full[chosen_y : chosen_y + ref_channel.shape[0], chosen_x : chosen_x + ref_channel.shape[1]]
        ref_png, sea_png, limit = signed_pair_images(ref_channel, sea_crop)
        writer.npy(f"period_difference_reference{expected_name}.npy", ref_channel, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation=f"Shipping periodic_difference_channels {axis} template array.")
        writer.npy(f"period_difference_search_crop{expected_name}.npy", sea_crop, kind="exact_numeric_array", sources=["success_reference", "success_search"], derivation=f"Candidate-aligned crop of shipping {axis} search array.")
        writer.image(f"period_difference_reference{expected_name}.png", ref_png, kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation=f"8-bit signed display of exact {axis} transformed-reference residual.", extra={"display_interval": [-limit, limit]})
        writer.image(f"period_difference_search_crop{expected_name}.png", sea_png, kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation=f"8-bit signed display of exact {axis} candidate-aligned search residual.", extra={"display_interval": [-limit, limit], "crop_top_left_xy": [chosen_x, chosen_y]})

    patch = score_map[chosen_y - 2 : chosen_y + 3, chosen_x - 2 : chosen_x + 3]
    writer.json("refinement_patch.json", {"integer_peak_top_left_xy": [chosen_x, chosen_y], "score_patch_top_left_xy": [chosen_x - 2, chosen_y - 2], "score_values": patch.tolist(), "refined_top_left_xy": [refined_x, refined_y], "refined_center_xy": reproduced}, kind="exact_numeric_record", sources=["success_reference", "success_search"], derivation="Five-by-five source window around the shipping selected integer peak and its parabolic refinement.")
    tiny = score_image(patch)
    writer.image("refinement_patch.png", tiny.resize((600, 600), Image.Resampling.NEAREST), kind="derived_evidence_raster", sources=["success_reference", "success_search"], derivation="Nearest-neighbour enlargement of the fixed [-1,1] mapped exact 5x5 score patch.", extra={"source_patch_shape_yx": [5, 5], "resampler": "nearest"})

    writer.image("success_overlay.png", overlay(search, tuple(archived["ground_truth"]), tuple(archived["prediction"])), kind="coordinate_overlay", sources=["success_search", "frozen_report_iid"], derivation="Archived prediction and construction ground truth over exact frozen search pixels.")
    sample = {
        "id": sample_id,
        "suite": suite,
        "architecture": archived["architecture"],
        "ground_truth": archived["ground_truth"],
        "prediction": archived["prediction"],
        "ground_truth_xy": archived["ground_truth"],
        "prediction_xy": archived["prediction"],
        "error_px": round(float(archived["error"]), 3),
        "error_px_binding": _rounded_binding(float(archived["error"]), round(float(archived["error"]), 3), "round-half-even to 3 decimal places", "results/frozen/reports/iid.json archived sample 000002_dram"),
        "review_recommended": False,
        "ambiguity_flag": bool(archived["diagnostics"]["ambiguity_flag"]),
        "diagnostics": {
            "selected_scale": float(prediction.selected_scale),
            "selected_rotation_deg": float(prediction.selected_rotation_deg),
            "spectral_confidence": float(prediction.spectral_confidence),
            "score": float(prediction.score),
            "runner_up_score": float(prediction.runner_up_score),
            "ambiguity_evidence": prediction.ambiguity_evidence,
            "tied_count": int(prediction.tied_count),
            "ambiguity_flag": bool(prediction.ambiguity_flag),
            "decision_support": prediction.decision_support,
        },
        "reference": sources["success_reference"],
        "search": sources["success_search"],
        "overlay_asset": "success_overlay.png",
        "baseline0": baseline_binding,
        "trace_assets": ["baseline_score_map.png", "baseline_candidate_overlay.png", "fft_reference.png", "fft_search.png", "candidate_score_map.png", "candidate_overlay.png", "matched_reference_template.png", "period_difference_reference.png", "period_difference_search_crop.png", "refinement_patch.png"],
    }
    execution = {
        "configuration": asdict(cfg),
        "prediction_without_runtime": {key: value for key, value in prediction.to_dict().items() if key != "runtime_ms"},
        "archived_prediction_match_tolerance_px": 1e-12,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__},
    }
    return sample, execution


def failure_sample(writer: Writer, sources: dict[str, Any]) -> dict[str, Any]:
    suite, sample_id = "scan_distortion", "000185_finfet"
    case_dir = ROOT / "results/frozen/cases/failure_scan_distortion_000185_finfet"
    reference_path = case_dir / f"{sample_id}_reference.png"
    search_path = case_dir / f"{sample_id}_search.png"
    case_path = case_dir / f"{sample_id}.json"
    for key, path in (("failure_reference", reference_path), ("failure_search", search_path), ("failure_case_manifest", case_path)):
        sources[key] = source_record(path)
    report, record = _frozen_record(suite, sample_id)
    for path in (reference_path, search_path):
        if file_sha256(path) != report["artifact_binding"]["input_images_sha256"][path.name]:
            raise AssertionError(f"frozen failure input hash mismatch: {path}")
    evidence = record["diagnostics"]["ambiguity_evidence"]
    policy = json.loads((ROOT / "results/comparisons/safety-audit.json").read_text(encoding="utf-8"))["policy"]
    residual_threshold = float(policy["ambiguous_residual_review_threshold"])
    transform_threshold = float(policy["transform_stability_review_threshold"])
    review = bool(
        record["diagnostics"]["ambiguity_flag"]
        and (
            float(evidence["residual_evidence"]) >= residual_threshold
            or float(evidence["transform_stability"]) < transform_threshold
        )
    )
    if not review:
        raise AssertionError("sealed large-error case must satisfy archived selective review policy")
    image = _load_gray(search_path)
    writer.image("failure_overlay.png", overlay(image, tuple(record["ground_truth"]), tuple(record["prediction"])), kind="coordinate_overlay", sources=["failure_search", "frozen_report_scan_distortion", "safety_audit_report"], derivation="Archived ground truth and returned coordinate rendered over the exact sealed failure search image.")
    return {
        "id": sample_id,
        "suite": suite,
        "architecture": record["architecture"],
        "ground_truth": record["ground_truth"],
        "prediction": record["prediction"],
        "ground_truth_xy": record["ground_truth"],
        "prediction_xy": record["prediction"],
        "error_px": float(record["error"]),
        "failure_category": record["failure_category"],
        "review_recommended": review,
        "ambiguity_flag": bool(record["diagnostics"]["ambiguity_flag"]),
        "tied_count": int(record["diagnostics"]["tied_count"]),
        "residual_evidence": float(evidence["residual_evidence"]),
        "transform_stability": float(evidence["transform_stability"]),
        "diagnostics": {
            "ambiguity_flag": bool(record["diagnostics"]["ambiguity_flag"]),
            "tied_count": int(record["diagnostics"]["tied_count"]),
            "score": float(record["diagnostics"]["score"]),
            "selected_score": float(record["diagnostics"]["selected_score"]),
            "runner_up_score": float(record["diagnostics"]["runner_up_score"]),
            "ambiguity_evidence": evidence,
            "review_policy_id": policy["id"],
            "review_thresholds": {
                "residual_evidence_minimum": residual_threshold,
                "transform_stability_minimum": transform_threshold,
            },
        },
        "reference": sources["failure_reference"],
        "search": sources["failure_search"],
        "overlay_asset": "failure_overlay.png",
    }


def live_capture(writer: Writer, sources: dict[str, Any], success: dict[str, Any]) -> dict[str, Any]:
    reference = success["reference"]["repo_path"]
    search = success["search"]["repo_path"]
    display_argv = [
        "python", "-m", "metralign", "--reference", "reference.png", "--search", "search.png", "--diagnostics"
    ]
    display_command = "PYTHONPATH=src " + " ".join(display_argv)
    actual_argv = [
        sys.executable,
        "-m",
        "metralign",
        "--reference",
        "reference.png",
        "--search",
        "search.png",
        "--diagnostics",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory(prefix="metralign-live-") as temporary_directory:
        input_directory = Path(temporary_directory)
        shutil.copyfile(ROOT / reference, input_directory / "reference.png")
        shutil.copyfile(ROOT / search, input_directory / "search.png")
        completed = subprocess.run(
            actual_argv,
            cwd=input_directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"live inference failed: {completed.stderr}")
    diagnostics = json.loads(completed.stderr)
    diagnostics.pop("runtime_ms", None)
    sanitized_stderr = json.dumps(diagnostics, sort_keys=True, separators=(",", ":")) + "\n"
    stdout = completed.stdout
    expected_stdout = f"{success['prediction_xy'][0]:.6f} {success['prediction_xy'][1]:.6f}\n"
    if stdout != expected_stdout:
        raise AssertionError("captured CLI stdout differs from source-bound prediction")
    writer.text("live_command.txt", display_command + "\n", kind="captured_command", sources=["success_reference", "success_search", "cli_source"], derivation="Portable rendering of the real invocation; interpreter and working directory are sanitized.")
    writer.text("live_stdout.txt", stdout, kind="captured_stdout", sources=["success_reference", "success_search", "cli_source"], derivation="Verbatim stdout from the executed command.")
    writer.text("live_stderr_sanitized.json", sanitized_stderr, kind="captured_diagnostics", sources=["success_reference", "success_search", "cli_source"], derivation="Verbatim structured stderr after removal of only nondeterministic runtime_ms.")
    capture = {
        "command": display_command,
        "return_code": completed.returncode,
        "stdout": stdout.rstrip("\n"),
        "stderr_sanitized_file": "live_stderr_sanitized.json",
        "cwd": "<temporary-input-directory>",
        "cwd_redaction": "ephemeral input directory replaced with <temporary-input-directory>",
        "interpreter_redaction": "absolute sys.executable replaced with python",
        "environment": {"PYTHONPATH": "src"},
        "sanitization": [
            "exact source-byte copies renamed reference.png and search.png for presentation",
            "diagnostics.runtime_ms removed because it is nondeterministic wall time",
        ],
        "stdout_sha256": sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sanitized_sha256": sha256(sanitized_stderr.encode("utf-8")).hexdigest(),
    }
    writer.json("live_capture.json", capture, kind="execution_capture", sources=["success_reference", "success_search", "cli_source"], derivation="Sanitized record of the real subprocess invocation and captured streams.")
    writer.image("terminal_capture.png", _terminal_png(display_command, stdout), kind="sanitized_terminal_render", sources=["success_reference", "success_search", "cli_source"], derivation="Deterministic typesetting of the verbatim real command and captured stdout; not a GUI screenshot.")
    search_pixels = _load_gray(ROOT / search)
    writer.image("live_overlay.png", overlay(search_pixels, tuple(success["ground_truth_xy"]), tuple(success["prediction_xy"])), kind="coordinate_overlay", sources=["success_search", "frozen_report_iid"], derivation="Captured live coordinate and archived ground truth over the exact command input search pixels.")
    return {
        "id": success["id"],
        "source_sample": "success_iid",
        "command": display_command,
        "return_code": completed.returncode,
        "stdout": stdout.rstrip("\n"),
        "prediction_xy": success["prediction_xy"],
        "diagnostics": diagnostics,
        "capture_asset": "terminal_capture.png",
        "overlay_asset": "live_overlay.png",
        "capture_record": "live_capture.json",
    }


def _carinthia_member(archive: ZipFile, record: dict[str, Any]) -> tuple[bytes, np.ndarray]:
    member = str(record["member"])
    if member.startswith("/") or ".." in Path(member).parts:
        raise ValueError("unsafe Carinthia member path")
    content = archive.read(member)
    if sha256(content).hexdigest() != record["member_sha256"]:
        raise AssertionError("Carinthia member hash mismatch")
    with Image.open(BytesIO(content)) as image:
        pixels = np.array(image, dtype=np.uint8, copy=True)
    return content, pixels


def real_sem_cases(writer: Writer, sources: dict[str, Any], archive_path: Path) -> dict[str, Any]:
    source_manifest_path = ROOT / "real_imagery/carinthia_source.json"
    report_path = ROOT / "real_imagery/results/real-sem-report.json"
    plate_path = ROOT / "real_imagery/results/real-microscopy-success-plate.json"
    sources["carinthia_source_manifest"] = source_record(source_manifest_path)
    sources["real_sem_report"] = source_record(report_path)
    sources["real_microscopy_plate_manifest"] = source_record(plate_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not archive_path.is_file():
        raise FileNotFoundError(f"pinned Carinthia archive not found: {archive_path}")
    if archive_path.stat().st_size != int(manifest["archive"]["bytes"]) or file_sha256(archive_path) != manifest["archive"]["sha256"]:
        raise AssertionError("Carinthia archive binding mismatch")
    sources["carinthia_archive"] = {
        "logical_name": manifest["archive"]["name"],
        "bytes": int(manifest["archive"]["bytes"]),
        "sha256": manifest["archive"]["sha256"],
        "doi": manifest["dataset"]["doi"],
        "license": manifest["dataset"]["license"],
        "license_url": manifest["dataset"]["license_url"],
        "local_path_recorded": False,
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [row for row in report["carinthia_semiconductor_sem_self_consistency"]["records"] if row["method"] == "full"]
    plate = json.loads(plate_path.read_text(encoding="utf-8"))
    selected_plate = next(row for row in plate["selected_records"] if row.get("track") == "carinthia" and row["selection"].startswith("nearest_p95"))
    success_record = next(row for row in rows if row["case_id"] == selected_plate["case_id"])
    failure_record = max(rows, key=lambda row: (float(row["error_px"]), row["case_id"]))
    if not (float(success_record["error_px"]) <= 1.0 and float(failure_record["error_px"]) > 1.0):
        raise AssertionError("real SEM success/failure selection invariant failed")

    output: dict[str, Any] = {}
    with ZipFile(archive_path) as archive:
        for label, record, selection in (
            ("success", success_record, selected_plate["selection"]),
            ("failure", failure_record, "maximum full-method error in the fixed 24-image Carinthia subset"),
        ):
            content, search = _carinthia_member(archive, record)
            reference, ground_truth, construction = digital_crop_pair(
                search,
                tuple(record["construction"]["position_fraction"]),
                nominal_scale=float(record["configuration"]["nominal_scale"]),
                minimum_crop_size=int(record["construction"]["minimum_crop_size_px"]),
            )
            if not np.allclose(ground_truth, record["ground_truth"], rtol=0.0, atol=0.0):
                raise AssertionError("reconstructed Carinthia ground truth mismatch")
            predicted = localize(reference, search, LocalizationConfig(**record["configuration"]))
            if not np.allclose([predicted.x, predicted.y], record["prediction"], rtol=0.0, atol=1e-12):
                raise AssertionError("reconstructed Carinthia inference mismatch")
            prefix = f"real_sem_{label}"
            writer.bytes(f"{prefix}_source.jpg", content, kind="licensed_source_image", sources=["carinthia_archive", "real_sem_report"], derivation="Exact archive member bytes; no modification.", extra={"license": manifest["dataset"]["license"], "member": record["member"]})
            writer.image(f"{prefix}_search.png", gray_image(search), kind="licensed_decoded_image", sources=["carinthia_archive", "real_sem_report"], derivation="Lossless PNG of the exact grayscale pixels decoded from the bound JPEG member.", extra={"license": manifest["dataset"]["license"]})
            writer.image(f"{prefix}_reference.png", gray_image(reference), kind="licensed_derived_reference", sources=["carinthia_archive", "real_sem_report"], derivation="Exact OpenCV INTER_LANCZOS4 digital-crop construction recorded by the real-SEM protocol.", extra={"license": manifest["dataset"]["license"]})
            writer.image(f"{prefix}_overlay.png", overlay(search, tuple(record["ground_truth"]), tuple(record["prediction"])), kind="licensed_coordinate_overlay", sources=["carinthia_archive", "real_sem_report"], derivation="Archived prediction and exact digital construction truth drawn over acquired SEM pixels.", extra={"license": manifest["dataset"]["license"]})
            output[label] = {
                "id": record["case_id"],
                "selection": selection,
                "member": record["member"],
                "member_sha256": record["member_sha256"],
                "ground_truth_xy": record["ground_truth"],
                "prediction_xy": record["prediction"],
                "error_px": float(record["error_px"]),
                "review_recommended": bool(record["diagnostics"]["pipeline_stages"].get("fallback")),
                "classification": "within_1px" if float(record["error_px"]) <= 1.0 else "greater_than_1px_within_5px",
                "construction": construction,
                "assets": {"source_jpeg": f"{prefix}_source.jpg", "search": f"{prefix}_search.png", "reference": f"{prefix}_reference.png", "overlay": f"{prefix}_overlay.png"},
                "claim_boundary": manifest["claim_boundary"],
                "license": manifest["dataset"]["license"],
                "doi": manifest["dataset"]["doi"],
            }
    writer.json("real_sem_cases.json", output, kind="licensed_evidence_record", sources=["carinthia_source_manifest", "real_sem_report", "real_microscopy_plate_manifest", "carinthia_archive"], derivation="One prespecified plate success plus the mechanically largest fixed-subset full-method real-SEM error.")
    return output


def trace_bundle_sha256(paths: list[Path]) -> str:
    digest = sha256(b"metralign-film-trace-source-v1\0")
    for path in sorted(paths, key=lambda value: value.relative_to(ROOT).as_posix()):
        name = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_export(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    writer = Writer(output)
    sources: dict[str, Any] = {}
    exporter_path = Path(__file__).resolve()
    trace_paths = [
        ROOT / "src/drift_sense/localizer.py",
        ROOT / "src/drift_sense/representations.py",
        ROOT / "src/drift_sense/correlation.py",
        ROOT / "src/drift_sense/candidates.py",
        ROOT / "src/drift_sense/ambiguity.py",
        ROOT / "src/drift_sense/refine.py",
        ROOT / "src/drift_sense/spectral.py",
        ROOT / "src/metralign/cli.py",
        ROOT / "real_imagery/protocol.py",
    ]
    sources["exporter_source"] = source_record(exporter_path)
    for path in trace_paths:
        sources["source_" + path.relative_to(ROOT).as_posix().replace("/", "_").replace(".", "_")] = source_record(path)
    sources["cli_source"] = source_record(ROOT / "src/metralign/cli.py")
    sources["frozen_aggregate_report"] = source_record(ROOT / "results/frozen/benchmark_report.json")

    frozen, _ = frozen_metrics(writer, sources)
    comparisons = comparison_metrics(sources, writer)
    success, execution = algorithm_trace(writer, sources)
    failure = failure_sample(writer, sources)
    live = live_capture(writer, sources, success)

    index = {
        "schema_version": 1,
        "project": "Metralign",
        "scope": "Judge-facing evidence export. Derived images are deterministic views of checked-in arrays, checked-in report rows, or acquired microscopy bytes; no microscopy or benchmark observation is synthesized here.",
        "exporter": {
            "repo_path": sources["exporter_source"]["repo_path"],
            "sha256": sources["exporter_source"]["sha256"],
            "trace_source_bundle_sha256": trace_bundle_sha256(trace_paths),
        },
        "sources": sources,
        "metrics": {
            "frozen": frozen,
            **comparisons,
        },
        "samples": {
            "success_iid": success,
            "failure_scan": failure,
            "live_inference": live,
        },
        "execution_binding": execution,
        "assets": dict(sorted(writer.assets.items())),
        "claim_boundaries": {
            "frozen": "Synthetic, sealed reporting population; acquired microscopy and independent-renderer checks are not pooled into this result.",
            "real_microscopy": "Acquired SEM/TEM evidence is supplied by the checked-in, hash-bound real-microscopy-success-plate.png and JSON sidecar; it remains separate from the sealed synthetic claim.",
            "classic_comparison": comparisons["external_baselines"]["scope"],
            "xfeat_development": comparisons["external_baselines"]["xfeat_claim_boundary"],
            "terminal_capture": "terminal_capture.png is a sanitized typesetting of a real command and verbatim captured stdout, not a GUI screenshot.",
        },
    }
    index_path = output / "evidence_index.json"
    index_path.write_text(canonical_json(index), encoding="utf-8")
    digest = file_sha256(index_path)
    (output / "evidence_index.sha256").write_text(f"{digest}  evidence_index.json\n", encoding="ascii")
    return index


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_strings(child)]
    return []


def verify_export(output: Path) -> dict[str, Any]:
    index_path = output / "evidence_index.json"
    sidecar_path = output / "evidence_index.sha256"
    if not index_path.is_file() or not sidecar_path.is_file():
        raise ValueError("missing evidence index or detached digest")
    expected_digest = sidecar_path.read_text(encoding="ascii").split()[0]
    if file_sha256(index_path) != expected_digest:
        raise ValueError("evidence index digest mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1:
        raise ValueError("unsupported evidence schema")
    for name, record in index["assets"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe asset path: {name}")
        path = output / relative
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
            raise ValueError(f"asset binding mismatch: {name}")
    for key, record in index["sources"].items():
        if "repo_path" not in record:
            continue
        path = ROOT / record["repo_path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]) or file_sha256(path) != record["sha256"]:
            raise ValueError(f"source binding mismatch: {key}")
    required_assets = {
        "baseline_score_map.png", "fft_search.png", "fft_reference.png",
        "period_difference_reference.png", "period_difference_search_crop.png",
        "baseline_candidate_overlay.png", "candidate_score_map.png", "candidate_overlay.png", "matched_reference_template.png", "refinement_patch.png",
        "evaluation_error_distribution.png", "evaluation_suite_strip.png",
        "external_comparison_table.png",
        "terminal_capture.png", "live_overlay.png", "failure_overlay.png",
    }
    missing = required_assets - set(index["assets"])
    if missing:
        raise ValueError(f"missing required assets: {sorted(missing)}")
    frozen = index["metrics"]["frozen"]
    plot = json.loads((output / "benchmark_plot_data.json").read_text(encoding="utf-8"))["aggregate"]
    source_bound = {
        "pair_count": int(plot["pair_count"]),
        "within_1px_count": int(plot["within_1px_count"]),
        "outside_1px_count": int(plot["outside_1px_count"]),
        "failure_gt_5px_count": int(plot["failure_gt_5px_count"]),
        "median_error_px": round(float(plot["median_error_px"]), 3),
        "p95_error_px": round(float(plot["p95_error_px"]), 3),
        "mean_runtime_ms": round(float(plot["mean_runtime_ms"])),
    }
    for key, value in source_bound.items():
        if frozen.get(key) != value:
            raise ValueError(f"frozen metric does not match recomputed plot data: {key}")
    if frozen["outside_1px_count"] != frozen["pair_count"] - frozen["within_1px_count"]:
        raise ValueError("outside-1px count is not the complement of within-1px count")
    if frozen.get("suite_count") != len(json.loads((ROOT / "results/frozen/benchmark_report.json").read_text(encoding="utf-8"))["results"]):
        raise ValueError("suite count does not match frozen aggregate report")
    classic_source = json.loads((ROOT / "results/comparisons/external-registration-frozen.json").read_text(encoding="utf-8"))
    eligible_source = {
        name: row["metrics"]
        for name, row in classic_source["methods"].items()
        if name != "metralign_archived"
    }
    best_source_name, best_source_metrics = max(
        eligible_source.items(),
        key=lambda item: (float(item[1]["success_le_1px"]), item[0]),
    )
    best_fraction = float(best_source_metrics["success_le_1px"])
    external = index["metrics"]["external_baselines"]
    if external["classic_best_within_1px_rate"] != round(100.0 * best_fraction, 2):
        raise ValueError("classic baseline percentage does not match source report")
    if external["classic_adapter_count"] != len(eligible_source):
        raise ValueError("classic adapter count does not match source report")
    if external["classic_best_method"] != best_source_name:
        raise ValueError("classic best method does not match source report")
    if external["classic_best_name"] != CLASSIC_DISPLAY_NAMES[best_source_name]:
        raise ValueError("classic best display name does not match source inventory")
    classic_rows = external["classic_adapters"]
    if [row["method_id"] for row in classic_rows] != list(eligible_source):
        raise ValueError("classic adapter table inventory does not match source order")
    for row in classic_rows:
        source_metrics = eligible_source[row["method_id"]]
        if row["display_name"] != CLASSIC_DISPLAY_NAMES[row["method_id"]]:
            raise ValueError("classic adapter display label mismatch")
        if row["count"] != int(source_metrics["count"]):
            raise ValueError("classic adapter population mismatch")
        if row["resolved_count"] != int(source_metrics["resolved_count"]):
            raise ValueError("classic adapter coverage count mismatch")
        exact_rate = float(source_metrics["success_le_1px"])
        if row["within_1px_rate_exact_fraction"] != exact_rate:
            raise ValueError("classic adapter exact rate mismatch")
        if row["within_1px_rate_display"] != round(100.0 * exact_rate, 2):
            raise ValueError("classic adapter display rate mismatch")

    xfeat_source = json.loads(
        (ROOT / "results/comparisons/xfeat-frozen-all1400-development.json").read_text(
            encoding="utf-8"
        )
    )
    xfeat_metrics = xfeat_source["pooled"]["metrics"]
    xfeat_count = int(xfeat_metrics["count"])
    xfeat_coverage = float(xfeat_metrics["coverage"])
    xfeat_success = float(xfeat_metrics["success_le_5px"])
    expected_xfeat = {
        "xfeat_population_count": xfeat_count,
        "xfeat_coverage_count": int(xfeat_metrics["resolved_count"]),
        "xfeat_coverage_rate": round(100.0 * xfeat_coverage, 2),
        "xfeat_within_5px_count": int(round(xfeat_count * xfeat_success)),
        "xfeat_within_5px_rate": round(100.0 * xfeat_success, 2),
        "xfeat_claim_boundary": str(xfeat_source["claim_boundary"]),
    }
    for key, expected in expected_xfeat.items():
        if external[key] != expected:
            raise ValueError(f"XFeat comparison binding mismatch: {key}")
    table = index["assets"]["external_comparison_table.png"]
    if table["classic_adapters"] != classic_rows:
        raise ValueError("external comparison raster classic rows are not index-bound")
    xfeat_table = table["xfeat_development"]
    if xfeat_table["count"] != xfeat_count:
        raise ValueError("external comparison raster XFeat population mismatch")
    if xfeat_table["resolved_count"] != int(xfeat_metrics["resolved_count"]):
        raise ValueError("external comparison raster XFeat coverage mismatch")
    if xfeat_table["within_5px_rate_exact_fraction"] != xfeat_success:
        raise ValueError("external comparison raster XFeat success mismatch")
    if xfeat_table["claim_boundary"] != str(xfeat_source["claim_boundary"]):
        raise ValueError("external comparison raster XFeat claim boundary mismatch")
    success_source = _frozen_record("iid", index["samples"]["success_iid"]["id"])[1]
    if index["samples"]["success_iid"]["error_px"] != round(float(success_source["error"]), 3):
        raise ValueError("success display error does not match archived row")
    if index["samples"]["failure_scan"]["review_recommended"] is not True:
        raise ValueError("failure review flag mismatch")
    live_diagnostics = json.loads((output / "live_stderr_sanitized.json").read_text(encoding="utf-8"))
    if "runtime_ms" in live_diagnostics:
        raise ValueError("nondeterministic runtime leaked into live diagnostics")
    forbidden_prefixes = (str(ROOT), "/Volumes/External/")
    for string in _walk_strings(index):
        if string.startswith(forbidden_prefixes):
            raise ValueError(f"absolute local path leaked into evidence index: {string}")
    return index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        index = verify_export(args.output.resolve())
        print(f"verified {len(index['assets'])} evidence assets")
        return 0
    index = build_export(args.output.resolve())
    verify_export(args.output.resolve())
    print(f"exported {len(index['assets'])} evidence assets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
