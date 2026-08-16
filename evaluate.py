#!/usr/bin/env python3
"""Evaluate localization with explicit accuracy and runtime definitions."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from PIL import Image, __version__ as pillow_version

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from drift_sense.dataset import load_manifest
from drift_sense.failures import classify_failure
from drift_sense.localizer import LocalizationConfig, localize


METHODS = ["baseline0", "multiscale", "structure_gradient", "structure_residual", "full"]
SUCCESS_THRESHOLDS_PX = (0.5, 1.0, 2.0, 3.0, 5.0)
EVIDENCE_CHANNELS = ("structural", "gradient", "raw")
SUBPIXEL_REFINEMENTS = ("parabolic", "dft", "none")
REQUIRED_RECORD_FIELDS = {
    "id",
    "architecture",
    "suite",
    "reference",
    "search",
    "center_x",
    "center_y",
    "actual_scale",
    "rotation_deg",
    "search_geometry",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_hashes_sha256(named_hashes: dict[str, str], manifest_sha256: str) -> str:
    """Bind a manifest and its named image files into one deterministic digest."""
    digest = sha256()
    digest.update(b"drift-sense-dataset-v1\0")
    digest.update(manifest_sha256.encode("ascii"))
    digest.update(b"\0")
    for name, value in sorted(named_hashes.items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def implementation_sha256(repo_root: Path) -> str:
    """Fingerprint every checked-in file that can affect generation or evaluation."""
    root = repo_root.resolve()
    paths = {
        path.resolve()
        for pattern in ("*.py", "src/drift_sense/**/*.py", "configs/*.json")
        for path in root.glob(pattern)
        if path.is_file()
    }
    paths.update(
        path.resolve()
        for name in ("pyproject.toml", "requirements.txt")
        if (path := root / name).is_file()
    )
    if not paths:
        raise ValueError(f"no implementation files found under {root}")
    digest = sha256()
    digest.update(b"drift-sense-implementation-v1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def git_state(repo_root: Path) -> tuple[str | None, bool]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def _dataset_path(root: Path, value: object, sample_id: str, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"sample {sample_id} has absolute {field} path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"sample {sample_id} has out-of-tree {field} path: {relative}")
    return resolved


def input_image_hashes(root: Path, records: list[dict]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        sample_id = str(record["id"])
        for field in ("reference", "search"):
            relative = Path(str(record[field])).as_posix()
            path = _dataset_path(root, record[field], sample_id, field)
            digest = _sha256_file(path)
            previous = hashes.setdefault(relative, digest)
            if previous != digest:  # pragma: no cover - one path cannot change within one call
                raise ValueError(f"input image changed while hashing: {relative}")
    return dict(sorted(hashes.items()))


def build_artifact_binding(
    manifest_path: Path,
    records: list[dict],
    repo_root: Path,
) -> dict:
    manifest_path = manifest_path.resolve()
    manifest_digest = _sha256_file(manifest_path)
    image_hashes = input_image_hashes(manifest_path.parent, records)
    commit, dirty = git_state(repo_root)
    return {
        "manifest_sha256": manifest_digest,
        "input_images_sha256": image_hashes,
        "dataset_sha256": _named_hashes_sha256(image_hashes, manifest_digest),
        "implementation_sha256": implementation_sha256(repo_root),
        "git_commit": commit,
        "working_tree_dirty": dirty,
    }


def verify_artifact_binding(
    report: dict,
    manifest_path: Path,
    records: list[dict],
    repo_root: Path,
    *,
    require_current_code: bool,
) -> dict:
    """Fail closed if a report is detached from its data or implementation."""
    binding = report.get("artifact_binding")
    if not isinstance(binding, dict):
        raise ValueError("report is missing artifact_binding")
    required = {
        "manifest_sha256",
        "input_images_sha256",
        "dataset_sha256",
        "implementation_sha256",
        "git_commit",
        "working_tree_dirty",
    }
    missing = required.difference(binding)
    if missing:
        raise ValueError(f"report artifact_binding is missing: {', '.join(sorted(missing))}")
    hash_fields = ("manifest_sha256", "dataset_sha256", "implementation_sha256")
    for field in hash_fields:
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"report artifact_binding has invalid {field}")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"report artifact_binding has invalid {field}") from exc
    image_hashes = binding.get("input_images_sha256")
    if not isinstance(image_hashes, dict) or not image_hashes:
        raise ValueError("report artifact_binding has invalid input_images_sha256")
    for name, value in image_hashes.items():
        if not isinstance(name, str) or not isinstance(value, str) or len(value) != 64:
            raise ValueError("report artifact_binding has invalid input image hash")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("report artifact_binding has invalid input image hash") from exc
    if binding.get("git_commit") is not None and not isinstance(binding.get("git_commit"), str):
        raise ValueError("report artifact_binding has invalid git_commit")
    if not isinstance(binding.get("working_tree_dirty"), bool):
        raise ValueError("report artifact_binding has invalid working_tree_dirty")
    expected = build_artifact_binding(manifest_path, records, repo_root)
    fields = ["manifest_sha256", "input_images_sha256", "dataset_sha256"]
    if require_current_code:
        fields.extend(("implementation_sha256", "git_commit"))
    for field in fields:
        if binding.get(field) != expected[field]:
            raise ValueError(f"report artifact binding mismatch: {field}")
    if report.get("manifest_sha256") != expected["manifest_sha256"]:
        raise ValueError("report top-level manifest hash does not match the dataset")
    return expected


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _finite_values(rows: list[dict], field: str) -> np.ndarray:
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty evaluation group")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite value in evaluation field {field!r}")
    return values


def metrics(rows: list[dict]) -> dict:
    """Summarize inclusive Euclidean-error thresholds and inference timing.

    ``runtime_ms`` is measured by the evaluator immediately around ``localize``;
    it excludes image decoding and report construction. Optional timing fields are
    summarized when all input rows provide them.
    """
    errors = _finite_values(rows, "error")
    runtimes = _finite_values(rows, "runtime_ms")
    result = {
        f"success_le_{limit:g}px": float(np.count_nonzero(errors <= limit) / errors.size)
        for limit in SUCCESS_THRESHOLDS_PX
    }
    result.update(
        {
            "count": int(errors.size),
            "failure_gt_5px_count": int(np.count_nonzero(errors > 5.0)),
            "mean_error_px": float(np.mean(errors)),
            "median_error_px": float(np.median(errors)),
            "p90_error_px": float(np.percentile(errors, 90, method="linear")),
            "p95_error_px": float(np.percentile(errors, 95, method="linear")),
            "p99_error_px": float(np.percentile(errors, 99, method="linear")),
            "max_error_px": float(np.max(errors)),
            "mean_runtime_ms": float(np.mean(runtimes)),
            "median_runtime_ms": float(np.median(runtimes)),
            "p95_runtime_ms": float(np.percentile(runtimes, 95, method="linear")),
        }
    )
    optional_timings = {
        "image_io_ms": "image_io",
        "localizer_runtime_ms": "localizer_runtime",
        "sample_wall_ms": "sample_wall",
    }
    for field, output_name in optional_timings.items():
        if all(field in row for row in rows):
            values = _finite_values(rows, field)
            result[f"mean_{output_name}_ms"] = float(np.mean(values))
            result[f"median_{output_name}_ms"] = float(np.median(values))
            result[f"p95_{output_name}_ms"] = float(
                np.percentile(values, 95, method="linear")
            )
    return result


def validate_records(
    root: Path,
    records: list[dict],
    *,
    expected_image_size: int | None = None,
) -> None:
    if not records:
        raise ValueError("manifest contains no records")
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        missing = REQUIRED_RECORD_FIELDS.difference(record)
        if missing:
            raise ValueError(f"manifest record {index} is missing: {', '.join(sorted(missing))}")
        sample_id = str(record["id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample id in manifest: {sample_id}")
        seen_ids.add(sample_id)
        for coordinate in ("center_x", "center_y"):
            if not math.isfinite(float(record[coordinate])):
                raise ValueError(f"sample {sample_id} has non-finite {coordinate}")
        actual_scale = float(record["actual_scale"])
        rotation_deg = float(record["rotation_deg"])
        if not math.isfinite(actual_scale) or actual_scale <= 0:
            raise ValueError(f"sample {sample_id} has invalid actual_scale")
        if not math.isfinite(rotation_deg):
            raise ValueError(f"sample {sample_id} has invalid rotation_deg")
        search_geometry = record["search_geometry"]
        if not isinstance(search_geometry, dict):
            raise ValueError(f"sample {sample_id} has invalid search_geometry")
        try:
            search_width = int(search_geometry["width"])
            search_height = int(search_geometry["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sample {sample_id} has incomplete search_geometry") from exc
        if min(search_width, search_height) < 1:
            raise ValueError(f"sample {sample_id} has invalid search geometry dimensions")
        dimensions: dict[str, tuple[int, int]] = {}
        for image_field in ("reference", "search"):
            image_path = _dataset_path(root, record[image_field], sample_id, image_field)
            if not image_path.is_file():
                raise ValueError(f"sample {sample_id} is missing {image_field}: {image_path}")
            try:
                with Image.open(image_path) as image:
                    if image.mode not in {"L", "I", "I;16", "F"}:
                        raise ValueError(
                            f"sample {sample_id} {image_field} is not a grayscale image"
                        )
                    dimensions[image_field] = image.size
            except OSError as exc:
                raise ValueError(
                    f"sample {sample_id} has unreadable {image_field}: {image_path}"
                ) from exc
        if dimensions["reference"] != dimensions["search"]:
            raise ValueError(f"sample {sample_id} reference/search dimensions differ")
        width, height = dimensions["search"]
        if (width, height) != (search_width, search_height):
            raise ValueError(f"sample {sample_id} search image disagrees with search_geometry")
        if expected_image_size is not None and (width, height) != (
            expected_image_size,
            expected_image_size,
        ):
            raise ValueError(
                f"sample {sample_id} has image size {width}x{height}, "
                f"expected {expected_image_size}x{expected_image_size}"
            )
        if "image_size" in record and int(record["image_size"]) != width:
            raise ValueError(f"sample {sample_id} image_size disagrees with image files")


def _load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def pipeline_configuration(args: argparse.Namespace) -> dict:
    """Return the artifact-bound full-pipeline controls for a parsed CLI."""
    return {
        "enable_phase_calibration": bool(getattr(args, "phase_calibration", True)),
        "periodic_evidence_channel": getattr(args, "evidence_channel", "structural"),
        "enable_spatial_residual": bool(getattr(args, "spatial_residual", True)),
        "enable_lattice_grouping": bool(getattr(args, "lattice_grouping", True)),
        "enable_ambiguity_rule": bool(getattr(args, "ambiguity_rule", True)),
        "subpixel_refinement": getattr(args, "subpixel_refinement", "parabolic"),
    }


def _group_metrics(rows: list[dict]) -> dict[str, dict]:
    groups = {"all": metrics(rows)}
    for field in ("architecture", "suite", "difficulty"):
        values = sorted({row[field] for row in rows if row.get(field) is not None})
        for value in values:
            groups[f"{field}:{value}"] = metrics([row for row in rows if row.get(field) == value])
    architectures = sorted({row["architecture"] for row in rows})
    suites = sorted({row["suite"] for row in rows})
    for architecture in architectures:
        for suite in suites:
            selected = [
                row
                for row in rows
                if row["architecture"] == architecture and row["suite"] == suite
            ]
            if selected:
                groups[f"architecture_suite:{architecture}|{suite}"] = metrics(selected)
    return groups


def evaluate_method(root: Path, records: list[dict], method: str, args: argparse.Namespace) -> dict:
    rows = []
    failures: Counter[str] = Counter()
    for index, record in enumerate(records):
        sample_start_ns = time.perf_counter_ns()
        io_start_ns = time.perf_counter_ns()
        reference = _load_grayscale(root / record["reference"])
        search = _load_grayscale(root / record["search"])
        image_io_ms = _elapsed_ms(io_start_ns)
        inference_start_ns = time.perf_counter_ns()
        result = localize(
            reference,
            search,
            LocalizationConfig(
                method=method,
                top_k=args.top_k,
                scale_range=args.scale_range,
                rotation_range=args.rotation_range,
                **pipeline_configuration(args),
            ),
        )
        inference_wall_ms = _elapsed_ms(inference_start_ns)
        predicted_x, predicted_y = float(result.x), float(result.y)
        if not (math.isfinite(predicted_x) and math.isfinite(predicted_y)):
            raise ValueError(f"method {method} produced a non-finite prediction for {record['id']}")
        localizer_runtime_ms = float(result.runtime_ms)
        if not math.isfinite(localizer_runtime_ms) or localizer_runtime_ms < 0:
            raise ValueError(f"method {method} produced invalid runtime for {record['id']}")
        error = float(np.hypot(predicted_x - record["center_x"], predicted_y - record["center_y"]))
        prediction = result.to_dict()
        category = classify_failure(record, prediction, error)
        if category:
            failures[category] += 1
        row = {
            "id": record["id"],
            "architecture": record["architecture"],
            "suite": record["suite"],
            "difficulty": record.get("difficulty"),
            "seed": record.get("seed"),
            "ground_truth": [float(record["center_x"]), float(record["center_y"])],
            "prediction": [predicted_x, predicted_y],
            "error": error,
            # Compatibility field. Its precise scope is declared at report level.
            "runtime_ms": inference_wall_ms,
            "image_io_ms": image_io_ms,
            "inference_wall_ms": inference_wall_ms,
            "localizer_runtime_ms": localizer_runtime_ms,
            "failure_category": category,
            "diagnostics": prediction,
        }
        row["sample_wall_ms"] = _elapsed_ms(sample_start_ns)
        rows.append(row)
        if not args.quiet:
            print(f"{method}: {index + 1}/{len(records)}", file=sys.stderr)
    return {"metrics": _group_metrics(rows), "failure_counts": dict(failures), "samples": rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS + ["all"], default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--scale-range", type=float, default=0.006)
    parser.add_argument("--rotation-range", type=float, default=3.0)
    parser.add_argument(
        "--phase-calibration", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evidence-channel", choices=EVIDENCE_CHANNELS, default="structural"
    )
    parser.add_argument(
        "--spatial-residual", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lattice-grouping", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ambiguity-rule", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--subpixel-refinement",
        choices=SUBPIXEL_REFINEMENTS,
        default="parabolic",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true", help="suppress per-sample progress")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.scale_range < 0 or args.rotation_range < 0:
        parser.error("search ranges must be non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def _manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.jsonl" if data_dir.is_dir() else data_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = _manifest_path(args.data_dir).resolve()
    try:
        all_records = load_manifest(manifest_path)
        root = manifest_path.parent
        validate_records(root, all_records)
        artifact_binding = build_artifact_binding(
            manifest_path,
            all_records,
            Path(__file__).resolve().parent,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    records = all_records[: args.limit] if args.limit is not None else all_records
    methods = METHODS if args.method == "all" else [args.method]
    started_ns = time.perf_counter_ns()
    report = {
        "schema_version": 2,
        "manifest": str(manifest_path),
        "manifest_sha256": artifact_binding["manifest_sha256"],
        "artifact_binding": artifact_binding,
        "dataset_record_count": len(all_records),
        "evaluated_record_count": len(records),
        "configuration": {
            "requested_method": args.method,
            "top_k": args.top_k,
            "scale_range": args.scale_range,
            "rotation_range": args.rotation_range,
            "limit": args.limit,
            **pipeline_configuration(args),
        },
        "metric_definition": {
            "coordinate_error": "Euclidean distance in search-image pixels",
            "success_comparison": "error <= threshold",
            "percentile_method": "NumPy linear",
            "primary_runtime_field": "runtime_ms",
            "runtime_scope": "evaluator wall clock around localize(); excludes image I/O and report construction",
            "localizer_runtime_field": "localizer_runtime_ms",
            "image_io_field": "image_io_ms",
            "timer": "time.perf_counter_ns",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported by OS",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__ if cv2 is not None else "SciPy fallback",
            "pillow": pillow_version,
        },
        "methods": {method: evaluate_method(root, records, method, args) for method in methods},
    }
    report["wall_time_seconds"] = _elapsed_ms(started_ns) / 1000.0
    try:
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        print(f"error: report contains invalid numeric data: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
