#!/usr/bin/env python3
"""Evaluate registered real MiniTEM Low/GT pairs from the publisher Test split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
from zipfile import ZipFile, ZipInfo

import cv2
import numpy as np
import PIL
from PIL import Image
import scipy

from drift_sense.localizer import LocalizationConfig, localize
from real_imagery.protocol import (
    digest_file,
    digital_crop_pair,
    error_metrics,
    informative_crop_positions,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "paired_tem_source.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=["full", "baseline0"], default=["full", "baseline0"]
    )
    return parser.parse_args()


def git_state() -> tuple[str | None, bool]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def tree_digest(paths: list[Path], domain: bytes) -> str:
    digest = sha256(domain + b"\0")
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def sample_name(path: PurePosixPath) -> str:
    for part in path.parts:
        lowered = part.casefold().replace("_", "").replace("-", "")
        if "kidney" in lowered:
            return "kidney"
        if "calibration" in lowered or lowered in {"grid", "calgrid"}:
            return "calibration_grid"
    test_indices = [index for index, part in enumerate(path.parts) if part.casefold() == "test"]
    if test_indices and test_indices[0] > 0:
        return path.parts[test_indices[0] - 1]
    return "unknown"


def normalized_members(archive: ZipFile) -> dict[str, ZipInfo]:
    output = {}
    for info in archive.infolist():
        if info.is_dir() or info.filename.startswith("__MACOSX/"):
            continue
        key = PurePosixPath(info.filename).as_posix()
        if key.casefold() in {existing.casefold() for existing in output}:
            raise ValueError(f"case-insensitive duplicate archive member: {key}")
        output[key] = info
    return output


def discover_test_pairs(archive: ZipFile) -> list[dict]:
    members = normalized_members(archive)
    image_suffixes = {".tif", ".tiff", ".png"}
    gt_paths = []
    for name in members:
        path = PurePosixPath(name)
        parts = [part.casefold() for part in path.parts]
        if (
            "test" in parts
            and path.parent.name.casefold() == "gt"
            and path.suffix.casefold() in image_suffixes
        ):
            gt_paths.append(path)
    if not gt_paths:
        raise ValueError("archive contains no Test/GT image members")

    rows = []
    member_names = list(members)
    for gt_path in sorted(gt_paths, key=lambda item: item.as_posix().casefold()):
        root = gt_path.parent.parent
        prefixes = [
            (root / "Low" / gt_path.stem).as_posix().casefold() + "/",
            (root / "Low" / gt_path.name).as_posix().casefold() + "/",
        ]
        low_paths = sorted(
            [
                PurePosixPath(name)
                for name in member_names
                if any(name.casefold().startswith(prefix) for prefix in prefixes)
                and PurePosixPath(name).suffix.casefold() in image_suffixes
            ],
            key=lambda item: item.as_posix().casefold(),
        )
        if not low_paths:
            raise ValueError(f"no registered Low images found for {gt_path}")
        for low_index, low_path in enumerate(low_paths):
            rows.append(
                {
                    "sample": sample_name(gt_path),
                    "gt_member": gt_path.as_posix(),
                    "low_member": low_path.as_posix(),
                    "low_index_within_gt": low_index,
                    "position_index": low_index % 5,
                    "gt_info": members[gt_path.as_posix()],
                    "low_info": members[low_path.as_posix()],
                }
            )
    return rows


def read_image_member(archive: ZipFile, info: ZipInfo) -> tuple[np.ndarray, str]:
    content = archive.read(info)
    with Image.open(BytesIO(content)) as image:
        gray = np.asarray(image)
    if gray.ndim != 2 or min(gray.shape) < 128:
        raise ValueError(f"invalid microscopy image member {info.filename}: {gray.shape}")
    if gray.dtype.kind not in {"u", "i", "f"}:
        raise ValueError(
            f"unsupported microscopy intensity dtype for {info.filename}: {gray.dtype}"
        )
    return np.ascontiguousarray(gray), sha256(content).hexdigest()


def prediction_record(method: str, reference: np.ndarray, search: np.ndarray) -> dict:
    config = LocalizationConfig(
        method=method,
        nominal_scale=0.1,
        scale_range=0.006,
        rotation_range=3.0,
    )
    first = localize(reference, search, config)
    second = localize(reference, search, config)
    return {
        "prediction": [float(first.x), float(first.y)],
        "repeat_delta_px": float(math.hypot(first.x - second.x, first.y - second.y)),
        "configuration": asdict(config),
        "diagnostics": {
            "score": float(first.score),
            "selected_score": float(first.selected_score),
            "ambiguity_flag": bool(first.ambiguity_flag),
            "tied_count": int(first.tied_count),
            "selected_scale": float(first.selected_scale),
            "selected_rotation_deg": float(first.selected_rotation_deg),
            "spectral_confidence": float(first.spectral_confidence),
            "runtime_ms": float(first.runtime_ms),
            "pipeline_stages": first.pipeline_stages,
        },
    }


def summarize(records: list[dict], methods: list[str]) -> dict:
    output = {}
    samples = sorted({row["sample"] for row in records})
    for method in methods:
        attempted = [row for row in records if row["method"] == method]
        completed = [row for row in attempted if "error_px" in row]
        runtimes = np.asarray(
            [row["diagnostics"]["runtime_ms"] for row in completed], dtype=np.float64
        )
        output[method] = {
            "attempted_count": len(attempted),
            "completed_count": len(completed),
            "error_count": len(attempted) - len(completed),
            "fallback_count": sum(
                "fallback" in row["diagnostics"]["pipeline_stages"] for row in completed
            ),
            "metrics": error_metrics([row["error_px"] for row in completed], thresholds=(1.0, 5.0)),
            "localizer_runtime_ms": (
                {
                    "count": int(runtimes.size),
                    "total": float(np.sum(runtimes)),
                    "mean": float(np.mean(runtimes)),
                    "median": float(np.median(runtimes)),
                    "p95": float(np.percentile(runtimes, 95, method="linear")),
                    "scope": "first deterministic localize() call; excludes image I/O and repeat call",
                }
                if runtimes.size
                else {"count": 0}
            ),
            "maximum_repeat_delta_px": max(
                (row["repeat_delta_px"] for row in completed), default=None
            ),
            "by_sample": {
                sample: error_metrics(
                    [row["error_px"] for row in completed if row["sample"] == sample],
                    thresholds=(1.0, 5.0),
                )
                for sample in samples
            },
            "by_position": {
                str(index): error_metrics(
                    [row["error_px"] for row in completed if row["position_index"] == index],
                    thresholds=(1.0, 5.0),
                )
                for index in range(5)
            },
        }
    return output


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing report: {args.output}")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    source_files = sorted((ROOT / "src" / "drift_sense").glob("*.py"))
    protocol_files = [Path(__file__).resolve(), HERE / "protocol.py", SOURCE]
    initial_core_sha256 = tree_digest(source_files, b"metralign-registered-tem-core-v1")
    initial_protocol_sha256 = tree_digest(
        protocol_files, b"metralign-registered-tem-protocol-v1"
    )
    manifest = json.loads(SOURCE.read_text())
    expected = manifest["archive"]
    if args.archive.stat().st_size != int(expected["bytes"]):
        raise ValueError("archive size does not match pinned source manifest")
    archive_md5 = digest_file(args.archive, "md5")
    if archive_md5 != expected["md5"]:
        raise ValueError("archive MD5 does not match pinned source manifest")
    archive_sha256 = digest_file(args.archive)

    with ZipFile(args.archive) as archive:
        pair_rows = discover_test_pairs(archive)
        cache: dict[str, tuple[np.ndarray, str]] = {}
        positions_by_gt: dict[str, list[dict]] = {}
        records = []
        inputs: dict[str, dict] = {}
        for pair in pair_rows:
            for key in ("gt", "low"):
                member_key = pair[f"{key}_member"]
                if member_key not in cache:
                    cache[member_key] = read_image_member(archive, pair[f"{key}_info"])
                info = pair[f"{key}_info"]
                inputs[member_key] = {
                    "member": member_key,
                    "bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": cache[member_key][1],
                    "shape_yx": list(cache[member_key][0].shape),
                    "dtype": str(cache[member_key][0].dtype),
                    "intensity_min": float(np.min(cache[member_key][0])),
                    "intensity_max": float(np.max(cache[member_key][0])),
                }
            gt = cache[pair["gt_member"]][0]
            low = cache[pair["low_member"]][0]
            if gt.shape != low.shape:
                raise ValueError(
                    f"registered pair shape mismatch: {pair['gt_member']} {gt.shape} vs "
                    f"{pair['low_member']} {low.shape}"
                )
            if pair["gt_member"] not in positions_by_gt:
                positions_by_gt[pair["gt_member"]] = informative_crop_positions(gt)
            selection = positions_by_gt[pair["gt_member"]][pair["position_index"]]
            position = tuple(selection["position_fraction"])
            reference, ground_truth, construction = digital_crop_pair(
                gt, position, minimum_crop_size=8
            )
            construction["informativeness_selection"] = selection
            for method in args.methods:
                base = {
                    "case_id": f"{pair['gt_member']}::{pair['low_member']}",
                    "sample": pair["sample"],
                    "gt_member": pair["gt_member"],
                    "gt_sha256": cache[pair["gt_member"]][1],
                    "low_member": pair["low_member"],
                    "low_sha256": cache[pair["low_member"]][1],
                    "low_index_within_gt": pair["low_index_within_gt"],
                    "position_index": pair["position_index"],
                    "position_fraction": list(position),
                    "method": method,
                    "ground_truth_kind": "digital crop coordinate in publisher-registered GT/Low frame",
                    "ground_truth": [float(ground_truth[0]), float(ground_truth[1])],
                    "construction": construction,
                }
                try:
                    prediction = prediction_record(method, reference, low)
                    error = float(
                        math.hypot(
                            prediction["prediction"][0] - ground_truth[0],
                            prediction["prediction"][1] - ground_truth[1],
                        )
                    )
                    records.append({**base, **prediction, "error_px": error})
                except Exception as exc:
                    records.append({**base, "error": f"{type(exc).__name__}: {exc}"})

    commit, dirty = git_state()
    if tree_digest(source_files, b"metralign-registered-tem-core-v1") != initial_core_sha256:
        raise RuntimeError("core implementation changed during evaluation")
    if (
        tree_digest(protocol_files, b"metralign-registered-tem-protocol-v1")
        != initial_protocol_sha256
    ):
        raise RuntimeError("registered-TEM protocol bundle changed during evaluation")
    report = {
        "schema_version": 1,
        "protocol": "registered-real-minitem-crop-localization-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "coordinate_truth": "Coordinates are exact only in the coordinate system of the publisher-registered Low/GT pair and the deterministic crop construction. Residual publisher-registration error is unknown.",
            "acquisition": "GT and Low are independent real acquisitions of the same field, but have been selected, ECC-registered, and intersection-cropped by the publisher. The evaluator preserves the archive's native uint16 intensities.",
            "task_scope": "The equal-field-of-view publisher pair is converted to a reference-in-search task by digital cropping. This is not native cross-magnification or microscope-stage ground truth.",
            "benchmark_scope": "This development result is separate from and must not be pooled with the frozen synthetic reporting split.",
        },
        "source": manifest["dataset"],
        "source_manifest_sha256": digest_file(SOURCE),
        "archive": {
            "name": expected["name"],
            "bytes": args.archive.stat().st_size,
            "md5": archive_md5,
            "sha256": archive_sha256,
        },
        "selection": {
            "publisher_split": "Test",
            "eligible_pair_count": len(pair_rows),
            "rule": manifest["selection_rule"],
            "position_strategy": "For each GT, rank a fixed 9x9 interior grid by GT-only mean absolute Gaussian high-pass residual; greedily retain the five highest-scoring non-overlapping crops with row/column tie breaks.",
            "low_frame_position_assignment": "zero-based lexicographic Low member index selects the same-rank informative GT crop",
            "excluded_from_evaluation": ["Train", "Val", "GT_hr", "RawData.zip"],
        },
        "preprocessing": {
            "intensity": "Native single-channel TIFF array retained without clipping or per-image rescaling",
            "reference_construction": "Nominal 0.1x GT crop enlarged to the registered frame shape with OpenCV INTER_LANCZOS4",
            "minimum_crop_size_px": 8,
        },
        "input_members": [inputs[key] for key in sorted(inputs, key=str.casefold)],
        "artifact_binding": {
            "git_commit": commit,
            "working_tree_dirty": dirty,
            "core_implementation_sha256": initial_core_sha256,
            "protocol_bundle_sha256": initial_protocol_sha256,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
        "records": records,
        "summary": summarize(records, args.methods),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
