#!/usr/bin/env python3
"""Evaluate Metralign on a pinned subset of openly licensed real SEM images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
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
    DIGITAL_POSITIONS,
    digest_file,
    digital_crop_pair,
    error_metrics,
    feature_registration_proxy,
    read_sem_content,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = HERE / "sources.json"
CARINTHIA_SOURCE = HERE / "carinthia_source.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--carinthia-archive", type=Path)
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


def implementation_sha256() -> str:
    paths = sorted((ROOT / "src" / "drift_sense").glob("*.py"))
    digest = sha256()
    digest.update(b"metralign-real-imagery-core-v1\0")
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def protocol_bundle_sha256() -> str:
    paths = [Path(__file__).resolve(), HERE / "protocol.py", SOURCES, CARINTHIA_SOURCE]
    digest = sha256(b"metralign-real-sem-protocol-v1\0")
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def source_rows(manifest: dict, data_dir: Path) -> list[dict]:
    rows = []
    for dataset in manifest["datasets"]:
        for area in dataset["areas"]:
            files = {}
            for record in area["files"]:
                path = data_dir / dataset["id"] / area["area"] / record["name"]
                if not path.is_file():
                    raise FileNotFoundError(f"missing source file: {path}")
                if path.stat().st_size != int(record["size"]):
                    raise ValueError(f"size mismatch: {path}")
                if digest_file(path, "md5") != record["md5"]:
                    raise ValueError(f"MD5 mismatch: {path}")
                files[int(record["magnification_k"])] = {
                    **record,
                    "path": path,
                    "sha256": digest_file(path),
                }
            rows.append(
                {
                    "dataset": dataset,
                    "area": area["area"],
                    "files": files,
                }
            )
    return rows


def carinthia_selection(archive_path: Path, manifest: dict) -> tuple[dict, list[dict]]:
    """Verify the archive and return the fixed balanced 24-image subset."""
    expected = manifest["archive"]
    if archive_path.stat().st_size != int(expected["bytes"]):
        raise ValueError("Carinthia archive size does not match the pinned source")
    if digest_file(archive_path, "md5") != expected["md5"]:
        raise ValueError("Carinthia archive MD5 does not match the pinned source")
    archive_sha256 = digest_file(archive_path)
    if archive_sha256 != expected["sha256"]:
        raise ValueError("Carinthia archive SHA-256 does not match the pinned source")

    index_spec = manifest["index"]
    with ZipFile(archive_path) as archive:
        infos: dict[str, ZipInfo] = {}
        casefolded: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = PurePosixPath(info.filename)
            normalized = member.as_posix()
            if member.is_absolute() or ".." in member.parts or normalized != info.filename:
                raise ValueError(f"unsafe Carinthia archive member: {info.filename}")
            folded = normalized.casefold()
            if folded in casefolded:
                raise ValueError(f"case-insensitive duplicate Carinthia member: {normalized}")
            casefolded.add(folded)
            infos[normalized] = info
        index_member = index_spec["member"]
        if index_member not in infos:
            raise ValueError("Carinthia archive is missing its CSV index")
        index_content = archive.read(index_member)
        if len(index_content) != int(index_spec["bytes"]):
            raise ValueError("Carinthia CSV byte count mismatch")
        if sha256(index_content).hexdigest() != index_spec["sha256"]:
            raise ValueError("Carinthia CSV SHA-256 mismatch")
        reader = csv.DictReader(
            StringIO(index_content.decode("utf-8-sig")),
            delimiter=index_spec["delimiter"],
        )
        if reader.fieldnames != index_spec["columns"]:
            raise ValueError("Carinthia CSV columns differ from the pinned schema")
        rows = list(reader)
        if len(rows) != int(manifest["dataset"]["record_image_count"]):
            raise ValueError("Carinthia CSV record count mismatch")
        csv_paths = [row["image_path"] for row in rows]
        if len(csv_paths) != len(set(csv_paths)):
            raise ValueError("Carinthia CSV contains duplicate image paths")
        archive_images = {
            name
            for name in infos
            if name.startswith("data/images/") and name.casefold().endswith(".jpg")
        }
        if set(csv_paths) != archive_images:
            raise ValueError("Carinthia CSV/image member coverage mismatch")
        labels = set(manifest["dataset"]["class_labels"])
        if {row["label"] for row in rows} != labels:
            raise ValueError("Carinthia CSV label set mismatch")
        for row in rows:
            if PurePosixPath(row["image_path"]).name != row["file_name"]:
                raise ValueError("Carinthia CSV file_name/image_path mismatch")

        selected: list[dict] = []
        per_class = int(manifest["selection"]["images_per_class"])
        positions = manifest["selection"]["positions"]
        if len(positions) != per_class:
            raise ValueError("Carinthia selection position count mismatch")
        for label in sorted(labels, key=int):
            candidates = [row for row in rows if row["label"] == label]
            if len(candidates) < per_class:
                raise ValueError(f"Carinthia label {label} has too few images")
            ranked = sorted(
                candidates,
                key=lambda row: (
                    sha256(
                        b"metralign-carinthia-balanced-v1\0"
                        + row["image_path"].encode("utf-8")
                    ).hexdigest(),
                    row["image_path"],
                ),
            )[:per_class]
            for rank, (row, position) in enumerate(zip(ranked, positions, strict=True)):
                content = archive.read(row["image_path"])
                with Image.open(BytesIO(content)) as image:
                    if image.mode != "L":
                        raise ValueError(
                            f"Carinthia image is not native grayscale: {row['image_path']}"
                        )
                    pixels = np.array(image, copy=True)
                if pixels.ndim != 2 or min(pixels.shape) < 128:
                    raise ValueError(f"invalid Carinthia image shape: {pixels.shape}")
                info = infos[row["image_path"]]
                selected.append(
                    {
                        "class_label": label,
                        "selection_rank_within_class": rank,
                        "position_fraction": [float(position[0]), float(position[1])],
                        "member": row["image_path"],
                        "member_bytes": int(info.file_size),
                        "member_crc32": f"{info.CRC:08x}",
                        "member_sha256": sha256(content).hexdigest(),
                        "shape_yx": list(pixels.shape),
                        "dtype": str(pixels.dtype),
                        "pixels": pixels,
                    }
                )
    archive_binding = {
        "name": expected["name"],
        "bytes": int(expected["bytes"]),
        "md5": expected["md5"],
        "sha256": archive_sha256,
        "csv_member": index_spec["member"],
        "csv_bytes": int(index_spec["bytes"]),
        "csv_sha256": index_spec["sha256"],
        "dataset_record_count": len(rows),
    }
    return archive_binding, selected


def prediction_record(method: str, reference: np.ndarray, search: np.ndarray, nominal: float) -> dict:
    scale_range = 0.006 if math.isclose(nominal, 0.1) else 0.12 * nominal
    config = LocalizationConfig(
        method=method,
        nominal_scale=nominal,
        scale_range=scale_range,
        rotation_range=3.0 if math.isclose(nominal, 0.1) else 5.0,
    )
    first = localize(reference, search, config)
    second = localize(reference, search, config)
    repeat_delta = float(math.hypot(first.x - second.x, first.y - second.y))
    return {
        "prediction": [float(first.x), float(first.y)],
        "repeat_delta_px": repeat_delta,
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


def runtime_summary(records: list[dict]) -> dict:
    values = np.asarray(
        [row["diagnostics"]["runtime_ms"] for row in records], dtype=np.float64
    )
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "total": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95, method="linear")),
        "scope": "first deterministic localize() call; excludes image I/O and repeat call",
    }


def summarize_digital(records: list[dict], methods: list[str]) -> dict:
    output = {}
    for method in methods:
        attempted_rows = [row for row in records if row["method"] == method]
        method_rows = [row for row in attempted_rows if "error_px" in row]
        output[method] = {
            "attempted_count": len(attempted_rows),
            "completed_count": len(method_rows),
            "error_count": len(attempted_rows) - len(method_rows),
            "fallback_count": sum(
                "fallback" in row["diagnostics"]["pipeline_stages"] for row in method_rows
            ),
            "localizer_runtime_ms": runtime_summary(method_rows),
            "metrics": error_metrics([row["error_px"] for row in method_rows], thresholds=(1.0, 5.0)),
            "maximum_repeat_delta_px": max((row["repeat_delta_px"] for row in method_rows), default=None),
            "by_category": {
                category: error_metrics(
                    [row["error_px"] for row in method_rows if row["category"] == category],
                    thresholds=(1.0, 5.0),
                )
                for category in ("ordered", "disordered")
            },
        }
    return output


def summarize_native(records: list[dict], methods: list[str]) -> dict:
    output = {
        "pair_count": len({row["pair_id"] for row in records}),
        "proxy_accepted_pair_count": len(
            {row["pair_id"] for row in records if row["proxy_accepted"]}
        ),
        "methods": {},
    }
    for method in methods:
        attempted_rows = [row for row in records if row["method"] == method]
        completed_rows = [row for row in attempted_rows if "prediction" in row]
        rows = [
            row
            for row in completed_rows
            if row["proxy_accepted"] and "proxy_disagreement_px" in row
        ]
        output["methods"][method] = {
            "attempted_count": len(attempted_rows),
            "completed_count": len(completed_rows),
            "error_count": len(attempted_rows) - len(completed_rows),
            "fallback_count": sum(
                "fallback" in row["diagnostics"]["pipeline_stages"] for row in completed_rows
            ),
            "localizer_runtime_ms": runtime_summary(completed_rows),
            "proxy_accepted_completed_count": len(rows),
            "agreement_metrics": error_metrics(
                [row["proxy_disagreement_px"] for row in rows], thresholds=(5.0, 20.0)
            ),
            "maximum_repeat_delta_px": max((row["repeat_delta_px"] for row in rows), default=None),
        }
    return output


def summarize_carinthia(records: list[dict], methods: list[str]) -> dict:
    output = {}
    labels = sorted({row["class_label"] for row in records}, key=int)
    for method in methods:
        attempted = [row for row in records if row["method"] == method]
        completed = [row for row in attempted if "error_px" in row]
        output[method] = {
            "attempted_count": len(attempted),
            "completed_count": len(completed),
            "error_count": len(attempted) - len(completed),
            "fallback_count": sum(
                "fallback" in row["diagnostics"]["pipeline_stages"] for row in completed
            ),
            "localizer_runtime_ms": runtime_summary(completed),
            "metrics": error_metrics(
                [row["error_px"] for row in completed], thresholds=(1.0, 5.0)
            ),
            "maximum_repeat_delta_px": max(
                (row["repeat_delta_px"] for row in completed), default=None
            ),
            "by_class_label": {
                label: error_metrics(
                    [row["error_px"] for row in completed if row["class_label"] == label],
                    thresholds=(1.0, 5.0),
                )
                for label in labels
            },
        }
    return output


def evaluate_carinthia(selected: list[dict], methods: list[str]) -> list[dict]:
    records: list[dict] = []
    for source in selected:
        search = source["pixels"]
        position = tuple(source["position_fraction"])
        reference, ground_truth, construction = digital_crop_pair(search, position)
        for method in methods:
            base = {
                "case_id": (
                    f"carinthia/label-{source['class_label']}/"
                    f"{PurePosixPath(source['member']).stem}"
                ),
                "dataset": "carinthia_production_wafer_sem_v1",
                "class_label": source["class_label"],
                "selection_rank_within_class": source["selection_rank_within_class"],
                "member": source["member"],
                "member_bytes": source["member_bytes"],
                "member_crc32": source["member_crc32"],
                "member_sha256": source["member_sha256"],
                "shape_yx": source["shape_yx"],
                "dtype": source["dtype"],
                "method": method,
                "ground_truth_kind": "exact digital crop construction within one real SEM image",
                "ground_truth": [float(ground_truth[0]), float(ground_truth[1])],
                "construction": construction,
            }
            try:
                prediction = prediction_record(method, reference, search, 0.1)
                error = float(
                    math.hypot(
                        prediction["prediction"][0] - ground_truth[0],
                        prediction["prediction"][1] - ground_truth[1],
                    )
                )
                records.append({**base, **prediction, "error_px": error})
            except Exception as exc:
                records.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    return records


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace existing report: {args.output}")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must not contain duplicates")
    initial_core_sha256 = implementation_sha256()
    initial_protocol_sha256 = protocol_bundle_sha256()
    manifest = json.loads(SOURCES.read_text())
    rows = source_rows(manifest, args.data_dir)
    crop_spec = manifest["content_crop"]
    carinthia_manifest = None
    carinthia_archive_binding = None
    carinthia_selected: list[dict] = []
    if args.carinthia_archive:
        carinthia_manifest = json.loads(CARINTHIA_SOURCE.read_text())
        carinthia_archive_binding, carinthia_selected = carinthia_selection(
            args.carinthia_archive, carinthia_manifest
        )

    digital_records = []
    native_records = []
    for row in rows:
        dataset = row["dataset"]
        area = row["area"]
        images = {
            magnification: read_sem_content(record["path"], crop_spec)
            for magnification, record in row["files"].items()
        }
        search = images[50]
        for position_index, position in enumerate(DIGITAL_POSITIONS):
            reference, ground_truth, construction = digital_crop_pair(search, position)
            for method in args.methods:
                base = {
                    "case_id": f"{dataset['id']}/{area}/digital-{position_index}",
                    "dataset": dataset["id"],
                    "category": dataset["category"],
                    "area": area,
                    "source_magnification_k": 50,
                    "source_sha256": row["files"][50]["sha256"],
                    "method": method,
                    "ground_truth_kind": "exact digital crop construction",
                    "ground_truth": [float(ground_truth[0]), float(ground_truth[1])],
                    "construction": construction,
                }
                try:
                    prediction = prediction_record(method, reference, search, 0.1)
                    error = float(
                        math.hypot(
                            prediction["prediction"][0] - ground_truth[0],
                            prediction["prediction"][1] - ground_truth[1],
                        )
                    )
                    digital_records.append({**base, **prediction, "error_px": error})
                except Exception as exc:
                    digital_records.append({**base, "error": f"{type(exc).__name__}: {exc}"})

        for reference_mag, search_mag in ((100, 50), (200, 100), (200, 50)):
            reference = images[reference_mag]
            native_search = images[search_mag]
            nominal = search_mag / reference_mag
            proxy = feature_registration_proxy(reference, native_search, nominal)
            pair_id = f"{dataset['id']}/{area}/{reference_mag}k-to-{search_mag}k"
            for method in args.methods:
                base = {
                    "pair_id": pair_id,
                    "dataset": dataset["id"],
                    "category": dataset["category"],
                    "area": area,
                    "reference_magnification_k": reference_mag,
                    "search_magnification_k": search_mag,
                    "nominal_scale_from_magnification_labels": nominal,
                    "reference_sha256": row["files"][reference_mag]["sha256"],
                    "search_sha256": row["files"][search_mag]["sha256"],
                    "method": method,
                    "proxy_accepted": bool(proxy["accepted_for_agreement"]),
                    "feature_registration_proxy": proxy,
                }
                try:
                    prediction = prediction_record(method, reference, native_search, nominal)
                    record = {**base, **prediction}
                    if proxy["accepted_for_agreement"]:
                        target = proxy["reference_center_in_search_px"]
                        record["proxy_disagreement_px"] = float(
                            math.hypot(
                                prediction["prediction"][0] - target[0],
                                prediction["prediction"][1] - target[1],
                            )
                        )
                    native_records.append(record)
                except Exception as exc:
                    native_records.append({**base, "error": f"{type(exc).__name__}: {exc}"})

    carinthia_records = (
        evaluate_carinthia(carinthia_selected, args.methods) if carinthia_selected else []
    )

    commit, dirty = git_state()
    if implementation_sha256() != initial_core_sha256:
        raise RuntimeError("core implementation changed during evaluation")
    if protocol_bundle_sha256() != initial_protocol_sha256:
        raise RuntimeError("real-SEM protocol bundle changed during evaluation")
    report = {
        "schema_version": 1,
        "protocol": "real-sem-self-consistency-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "digital_crop": "Exact coordinates come only from deterministic crop-and-resize transformations of already acquired real SEM pixels. They are not microscope stage ground truth and do not represent independent captures.",
            "native_pair": "Same-area native magnification pairs are compared to an independently computed feature-registration proxy. Proxy agreement is not physical ground truth or accuracy.",
            "carinthia": (
                carinthia_manifest["claim_boundary"] if carinthia_manifest else "not evaluated"
            ),
            "frozen_benchmark": "This development track is separate from and cannot be pooled with the frozen synthetic reporting split.",
        },
        "sources": [
            {
                key: dataset[key]
                for key in ("id", "title", "doi", "landing_url", "license", "license_url", "authors", "category")
            }
            for dataset in manifest["datasets"]
        ],
        "source_manifest_sha256": sha256(SOURCES.read_bytes()).hexdigest(),
        "source_selection_rule": manifest["selection_rule"],
        "input_files": [
            {
                "dataset": row["dataset"]["id"],
                "area": row["area"],
                "magnification_k": mag,
                "name": record["name"],
                "bytes": record["size"],
                "md5": record["md5"],
                "sha256": record["sha256"],
            }
            for row in rows
            for mag, record in sorted(row["files"].items())
        ],
        "preprocessing": {
            **crop_spec,
            "remaining_content_size": [
                crop_spec["expected_width_px"],
                crop_spec["expected_height_px"] - crop_spec["caption_rows_removed_from_bottom"],
            ],
            "intensity": "Pillow conversion to 8-bit grayscale",
        },
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
        "digital_crop_self_consistency": {
            "description": "Five fixed positions in each 50k image; 0.1x crops are enlarged with Lanczos and localized back into the unchanged acquired image.",
            "records": digital_records,
            "summary": summarize_digital(digital_records, args.methods),
        },
        "native_multimagnification_agreement": {
            "description": "Native 100k→50k, 200k→100k, and 200k→50k pairs sharing publisher area codes; agreement measured only where the feature proxy passes its fixed quality gate.",
            "records": native_records,
            "summary": summarize_native(native_records, args.methods),
        },
    }
    if carinthia_manifest and carinthia_archive_binding:
        report["carinthia_semiconductor_sem_self_consistency"] = {
            "description": (
                "Balanced 24-image supplement: four path-hash-selected images from each "
                "of six publisher labels, one fixed 0.1x digital crop per image."
            ),
            "source": carinthia_manifest["dataset"],
            "source_manifest_sha256": digest_file(CARINTHIA_SOURCE),
            "archive": carinthia_archive_binding,
            "selection": carinthia_manifest["selection"],
            "input_members": [
                {key: value for key, value in source.items() if key != "pixels"}
                for source in carinthia_selected
            ],
            "records": carinthia_records,
            "summary": summarize_carinthia(carinthia_records, args.methods),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    console_summary = {
        "output": str(args.output),
        "digital": report["digital_crop_self_consistency"]["summary"],
        "native": report["native_multimagnification_agreement"]["summary"],
    }
    if carinthia_records:
        console_summary["carinthia"] = report[
            "carinthia_semiconductor_sem_self_consistency"
        ]["summary"]
    print(json.dumps(console_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
