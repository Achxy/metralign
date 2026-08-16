#!/usr/bin/env python3
"""Run the locked, optional official-XFeat external registration benchmark."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from drift_sense.xfeat_baseline import (
    OFFICIAL_CODE_BUNDLE_SHA256,
    OFFICIAL_COMMIT,
    OFFICIAL_WEIGHTS_SHA256,
    XFeatBaselineConfig,
    load_official_xfeat,
    run_xfeat_official_star,
)


METHOD = "xfeat_official_star_homography"
THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0)
REQUIRED_FIELDS = {
    "id",
    "architecture",
    "suite",
    "reference",
    "search",
    "center_x",
    "center_y",
}
ROOT = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_hashes_sha256(named_hashes: dict[str, str], manifest_sha256: str) -> str:
    digest = sha256()
    digest.update(b"drift-sense-dataset-v1\0")
    digest.update(manifest_sha256.encode("ascii"))
    digest.update(b"\0")
    for name, value in sorted(named_hashes.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool]:
    commit_run = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    status_run = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = commit_run.stdout.strip() if commit_run.returncode == 0 else None
    return commit, status_run.returncode != 0 or bool(status_run.stdout.strip())


def _portable_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"{resolved.parent.name}/{resolved.name}"


def _load_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object: {path}")
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                raise ValueError(
                    f"manifest line {line_number} is missing {sorted(missing)}: {path}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest has duplicate record IDs: {path}")
    return rows


def _input_path(root: Path, value: object, sample_id: str, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError(f"sample {sample_id} has absolute {field} path")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"sample {sample_id} has out-of-tree {field} path")
    if not resolved.is_file():
        raise ValueError(f"sample {sample_id} is missing {field}: {relative}")
    return resolved


def _artifact_binding(
    manifest: Path,
    records: list[dict[str, object]],
) -> dict[str, object]:
    manifest_digest = _sha256_file(manifest)
    image_hashes: dict[str, str] = {}
    for row in records:
        sample_id = str(row["id"])
        for field in ("reference", "search"):
            relative = Path(str(row[field])).as_posix()
            digest = _sha256_file(
                _input_path(manifest.parent, row[field], sample_id, field)
            )
            previous = image_hashes.setdefault(relative, digest)
            if previous != digest:
                raise ValueError(f"input changed while hashing: {relative}")
    commit, dirty = _git_state()
    adapter_files = (
        ROOT / "compare_xfeat.py",
        ROOT / "src/drift_sense/xfeat_baseline.py",
        ROOT / "evidence/external/xfeat-predeclared-protocol.json",
    )
    adapter_digest = sha256()
    adapter_digest.update(b"metralign-xfeat-adapter-v1\0")
    for path in adapter_files:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        adapter_digest.update(len(relative).to_bytes(8, "big"))
        adapter_digest.update(relative)
        adapter_digest.update(len(content).to_bytes(8, "big"))
        adapter_digest.update(content)
    return {
        "manifest_sha256": manifest_digest,
        "input_images_sha256": dict(sorted(image_hashes.items())),
        "dataset_sha256": _named_hashes_sha256(image_hashes, manifest_digest),
        "adapter_sha256": adapter_digest.hexdigest(),
        "git_commit": commit,
        "working_tree_dirty": dirty,
    }


def _selection_key(manifest_sha256: str, record: dict[str, object]) -> str:
    payload = (
        manifest_sha256.encode("ascii")
        + b"\0"
        + str(record["id"]).encode("utf-8")
    )
    return sha256(payload).hexdigest()


def select_records(
    records: list[dict[str, object]],
    manifest_sha256: str,
    selection: str,
) -> list[dict[str, object]]:
    """Apply only the population rules fixed in the predeclared protocol."""

    if selection in {"all1400", "independent100"}:
        return list(records)
    per_architecture: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        per_architecture[str(row["architecture"])].append(row)
    expected = {"dram", "finfet"}
    if set(per_architecture) != expected:
        raise ValueError(
            f"locked selection expects architectures {sorted(expected)}, got "
            f"{sorted(per_architecture)}"
        )
    count = 1 if selection == "timing-probe" else 20
    selected: list[dict[str, object]] = []
    for architecture in sorted(per_architecture):
        ranked = sorted(
            per_architecture[architecture],
            key=lambda row: (_selection_key(manifest_sha256, row), str(row["id"])),
        )
        if len(ranked) < count:
            raise ValueError(
                f"locked selection needs {count} {architecture} records, got {len(ranked)}"
            )
        selected.extend(ranked[:count])
    selected_ids = {str(row["id"]) for row in selected}
    return [row for row in records if str(row["id"]) in selected_ids]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    resolved = [row for row in rows if row["error"] is not None]
    errors = np.asarray([row["error"] for row in resolved], dtype=np.float64)
    runtimes = np.asarray([row["runtime_ms"] for row in rows], dtype=np.float64)
    result: dict[str, object] = {
        "count": total,
        "resolved_count": len(resolved),
        "unresolved_count": total - len(resolved),
        "coverage": len(resolved) / total,
        "mean_runtime_ms": float(np.mean(runtimes)),
        "median_runtime_ms": float(np.median(runtimes)),
        "p95_runtime_ms": float(np.percentile(runtimes, 95, method="linear")),
    }
    for threshold in THRESHOLDS:
        successes = int(np.count_nonzero(errors <= threshold)) if errors.size else 0
        result[f"success_le_{threshold:g}px"] = successes / total
        result[f"resolved_success_le_{threshold:g}px"] = (
            successes / len(resolved) if resolved else 0.0
        )
    for name, function in (
        ("mean_error_px_resolved", np.mean),
        ("median_error_px_resolved", np.median),
        (
            "p95_error_px_resolved",
            lambda values: np.percentile(values, 95, method="linear"),
        ),
        ("max_error_px_resolved", np.max),
    ):
        result[name] = float(function(errors)) if errors.size else None
    return result


def _evaluate(
    model: object,
    manifest: Path,
    records: list[dict[str, object]],
    config: XFeatBaselineConfig,
    *,
    quiet: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        reference_path = _input_path(
            manifest.parent, record["reference"], str(record["id"]), "reference"
        )
        search_path = _input_path(
            manifest.parent, record["search"], str(record["id"]), "search"
        )
        with Image.open(reference_path) as image:
            reference = np.asarray(image.convert("L"), dtype=np.uint8)
        with Image.open(search_path) as image:
            search = np.asarray(image.convert("L"), dtype=np.uint8)
        result = run_xfeat_official_star(model, reference, search, config).to_dict()
        x, y = result["x"], result["y"]
        error = (
            None
            if x is None or y is None
            else float(
                math.hypot(
                    float(x) - float(record["center_x"]),
                    float(y) - float(record["center_y"]),
                )
            )
        )
        rows.append(
            {
                "id": str(record["id"]),
                "suite": str(record["suite"]),
                "architecture": str(record["architecture"]),
                "ground_truth": [
                    float(record["center_x"]),
                    float(record["center_y"]),
                ],
                "prediction": None if x is None else [float(x), float(y)],
                "error": error,
                "status": result["status"],
                "runtime_ms": float(result["runtime_ms"]),
                "diagnostics": result["diagnostics"],
            }
        )
        if not quiet:
            print(f"{manifest.parent.name}: {index + 1}/{len(records)}", file=sys.stderr)
    return rows


def _read_protocol(path: Path) -> tuple[dict[str, object], str]:
    content = path.read_bytes()
    protocol = json.loads(content)
    if protocol.get("protocol_status") != "locked_before_outcome_inspection":
        raise ValueError("XFeat protocol is not marked locked")
    upstream = protocol.get("upstream", {})
    if upstream.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("protocol official commit differs from adapter lock")
    if upstream.get("inference_code_bundle_sha256") != OFFICIAL_CODE_BUNDLE_SHA256:
        raise ValueError("protocol code bundle differs from adapter lock")
    if upstream.get("weights_sha256") != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError("protocol checkpoint differs from adapter lock")
    return protocol, sha256(content).hexdigest()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "evidence/external/xfeat-predeclared-protocol.json",
    )
    parser.add_argument(
        "--selection",
        choices=("timing-probe", "all1400", "hash280", "independent100"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware-label", default="unspecified CPU")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output.exists() and not args.force:
        parser.error(f"output already exists (use --force): {args.output}")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol, protocol_digest = _read_protocol(args.protocol)
        weights = args.weights or args.official_source / "weights/xfeat.pt"

        import torch
        import tqdm

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        model, identity = load_official_xfeat(
            args.official_source,
            weights,
            top_k=8000,
        )
        config = XFeatBaselineConfig()
        datasets: dict[str, object] = {}
        all_rows: list[dict[str, object]] = []
        seen_names: set[str] = set()
        for data_dir in args.data_dir:
            manifest = (
                data_dir / "manifest.jsonl" if data_dir.is_dir() else data_dir
            ).resolve()
            records = _load_manifest(manifest)
            binding = _artifact_binding(manifest, records)
            manifest_digest = str(binding["manifest_sha256"])
            name = manifest.parent.name
            if name in seen_names:
                raise ValueError(f"duplicate dataset name: {name}")
            seen_names.add(name)
            selected = select_records(records, manifest_digest, args.selection)
            rows = _evaluate(
                model,
                manifest,
                selected,
                config,
                quiet=args.quiet,
            )
            all_rows.extend(rows)
            if args.selection == "timing-probe":
                result: dict[str, object] = {
                    "runtime_only": True,
                    "runtime_ms": [float(row["runtime_ms"]) for row in rows],
                }
            else:
                result = {"metrics": summarize(rows), "samples": rows}
            datasets[name] = {
                "manifest": _portable_locator(manifest),
                "artifact_binding": binding,
                "dataset_record_count": len(records),
                "evaluated_record_count": len(selected),
                "selected_id_sha256": sha256(
                    "\n".join(str(row["id"]) for row in selected).encode("utf-8")
                ).hexdigest(),
                "architecture_counts": dict(
                    sorted(Counter(str(row["architecture"]) for row in selected).items())
                ),
                "result": result,
            }

        if args.selection == "timing-probe":
            runtimes = np.asarray(
                [float(row["runtime_ms"]) for row in all_rows], dtype=np.float64
            )
            pooled: dict[str, object] = {
                "runtime_only": True,
                "count": len(all_rows),
                "median_runtime_ms": float(np.median(runtimes)),
                "projected_all1400_runtime_seconds": float(np.median(runtimes) * 1.4),
                "full_population_runtime_limit_seconds": 4500.0,
                "full_population_permitted": bool(np.median(runtimes) * 1.4 <= 4500.0),
            }
        else:
            pooled = {"metrics": summarize(all_rows)}

        identity_record = identity.to_dict()
        identity_record["source_root"] = args.official_source.resolve().name
        identity_record["weights_file"] = (
            f"{args.official_source.resolve().name}/weights/{weights.name}"
        )
        report = {
            "schema_version": 1,
            "study": "xfeat-official-fixed-external-development-benchmark",
            "claim_boundary": protocol["claim_boundary"],
            "selection": args.selection,
            "protocol": {
                "locator": _portable_locator(args.protocol),
                "sha256": protocol_digest,
                "locked_snapshot": protocol,
            },
            "method": METHOD,
            "configuration": asdict(config),
            "upstream_identity": identity_record,
            "method_reference": "https://github.com/verlab/accelerated_features",
            "paper_reference": "https://doi.org/10.1109/CVPR52733.2024.00259",
            "external_software": {
                "xfeat": {
                    "commit": OFFICIAL_COMMIT,
                    "license": "Apache License 2.0",
                    "license_source": "https://github.com/verlab/accelerated_features/blob/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/LICENSE",
                    "checkpoint_license_note": (
                        "checkpoint distributed in the same repository; no separate "
                        "checkpoint license was found in the pinned tree"
                    ),
                },
                "torch": {"version": torch.__version__},
                "opencv": {
                    "version": cv2.__version__,
                    "license": "Apache License 2.0",
                },
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "hardware_label": args.hardware_label,
                "device": str(model.dev),
                "torch": torch.__version__,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "opencv_threads": cv2.getNumThreads(),
                "pillow": pillow_version,
                "tqdm": tqdm.__version__,
                "installed_distributions": dict(
                    sorted(
                        (
                            str(distribution.metadata.get("Name") or "unknown"),
                            distribution.version,
                        )
                        for distribution in importlib.metadata.distributions()
                    )
                ),
            },
            "metric_definition": {
                "coordinate_error": "Euclidean distance in search-image pixels",
                "success_denominator": "all evaluated records, including unresolved",
                "runtime_scope": protocol["determinism"]["runtime_scope"],
                "model_load_reported_separately": True,
            },
            "datasets": datasets,
            "pooled": pooled,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        if args.quiet:
            print(
                json.dumps(
                    {
                        "output": _portable_locator(args.output),
                        "selection": args.selection,
                        "pooled": pooled,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(rendered, end="")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
