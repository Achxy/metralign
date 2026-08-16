#!/usr/bin/env python3
"""Run a paired leave-one-component-out generator robustness study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from drift_sense.dataset import AUGMENTATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=71_000_019)
    parser.add_argument("--architecture", choices=["dram", "finfet", "both"], default="both")
    parser.add_argument("--suite", choices=["iid", "high_noise", "geometry_ood", "transform_ood", "periodic_ambiguity", "scan_distortion", "cross_generator"], default="iid")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--components", nargs="+", choices=AUGMENTATIONS, default=list(AUGMENTATIONS))
    parser.add_argument("--method", choices=["baseline0", "multiscale", "structure_gradient", "structure_residual", "full"], default="full")
    return parser.parse_args()


def _run(command: list[str], root: Path) -> None:
    print("running: " + " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.num_pairs < 1 or args.image_size < 64 or args.supersample < 1:
        raise SystemExit("num-pairs >=1, image-size >=64, and supersample >=1 are required")
    root = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs: dict[str, dict] = {}
    variants: list[tuple[str, str | None]] = [("all_enabled", None)] + [
        (f"without_{name}", name) for name in dict.fromkeys(args.components)
    ]
    for label, disabled in variants:
        data_dir = output / "datasets" / label
        report_path = output / "reports" / f"{label}.json"
        generate = [
            sys.executable,
            str(root / "generate_dataset.py"),
            "--architecture", args.architecture,
            "--num-pairs", str(args.num_pairs),
            "--output-dir", str(data_dir),
            "--seed", str(args.seed),
            "--difficulty", args.difficulty,
            "--suite", args.suite,
            "--image-size", str(args.image_size),
            "--supersample", str(args.supersample),
        ]
        if disabled:
            generate.extend(["--disable-augmentation", disabled])
        _run(generate, root)
        _run(
            [
                sys.executable,
                str(root / "evaluate.py"),
                "--data-dir", str(data_dir),
                "--method", args.method,
                "--output", str(report_path),
                "--quiet",
            ],
            root,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report["methods"][args.method]["metrics"]["all"]
        runs[label] = {
            "disabled_component": disabled,
            "manifest": str((data_dir / "manifest.jsonl").resolve()),
            "manifest_sha256": _hash(data_dir / "manifest.jsonl"),
            "report": str(report_path.resolve()),
            "metrics": metrics,
            "failure_counts": report["methods"][args.method]["failure_counts"],
        }

    reference = runs["all_enabled"]["metrics"]
    for label, run in runs.items():
        metrics = run["metrics"]
        run["delta_vs_all_enabled"] = {
            "success_le_0.5px": metrics["success_le_0.5px"] - reference["success_le_0.5px"],
            "median_error_px": metrics["median_error_px"] - reference["median_error_px"],
            "p95_error_px": metrics["p95_error_px"] - reference["p95_error_px"],
        }
    aggregate = {
        "schema_version": 1,
        "study": "paired leave-one-generator-component-out robustness sensitivity",
        "interpretation": (
            "This training-free method does not learn from augmentations. Deltas measure "
            "localizer sensitivity to removing one simulated mechanism, not a training gain "
            "or physical parameter calibration."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "num_pairs": args.num_pairs,
            "seed": args.seed,
            "architecture": args.architecture,
            "suite": args.suite,
            "difficulty": args.difficulty,
            "image_size": args.image_size,
            "supersample": args.supersample,
            "method": args.method,
        },
        "runs": runs,
    }
    path = output / "augmentation_ablation.json"
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
