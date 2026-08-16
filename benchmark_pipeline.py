#!/usr/bin/env python3
"""Measure final-pipeline stages, candidate K, and subpixel alternatives on one dev set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    return parser.parse_args()


def _evaluate(root: Path, data_dir: Path, report_path: Path, arguments: list[str]) -> dict:
    command = [
        sys.executable,
        str(root / "evaluate.py"),
        "--data-dir",
        str(data_dir),
        "--output",
        str(report_path),
        "--quiet",
        *arguments,
    ]
    print("running: " + " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _summary(report: dict, method: str) -> dict:
    result = report["methods"][method]
    return {
        "configuration": report["configuration"],
        "metrics": result["metrics"]["all"],
        "failure_counts": result["failure_counts"],
        "artifact_binding": report["artifact_binding"],
        "report": report["manifest"],
    }


def main() -> int:
    args = parse_args()
    if not args.top_k_values or min(args.top_k_values) < 1:
        raise SystemExit("top-k values must be positive")
    root = Path(__file__).resolve().parent
    data_dir = args.data_dir.resolve()
    manifest = data_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"manifest does not exist: {manifest}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # One shared pipeline. Rows add one production stage at a time; gradient is
    # retained as an explicitly measured representation alternative because it
    # did not become part of the selected structural-residual path.
    stages: list[tuple[str, list[str]]] = [
        ("zncc", ["--method", "baseline0"]),
        (
            "phase_calibration",
            ["--method", "full", "--no-spatial-residual", "--no-lattice-grouping", "--no-ambiguity-rule", "--subpixel-refinement", "none"],
        ),
        (
            "raw_spatial_residual",
            ["--method", "full", "--evidence-channel", "raw", "--no-lattice-grouping", "--no-ambiguity-rule", "--subpixel-refinement", "none"],
        ),
        (
            "structural_spatial_residual",
            ["--method", "full", "--evidence-channel", "structural", "--no-lattice-grouping", "--no-ambiguity-rule", "--subpixel-refinement", "none"],
        ),
        (
            "lattice_family_candidates",
            ["--method", "full", "--evidence-channel", "structural", "--lattice-grouping", "--no-ambiguity-rule", "--subpixel-refinement", "none"],
        ),
        (
            "ambiguity_rule",
            ["--method", "full", "--evidence-channel", "structural", "--lattice-grouping", "--ambiguity-rule", "--subpixel-refinement", "none"],
        ),
        (
            "parabolic_subpixel",
            ["--method", "full", "--evidence-channel", "structural", "--lattice-grouping", "--ambiguity-rule", "--subpixel-refinement", "parabolic"],
        ),
    ]
    stage_results = {}
    for label, arguments in stages:
        report_path = output / f"stage_{label}.json"
        report = _evaluate(root, data_dir, report_path, arguments)
        stage_results[label] = _summary(report, arguments[1]) | {"report": str(report_path)}

    representation_results = {}
    for channel in ("raw", "gradient", "structural"):
        report_path = output / f"representation_{channel}.json"
        report = _evaluate(
            root,
            data_dir,
            report_path,
            ["--method", "full", "--evidence-channel", channel],
        )
        representation_results[channel] = _summary(report, "full") | {"report": str(report_path)}

    refinement_results = {}
    for refinement in ("none", "dft", "parabolic"):
        report_path = output / f"refinement_{refinement}.json"
        report = _evaluate(
            root,
            data_dir,
            report_path,
            ["--method", "full", "--subpixel-refinement", refinement],
        )
        refinement_results[refinement] = _summary(report, "full") | {"report": str(report_path)}

    top_k_results = {}
    for top_k in dict.fromkeys(args.top_k_values):
        report_path = output / f"top_k_{top_k}.json"
        report = _evaluate(
            root,
            data_dir,
            report_path,
            ["--method", "full", "--top-k", str(top_k)],
        )
        top_k_results[str(top_k)] = _summary(report, "full") | {"report": str(report_path)}

    aggregate = {
        "schema_version": 1,
        "study": "development-only final-pipeline selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "interpretation": (
            "Stage rows share the full implementation and add production controls. "
            "Representation, K, and refinement blocks are alternatives rather than "
            "cumulative claims. This development artifact is not a frozen result."
        ),
        "pipeline_stages": stage_results,
        "representation_alternatives": representation_results,
        "refinement_alternatives": refinement_results,
        "top_k_sweep": top_k_results,
    }
    aggregate_path = output / "development_pipeline_study.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(aggregate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
