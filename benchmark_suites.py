#!/usr/bin/env python3
"""Generate and evaluate deterministic development or frozen-report suites."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from drift_sense.dataset import SUITE_SEED_OFFSETS
from evaluate import METHODS as EVALUATION_METHODS
from evaluate import pipeline_configuration, validate_records, verify_artifact_binding


SUITES = (
    "iid",
    "high_noise",
    "geometry_ood",
    "transform_ood",
    "periodic_ambiguity",
    "scan_distortion",
    "cross_generator",
)
METHODS = ("baseline0", "multiscale", "structure_gradient", "structure_residual", "full", "all")

# Widely separated bases keep every suite independent even after the generator's
# deterministic per-suite offset and per-sample stride are applied.
SPLIT_SEEDS = {
    "dev": {suite: 11_000_003 + index * 100_000_000 for index, suite in enumerate(SUITES)},
    "report": {
        suite: 1_011_000_003 + index * 100_000_000 for index, suite in enumerate(SUITES)
    },
}

ACCURACY_KEYS = (
    "success_le_0.5px",
    "success_le_1px",
    "success_le_2px",
    "success_le_3px",
    "success_le_5px",
    "mean_error_px",
    "median_error_px",
    "p90_error_px",
    "p95_error_px",
    "p99_error_px",
    "max_error_px",
    "failure_gt_5px_count",
    "count",
)
RUNTIME_KEYS = (
    "mean_runtime_ms",
    "median_runtime_ms",
    "p95_runtime_ms",
    "mean_image_io_ms",
    "median_image_io_ms",
    "p95_image_io_ms",
    "mean_localizer_runtime_ms",
    "median_localizer_runtime_ms",
    "p95_localizer_runtime_ms",
)
SAMPLE_SEED_STRIDE = 104_729


def _json_text(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, indent=indent, sort_keys=True, allow_nan=False)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _split_names(selection: str) -> tuple[str, ...]:
    return ("dev", "report") if selection == "both" else (selection,)


def build_plan(args: argparse.Namespace) -> dict:
    root = args.output_dir.resolve()
    suites = tuple(dict.fromkeys(args.suites or SUITES))
    jobs = []
    for split in _split_names(args.split):
        pair_count = args.dev_pairs if split == "dev" else args.report_pairs
        for suite in suites:
            data_dir = root / "datasets" / split / suite
            report_path = root / "reports" / split / f"{suite}.json"
            jobs.append(
                {
                    "split": split,
                    "suite": suite,
                    "seed_base": SPLIT_SEEDS[split][suite],
                    "pair_count": pair_count,
                    "data_dir": str(data_dir),
                    "report_path": str(report_path),
                    "expected_configuration": {
                        "architecture": args.architecture,
                        "difficulty": args.difficulty,
                        "image_size": args.image_size,
                        "supersample": args.supersample,
                        "resampler": args.resampler,
                    },
                }
            )
    bases = [job["seed_base"] for job in jobs]
    if len(bases) != len(set(bases)):
        raise ValueError("benchmark suite seed bases are not independent")
    return {
        "schema_version": 1,
        "protocol": "development" if args.split == "dev" else "frozen-report",
        "report_access_confirmed": bool(args.confirm_report),
        "configuration": {
            "split_selection": args.split,
            "suites": list(suites),
            "architecture": args.architecture,
            "dev_pairs_per_suite": args.dev_pairs,
            "report_pairs_per_suite": args.report_pairs,
            "difficulty": args.difficulty,
            "image_size": args.image_size,
            "supersample": args.supersample,
            "resampler": args.resampler,
            "method": args.method,
            "top_k": args.top_k,
            "scale_range": args.scale_range,
            "rotation_range": args.rotation_range,
            **pipeline_configuration(args),
        },
        "jobs": jobs,
    }


def _read_manifest(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty suite manifest: {path}")
    return records


def _expected_architectures(selection: str, count: int) -> list[str]:
    choices = ("dram", "finfet") if selection == "both" else (selection,)
    return [choices[index % len(choices)] for index in range(count)]


def _expected_rendering(
    suite: str,
    supersample: int,
    requested_resampler: str,
) -> dict[str, str]:
    if suite == "cross_generator":
        return {
            "reference_renderer": "alternate_polyphase_kaiser",
            "search_renderer": "alternate_polyphase_hann",
            "reference_resampler": "none" if supersample == 1 else "polyphase_kaiser",
            "search_resampler": "none" if supersample == 1 else "polyphase_hann",
        }
    search_resampler = "lanczos" if requested_resampler == "area" else "area"
    return {
        "reference_renderer": f"primary_{requested_resampler}",
        "search_renderer": f"primary_{search_resampler}",
        "reference_resampler": "none" if supersample == 1 else requested_resampler,
        "search_resampler": "none" if supersample == 1 else search_resampler,
    }


def _verify_manifest(job: dict, manifest_path: Path) -> list[dict]:
    records = _read_manifest(manifest_path)
    if len(records) != job["pair_count"]:
        raise ValueError(
            f"{manifest_path} has {len(records)} records, expected {job['pair_count']}"
        )
    if {record.get("suite") for record in records} != {job["suite"]}:
        raise ValueError(f"suite mismatch in {manifest_path}")
    seeds = [record.get("seed") for record in records]
    if any(not isinstance(seed, int) for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError(f"missing or duplicate sample seeds in {manifest_path}")
    expected_seeds = [
        job["seed_base"] + SUITE_SEED_OFFSETS[job["suite"]] + index * SAMPLE_SEED_STRIDE
        for index in range(job["pair_count"])
    ]
    if seeds != expected_seeds:
        raise ValueError(f"sample seed sequence does not match the benchmark plan: {manifest_path}")
    expected = job.get("expected_configuration")
    if not isinstance(expected, dict):
        raise ValueError("benchmark job is missing expected_configuration")
    expected_architectures = _expected_architectures(expected["architecture"], job["pair_count"])
    actual_architectures = [record.get("architecture") for record in records]
    if actual_architectures != expected_architectures:
        raise ValueError(f"architecture sequence does not match the benchmark plan: {manifest_path}")
    rendering = _expected_rendering(
        job["suite"],
        int(expected["supersample"]),
        str(expected["resampler"]),
    )
    geometry_variant = (
        "ood"
        if job["suite"] == "geometry_ood"
        else "ideal"
        if job["suite"] == "periodic_ambiguity"
        else "default"
    )
    root = manifest_path.parent
    validate_records(root, records, expected_image_size=int(expected["image_size"]))
    seen_files: set[str] = set()
    for index, (record, architecture) in enumerate(zip(records, expected_architectures)):
        sample_id = f"{index:06d}_{architecture}"
        if record.get("id") != sample_id:
            raise ValueError(f"unexpected sample id at record {index}: {record.get('id')!r}")
        if record.get("difficulty") != expected["difficulty"]:
            raise ValueError(f"difficulty mismatch for sample {sample_id}")
        if record.get("image_size") != expected["image_size"]:
            raise ValueError(f"image_size mismatch for sample {sample_id}")
        if record.get("nominal_scale") != 0.1:
            raise ValueError(f"nominal_scale mismatch for sample {sample_id}")
        expected_files = {
            "reference": f"{sample_id}_reference.png",
            "search": f"{sample_id}_search.png",
        }
        for field, name in expected_files.items():
            if record.get(field) != name:
                raise ValueError(f"unexpected {field} filename for sample {sample_id}")
            if name in seen_files:
                raise ValueError(f"duplicate benchmark image filename: {name}")
            seen_files.add(name)
        distortion = record.get("distortion_parameters")
        if not isinstance(distortion, dict):
            raise ValueError(f"missing distortion_parameters for sample {sample_id}")
        expected_distortion = {
            **rendering,
            "geometry_variant": geometry_variant,
            "supersample": expected["supersample"],
        }
        for field, value in expected_distortion.items():
            if distortion.get(field) != value:
                raise ValueError(f"{field} mismatch for sample {sample_id}")
        for geometry_field in ("reference_geometry", "search_geometry"):
            geometry = record.get(geometry_field)
            if not isinstance(geometry, dict) or (
                geometry.get("width"), geometry.get("height")
            ) != (expected["image_size"], expected["image_size"]):
                raise ValueError(f"{geometry_field} mismatch for sample {sample_id}")
        sidecar = root / f"{sample_id}.json"
        if not sidecar.is_file():
            raise ValueError(f"missing sidecar for sample {sample_id}: {sidecar}")
        try:
            sidecar_record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid sidecar for sample {sample_id}: {sidecar}") from exc
        if sidecar_record != record:
            raise ValueError(f"sidecar does not match manifest for sample {sample_id}")
    return records


def _run(command: list[str], repo_root: Path) -> None:
    print("running: " + " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=repo_root, check=True, stdout=subprocess.DEVNULL)


def _pipeline_cli_args(args: argparse.Namespace) -> list[str]:
    configuration = pipeline_configuration(args)
    return [
        "--phase-calibration"
        if configuration["enable_phase_calibration"]
        else "--no-phase-calibration",
        "--evidence-channel",
        configuration["periodic_evidence_channel"],
        "--spatial-residual"
        if configuration["enable_spatial_residual"]
        else "--no-spatial-residual",
        "--lattice-grouping"
        if configuration["enable_lattice_grouping"]
        else "--no-lattice-grouping",
        "--ambiguity-rule"
        if configuration["enable_ambiguity_rule"]
        else "--no-ambiguity-rule",
        "--subpixel-refinement",
        configuration["subpixel_refinement"],
    ]


def _generate(job: dict, args: argparse.Namespace, repo_root: Path) -> tuple[Path, list[dict]]:
    data_dir = Path(job["data_dir"])
    manifest_path = data_dir / "manifest.jsonl"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"suite already exists; use --resume after verification: {data_dir}")
        return manifest_path, _verify_manifest(job, manifest_path)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"non-empty suite directory has no manifest: {data_dir}")
    command = [
        sys.executable,
        str(repo_root / "generate_dataset.py"),
        "--architecture",
        args.architecture,
        "--num-pairs",
        str(job["pair_count"]),
        "--output-dir",
        str(data_dir),
        "--seed",
        str(job["seed_base"]),
        "--difficulty",
        args.difficulty,
        "--suite",
        job["suite"],
        "--image-size",
        str(args.image_size),
        "--supersample",
        str(args.supersample),
        "--resampler",
        args.resampler,
    ]
    _run(command, repo_root)
    return manifest_path, _verify_manifest(job, manifest_path)


def _evaluate(
    job: dict,
    args: argparse.Namespace,
    repo_root: Path,
    manifest_path: Path,
    records: list[dict],
) -> dict:
    report_path = Path(job["report_path"])
    if report_path.exists():
        if not args.resume:
            raise FileExistsError(f"report already exists; use --resume after verification: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        command = [
            sys.executable,
            str(repo_root / "evaluate.py"),
            "--data-dir",
            str(manifest_path.parent),
            "--method",
            args.method,
            "--output",
            str(report_path),
            "--top-k",
            str(args.top_k),
            "--scale-range",
            str(args.scale_range),
            "--rotation-range",
            str(args.rotation_range),
            *_pipeline_cli_args(args),
            "--quiet",
        ]
        _run(command, repo_root)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        verify_artifact_binding(
            report,
            manifest_path,
            records,
            repo_root,
            require_current_code=True,
        )
    except ValueError as exc:
        raise ValueError(f"evaluation report cannot be reused: {report_path}: {exc}") from exc
    report_configuration = report.get("configuration", {})
    expected_configuration = {
        "requested_method": args.method,
        "top_k": args.top_k,
        "scale_range": args.scale_range,
        "rotation_range": args.rotation_range,
        "limit": None,
        **pipeline_configuration(args),
    }
    if report_configuration != expected_configuration:
        raise ValueError(f"evaluation configuration mismatch in {report_path}")
    if report.get("schema_version") != 2:
        raise ValueError(f"unsupported evaluation report schema in {report_path}")
    if report.get("dataset_record_count") != len(records) or report.get(
        "evaluated_record_count"
    ) != len(records):
        raise ValueError(f"evaluation report record count mismatch in {report_path}")
    expected_methods = set(EVALUATION_METHODS if args.method == "all" else (args.method,))
    reported_methods = report.get("methods")
    if not isinstance(reported_methods, dict) or set(reported_methods) != expected_methods:
        raise ValueError(f"evaluation report method set mismatch in {report_path}")
    for method, method_result in reported_methods.items():
        try:
            samples = method_result["samples"]
            count = method_result["metrics"]["all"]["count"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed {method} result in {report_path}") from exc
        if not isinstance(samples, list) or len(samples) != len(records) or count != len(records):
            raise ValueError(f"{method} result count mismatch in {report_path}")
    return report


def _actual_result(job: dict, manifest_path: Path, records: list[dict], report: dict) -> dict:
    return {
        "split": job["split"],
        "suite": job["suite"],
        "seed_base": job["seed_base"],
        "sample_seeds": [record["seed"] for record in records],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_binding": report["artifact_binding"],
        "report": str(Path(job["report_path"]).resolve()),
        "methods": {
            method: {
                "metrics": method_result["metrics"]["all"],
                "failure_counts": method_result["failure_counts"],
            }
            for method, method_result in report["methods"].items()
        },
        "environment": report.get("environment", {}),
        "metric_definition": report.get("metric_definition", {}),
    }


def _git_state(repo_root: Path) -> tuple[str | None, bool]:
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    status_result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
    )
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return commit, dirty


def ledger_entry(
    args: argparse.Namespace,
    plan: dict,
    actual_results: list[dict],
    repo_root: Path,
    aggregate_report: Path,
) -> dict:
    commit, dirty = _git_state(repo_root)
    accuracy: dict[str, dict] = {}
    runtime: dict[str, dict] = {}
    failure_counts: dict[str, dict] = {}
    seed_set: dict[str, dict] = {}
    for result in actual_results:
        key = f"{result['split']}:{result['suite']}"
        seed_set[key] = {
            "base_seed": result["seed_base"],
            "sample_seeds": result["sample_seeds"],
            "manifest_sha256": result["manifest_sha256"],
            "dataset_sha256": result["artifact_binding"]["dataset_sha256"],
            "implementation_sha256": result["artifact_binding"]["implementation_sha256"],
            "git_commit": result["artifact_binding"]["git_commit"],
            "input_image_count": len(result["artifact_binding"]["input_images_sha256"]),
        }
        accuracy[key] = {}
        runtime[key] = {}
        failure_counts[key] = {}
        for method, method_result in result["methods"].items():
            summary = method_result["metrics"]
            accuracy[key][method] = {name: summary[name] for name in ACCURACY_KEYS if name in summary}
            runtime[key][method] = {name: summary[name] for name in RUNTIME_KEYS if name in summary}
            failure_counts[key][method] = method_result["failure_counts"]
    return {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": args.hypothesis,
        "commit": commit,
        "working_tree_dirty": dirty,
        "configuration": plan["configuration"],
        "seed_set": seed_set,
        "accuracy": accuracy,
        "runtime": runtime,
        "failure_counts": failure_counts,
        "decision": args.decision,
        "artifacts": {"aggregate_report": str(aggregate_report.resolve())},
    }


def append_ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json_text(entry, indent=None) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "report", "both"), default="dev")
    parser.add_argument("--suite", dest="suites", action="append", choices=SUITES)
    parser.add_argument("--dev-pairs", type=int, default=100)
    parser.add_argument("--report-pairs", type=int, default=200)
    parser.add_argument("--architecture", choices=("dram", "finfet", "both"), default="both")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    parser.add_argument("--method", choices=METHODS, default="full")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--scale-range", type=float, default=0.006)
    parser.add_argument("--rotation-range", type=float, default=3.0)
    parser.add_argument(
        "--phase-calibration", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evidence-channel",
        choices=("structural", "gradient", "raw"),
        default="structural",
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
        choices=("parabolic", "dft", "none"),
        default="parabolic",
    )
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--resampler", choices=("area", "lanczos"), default="area")
    parser.add_argument("--confirm-report", action="store_true", help="confirm the frozen split may be read")
    parser.add_argument("--resume", action="store_true", help="reuse only verified manifests and reports")
    parser.add_argument("--dry-run", action="store_true", help="print the deterministic plan without writing")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--hypothesis")
    parser.add_argument("--decision")
    args = parser.parse_args(argv)
    if min(args.dev_pairs, args.report_pairs, args.top_k, args.supersample) < 1:
        parser.error("pair counts, top-k, and supersample must be positive")
    if args.image_size < 64:
        parser.error("--image-size must be at least 64")
    if args.scale_range < 0 or args.rotation_range < 0:
        parser.error("search ranges must be non-negative")
    if args.split in {"report", "both"} and not (args.confirm_report or args.dry_run):
        parser.error("report evaluation requires --confirm-report to protect the frozen split")
    ledger_fields = (args.experiment_id, args.hypothesis, args.decision)
    if args.ledger and not all(ledger_fields):
        parser.error("--ledger requires --experiment-id, --hypothesis, and --decision")
    if not args.ledger and any(ledger_fields):
        parser.error("experiment metadata requires --ledger")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(_json_text(plan))
        return 0
    repo_root = Path(__file__).resolve().parent
    actual_results = []
    all_sample_seeds: set[int] = set()
    try:
        for job in plan["jobs"]:
            manifest_path, records = _generate(job, args, repo_root)
            overlap = all_sample_seeds.intersection(record["seed"] for record in records)
            if overlap:
                raise ValueError(f"sample seed collision across suites: {min(overlap)}")
            all_sample_seeds.update(record["seed"] for record in records)
            report = _evaluate(job, args, repo_root, manifest_path, records)
            actual_results.append(_actual_result(job, manifest_path, records, report))
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: benchmark aborted: {exc}", file=sys.stderr)
        return 2
    aggregate_report = args.output_dir.resolve() / "reports" / f"benchmark_{args.split}.json"
    aggregate_report.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {"schema_version": 1, "plan": plan, "results": actual_results}
    aggregate_report.write_text(_json_text(aggregate) + "\n", encoding="utf-8")
    if args.ledger:
        entry = ledger_entry(args, plan, actual_results, repo_root, aggregate_report)
        append_ledger(args.ledger, entry)
    print(_json_text({"aggregate_report": str(aggregate_report), "jobs_completed": len(actual_results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
