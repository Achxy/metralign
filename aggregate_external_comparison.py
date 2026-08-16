#!/usr/bin/env python3
"""Aggregate per-suite external comparison reports without discarding source rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from compare_external_baselines import summarize
from drift_sense.external_baselines import METHODS


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _portable_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"{resolved.parent.name}/{resolved.name}"


def _validate_method_result(dataset_name: str, method: str, result: dict) -> set[str]:
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{dataset_name}/{method} has no sample rows")
    sample_ids = [row.get("id") for row in samples]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise ValueError(f"{dataset_name}/{method} has an invalid sample ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{dataset_name}/{method} has duplicate sample IDs")
    recomputed = summarize(samples)
    if result.get("metrics") != recomputed:
        raise ValueError(f"{dataset_name}/{method} metric arithmetic mismatch")
    return set(sample_ids)


def _archived_samples(dataset_name: str, reference: dict) -> list[dict]:
    report_hash = reference.get("report_sha256")
    if not isinstance(report_hash, str) or len(report_hash) != 64:
        raise ValueError(f"{dataset_name} archived Metralign report hash is invalid")
    source_report = reference.get("source_report")
    if not isinstance(source_report, str) or Path(source_report).is_absolute():
        raise ValueError(f"{dataset_name} archived Metralign locator is not portable")
    samples = reference.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(
            f"{dataset_name} archived Metralign evidence is not self-contained; "
            "rerun compare_external_baselines.py"
        )
    sample_ids = [row.get("id") for row in samples]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise ValueError(f"{dataset_name} archived Metralign sample ID is invalid")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{dataset_name} archived Metralign sample IDs are not unique")
    if reference.get("evaluated_record_count") != len(samples):
        raise ValueError(f"{dataset_name} archived Metralign count mismatch")
    if reference.get("metrics") != summarize(samples):
        raise ValueError(f"{dataset_name} archived Metralign metric arithmetic mismatch")
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-method",
        action="append",
        choices=[*METHODS, "metralign"],
        help=(
            "method required in every merged dataset; repeat for a custom set "
            "(default: all six external adapters)"
        ),
    )
    args = parser.parse_args(argv)
    if args.expected_method and len(args.expected_method) != len(set(args.expected_method)):
        parser.error("--expected-method must not contain duplicates")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected_methods = set(args.expected_method or METHODS)
        sources: dict[str, str] = {}
        datasets: dict[str, dict] = {}
        method_metadata: dict[str, object] = {}
        software: dict[str, object] = {}
        for directory in args.input_dir:
            for path in sorted(directory.glob("*.json")):
                report = json.loads(path.read_text(encoding="utf-8"))
                if report.get("study") != "external-registration-baseline-comparison":
                    continue
                if report.get("schema_version") != 2:
                    raise ValueError(
                        f"unsupported comparison schema in {path}; rerun the schema-v2 comparator"
                    )
                source_locator = _portable_locator(path)
                source_hash = _sha256(path)
                if source_locator in sources and sources[source_locator] != source_hash:
                    raise ValueError(f"source locator collision: {source_locator}")
                sources[source_locator] = source_hash
                method_metadata.update(report.get("method_metadata", {}))
                external_software = report.get("external_software", {})
                if "name" in external_software:
                    software[str(external_software["name"]).casefold()] = {
                        key: value for key, value in external_software.items() if key != "name"
                    }
                else:
                    software.update(external_software)
                for name, dataset in report["datasets"].items():
                    combined = datasets.setdefault(
                        name,
                        {
                            "manifest": dataset["manifest"],
                            "artifact_binding": dataset["artifact_binding"],
                            "archived_metralign": dataset.get("archived_metralign"),
                            "dataset_record_count": dataset.get("dataset_record_count"),
                            "evaluated_record_count": dataset.get("evaluated_record_count"),
                            "methods": {},
                        },
                    )
                    if combined["artifact_binding"]["dataset_sha256"] != dataset[
                        "artifact_binding"
                    ]["dataset_sha256"]:
                        raise ValueError(f"dataset binding mismatch for {name}")
                    if combined["archived_metralign"] is None:
                        combined["archived_metralign"] = dataset.get("archived_metralign")
                    elif (
                        dataset.get("archived_metralign") is not None
                        and combined["archived_metralign"] != dataset["archived_metralign"]
                    ):
                        raise ValueError(f"archived Metralign evidence mismatch for {name}")
                    for count_key in ("dataset_record_count", "evaluated_record_count"):
                        if combined[count_key] != dataset.get(count_key):
                            raise ValueError(f"{count_key} mismatch for {name}")
                    for method, result in dataset["methods"].items():
                        if method in combined["methods"]:
                            raise ValueError(f"duplicate {method} result for {name}")
                        combined["methods"][method] = result
        if not datasets:
            raise ValueError("no comparison reports found")

        archived_presence = {
            dataset["archived_metralign"] is not None for dataset in datasets.values()
        }
        if len(archived_presence) != 1:
            raise ValueError("archived Metralign evidence is missing for only some datasets")
        for name, dataset in sorted(datasets.items()):
            actual_methods = set(dataset["methods"])
            if actual_methods != expected_methods:
                missing = sorted(expected_methods - actual_methods)
                unexpected = sorted(actual_methods - expected_methods)
                raise ValueError(
                    f"method set mismatch for {name}: missing={missing}, "
                    f"unexpected={unexpected}"
                )
            method_sample_ids = {
                method: _validate_method_result(name, method, result)
                for method, result in dataset["methods"].items()
            }
            first_ids = next(iter(method_sample_ids.values()))
            for method, sample_ids in method_sample_ids.items():
                if sample_ids != first_ids:
                    raise ValueError(f"sample coverage mismatch for {name}/{method}")
            evaluated_count = dataset.get("evaluated_record_count")
            if evaluated_count != len(first_ids):
                raise ValueError(f"evaluated record count mismatch for {name}")
            if dataset["archived_metralign"] is not None:
                archived_ids = {
                    row["id"] for row in _archived_samples(name, dataset["archived_metralign"])
                }
                if archived_ids != first_ids:
                    raise ValueError(f"archived Metralign sample coverage mismatch for {name}")

        method_names = sorted(expected_methods)
        pooled: dict[str, object] = {}
        for method in method_names:
            rows = [
                row
                for dataset in datasets.values()
                for row in dataset["methods"][method]["samples"]
            ]
            pooled[method] = {
                "metrics": summarize(rows),
                "by_suite": {
                    name: dataset["methods"][method]["metrics"]
                    for name, dataset in sorted(datasets.items())
                },
            }

        archived_rows: list[dict[str, object]] = []
        archived_hashes: dict[str, str] = {}
        for name, dataset in sorted(datasets.items()):
            reference = dataset.get("archived_metralign")
            if not reference:
                continue
            archived_hashes[name] = reference["report_sha256"]
            archived_rows.extend(
                {
                    "error": row["error"],
                    "runtime_ms": row["runtime_ms"],
                }
                for row in _archived_samples(name, reference)
            )
        if archived_rows:
            pooled["metralign_archived"] = {
                "metrics": summarize(archived_rows),
                "source_report_sha256": archived_hashes,
                "runtime_note": (
                    "Archived evaluator wall time from the frozen single-method run; "
                    "external adapters were measured later and suite jobs may run in parallel."
                ),
            }

        output = {
            "schema_version": 2,
            "study": "external-registration-baseline-aggregate",
            "label": args.label,
            "suite_count": len(datasets),
            "source_report_sha256": sources,
            "expected_methods": sorted(expected_methods),
            "method_metadata": {name: method_metadata.get(name) for name in pooled},
            "external_software": software,
            "metric_definition": {
                "success_denominator": "all records, including unresolved estimates",
                "error": "Euclidean reference-center error in search-image pixels",
                "runtime": "adapter wall time excluding image I/O",
            },
            "execution_note": (
                "Accuracy is invariant to process scheduling. Runtime measurements reflect "
                "the recorded machine load and must not be compared as isolated microbenchmarks."
            ),
            "methods": pooled,
        }
        rendered = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
