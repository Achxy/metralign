#!/usr/bin/env python3
"""Audit Metralign ambiguity review policies without changing frozen predictions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image

from drift_sense.localizer import LocalizationConfig, localize
from drift_sense.safety import (
    AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD,
    POLICY_ID,
    TRANSFORM_STABILITY_REVIEW_THRESHOLD,
    assess_absolute_site,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DEVELOPMENT_DIAGNOSTICS = (
    ROOT / "evidence" / "safety" / "development-diagnostics.json"
)
DEFAULT_EVALUATION_REPORTS = ROOT / "results" / "frozen" / "reports"
DEFAULT_CASE_ROOT = ROOT / "results" / "frozen" / "cases"
COMPACT_COLUMNS = (
    "source_report",
    "id",
    "error",
    "confidence",
    "ambiguity_flag",
    "score_tied",
    "residual_evidence",
    "transform_stability",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _report_rows(report_dir: Path) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    hashes: dict[str, str] = {}
    paths = sorted(report_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"no JSON reports under {report_dir}")
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        try:
            samples = report["methods"]["full"]["samples"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid full-method report: {path}") from exc
        rows.extend(samples)
        hashes[path.name] = _sha256(path)
    return rows, hashes


def _compact_rows(path: Path) -> tuple[list[dict], dict[str, str]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported compact diagnostic schema: {path}")
    if document.get("study") != "metralign-ambiguity-development-diagnostics":
        raise ValueError(f"unexpected compact diagnostic study: {path}")
    if tuple(document.get("columns", ())) != COMPACT_COLUMNS:
        raise ValueError(f"compact diagnostic columns changed: {path}")
    source_hashes = document.get("source_report_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"compact diagnostics omit source report hashes: {path}")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for name, digest in source_hashes.items()
    ):
        raise ValueError(f"compact diagnostic source binding is invalid: {path}")
    encoded_rows = document.get("rows")
    if not isinstance(encoded_rows, list) or not encoded_rows:
        raise ValueError(f"compact diagnostics contain no rows: {path}")
    if document.get("row_count") != len(encoded_rows):
        raise ValueError(f"compact diagnostic row count mismatch: {path}")

    rows: list[dict] = []
    identities: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for encoded in encoded_rows:
        if not isinstance(encoded, list) or len(encoded) != len(COMPACT_COLUMNS):
            raise ValueError(f"malformed compact diagnostic row: {path}")
        values = dict(zip(COMPACT_COLUMNS, encoded, strict=True))
        source_report = values["source_report"]
        sample_id = values["id"]
        if source_report not in source_hashes or not isinstance(sample_id, str):
            raise ValueError(f"unbound compact diagnostic row: {path}")
        identity = (source_report, sample_id)
        if identity in identities:
            raise ValueError(f"duplicate compact diagnostic row: {identity}")
        identities.add(identity)
        counts[source_report] = counts.get(source_report, 0) + 1
        numeric = (
            values["error"],
            values["confidence"],
            values["residual_evidence"],
            values["transform_stability"],
        )
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in numeric):
            raise ValueError(f"non-finite compact diagnostic value: {identity}")
        if float(values["error"]) < 0:
            raise ValueError(f"negative compact diagnostic error: {identity}")
        if not isinstance(values["ambiguity_flag"], bool) or not isinstance(
            values["score_tied"], bool
        ):
            raise ValueError(f"invalid compact diagnostic flag: {identity}")
        rows.append(
            {
                "id": sample_id,
                "error": float(values["error"]),
                "diagnostics": {
                    "confidence": float(values["confidence"]),
                    "ambiguity_flag": values["ambiguity_flag"],
                    "ambiguity_evidence": {
                        "score_tied": values["score_tied"],
                        "residual_evidence": float(values["residual_evidence"]),
                        "transform_stability": float(values["transform_stability"]),
                    },
                },
            }
        )
    if counts != document.get("source_record_counts"):
        raise ValueError(f"compact diagnostic per-source counts mismatch: {path}")
    return rows, source_hashes


def _mechanical_development_threshold(rows: list[dict]) -> tuple[float, float]:
    ambiguous_residuals = [
        float(row["diagnostics"]["ambiguity_evidence"]["residual_evidence"])
        for row in rows
        if row["diagnostics"]["ambiguity_flag"]
    ]
    if not ambiguous_residuals:
        raise ValueError("development reports contain no ambiguous cases")
    maximum = max(ambiguous_residuals)
    threshold = round(math.ceil(maximum / 0.05 - 1e-12) * 0.05, 10)
    return maximum, threshold


def _mechanical_transform_threshold(rows: list[dict]) -> tuple[float, float]:
    minimum = min(
        float(row["diagnostics"]["ambiguity_evidence"]["transform_stability"])
        for row in rows
    )
    threshold = math.floor(minimum / 0.05 + 1e-12) * 0.05
    return minimum, round(threshold, 10)


def _policy_metrics(rows: list[dict], conservative: bool) -> dict[str, object]:
    decisions: list[tuple[dict, bool]] = []
    for row in rows:
        diagnostics = row["diagnostics"]
        evidence = diagnostics["ambiguity_evidence"]
        assessment = assess_absolute_site(
            match_confidence=float(diagnostics["confidence"]),
            ambiguous=bool(evidence.get("score_tied", diagnostics["ambiguity_flag"])),
            residual_evidence=float(evidence["residual_evidence"]),
            transform_stability=float(evidence["transform_stability"]),
        )
        review = (
            assessment.conservative_abstention_recommended
            if conservative
            else assessment.review_recommended
        )
        decisions.append((row, review))
    reviewed = [row for row, review in decisions if review]
    accepted = [row for row, review in decisions if not review]
    large = lambda row: float(row["error"]) > 5.0
    true_positive = sum(review and large(row) for row, review in decisions)
    false_positive = sum(review and not large(row) for row, review in decisions)
    false_negative = sum(not review and large(row) for row, review in decisions)
    true_negative = sum(not review and not large(row) for row, review in decisions)
    accepted_errors = np.asarray([row["error"] for row in accepted], dtype=np.float64)
    return {
        "count": len(rows),
        "reviewed_count": len(reviewed),
        "accepted_count": len(accepted),
        "accepted_coverage": len(accepted) / len(rows),
        "large_error_definition": "error > 5 px",
        "large_error_count": sum(large(row) for row in rows),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "large_error_recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "review_precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "accepted_success_le_1px": float(np.mean(accepted_errors <= 1.0)),
        "accepted_success_le_5px": float(np.mean(accepted_errors <= 5.0)),
        "accepted_max_error_px": float(np.max(accepted_errors)),
    }


def _prior_recovery(case_root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in sorted(case_root.glob("failure_*")):
        metadata_paths = list(case.glob("*.json"))
        reference_paths = list(case.glob("*_reference.png"))
        search_paths = list(case.glob("*_search.png"))
        if not (len(metadata_paths) == len(reference_paths) == len(search_paths) == 1):
            continue
        metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        reference = np.asarray(Image.open(reference_paths[0]), dtype=np.float32)
        search = np.asarray(Image.open(search_paths[0]), dtype=np.float32)
        ground_truth = np.array([metadata["center_x"], metadata["center_y"]], dtype=np.float64)
        # A fixed nonzero offset demonstrates a stage-like approximate prior;
        # ground truth is used here only to construct and score this audit case.
        prior_offset = np.array([3.0, -4.0], dtype=np.float64)
        prior = ground_truth + prior_offset
        prediction = localize(
            reference,
            search,
            LocalizationConfig(
                method="full",
                prior_center_x=float(prior[0]),
                prior_center_y=float(prior[1]),
            ),
        )
        predicted = np.array([prediction.x, prediction.y], dtype=np.float64)
        results.append(
            {
                "case": case.name,
                "ground_truth": ground_truth.tolist(),
                "external_prior": prior.tolist(),
                "external_prior_offset": prior_offset.tolist(),
                "prediction": predicted.tolist(),
                "error_px": float(np.linalg.norm(predicted - ground_truth)),
                "ambiguity_flag": prediction.ambiguity_flag,
                "review_recommended": prediction.decision_support["review_recommended"],
                "hypothesis_count": prediction.hypothesis_count,
            }
        )
    if not results:
        raise ValueError(f"no complete failure cases under {case_root}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    development = parser.add_mutually_exclusive_group()
    development.add_argument("--development-reports", type=Path)
    development.add_argument("--development-diagnostics", type=Path)
    parser.add_argument(
        "--evaluation-reports", type=Path, default=DEFAULT_EVALUATION_REPORTS
    )
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.development_reports is not None:
            development_rows, development_hashes = _report_rows(
                args.development_reports
            )
            development_input = {
                "kind": "archived_report_directory",
                "source_report_sha256": development_hashes,
            }
        else:
            compact_path = (
                args.development_diagnostics or DEFAULT_DEVELOPMENT_DIAGNOSTICS
            )
            development_rows, development_hashes = _compact_rows(compact_path)
            try:
                locator = compact_path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                locator = compact_path.name
            development_input = {
                "kind": "checked_in_compact_diagnostics",
                "path": locator,
                "sha256": _sha256(compact_path),
                "source_report_sha256": development_hashes,
            }
        evaluation_rows, evaluation_hashes = _report_rows(args.evaluation_reports)
        development_maximum, derived_threshold = _mechanical_development_threshold(
            development_rows
        )
        development_transform_minimum, derived_transform_threshold = (
            _mechanical_transform_threshold(development_rows)
        )
        if not math.isclose(
            derived_threshold, AMBIGUOUS_RESIDUAL_REVIEW_THRESHOLD, abs_tol=1e-12
        ):
            raise ValueError(
                "implemented safety threshold does not match development derivation"
            )
        if not math.isclose(
            derived_transform_threshold,
            TRANSFORM_STABILITY_REVIEW_THRESHOLD,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "implemented transform threshold does not match development derivation"
            )
        report = {
            "schema_version": 1,
            "study": "post-release-ambiguity-safety-audit",
            "interpretation": (
                "This retrospective audit does not alter the frozen coordinates or "
                "their reported metrics. The selective rule is derived mechanically "
                "from archived development diagnostics, but the choice to study this "
                "signal followed inspection of the released failures."
            ),
            "policy": {
                "id": POLICY_ID,
                "development_ambiguous_residual_max": development_maximum,
                "rounding_rule": "ceil(maximum / 0.05) * 0.05",
                "ambiguous_residual_review_threshold": derived_threshold,
                "development_transform_stability_min": development_transform_minimum,
                "transform_rounding_rule": "floor(minimum / 0.05) * 0.05",
                "transform_stability_review_threshold": derived_transform_threshold,
                "conservative_rule": "review every score_tied=true prediction",
                "selective_rule": (
                    "review an ambiguous prediction when residual evidence >= 0.15 "
                    "or transform stability < 0.95"
                ),
            },
            "input_binding": {
                "development": development_input,
                "frozen_evaluation_source_report_sha256": evaluation_hashes,
            },
            "development": {
                "source_report_sha256": development_hashes,
                "conservative": _policy_metrics(development_rows, conservative=True),
                "selective": _policy_metrics(development_rows, conservative=False),
            },
            "frozen_evaluation": {
                "source_report_sha256": evaluation_hashes,
                "conservative": _policy_metrics(evaluation_rows, conservative=True),
                "selective": _policy_metrics(evaluation_rows, conservative=False),
            },
            "external_prior_demonstration": {
                "scope": (
                    "Oracle-anchored sensitivity check with a fixed (+3, -4) px prior "
                    "offset; not an evaluation of a real stage-position sensor."
                ),
                "cases": _prior_recovery(args.case_root),
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
        }
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
