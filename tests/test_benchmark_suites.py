from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import pytest

import benchmark_suites
import evaluate
from drift_sense.dataset import GeneratorConfig, generate_pair, write_manifest


def test_default_plan_covers_all_suites_with_independent_dev_report_seeds(tmp_path: Path):
    args = benchmark_suites.parse_args(
        ["--output-dir", str(tmp_path), "--split", "both", "--dry-run"]
    )

    plan = benchmark_suites.build_plan(args)

    assert len(plan["jobs"]) == 2 * len(benchmark_suites.SUITES)
    assert {(job["split"], job["suite"]) for job in plan["jobs"]} == {
        (split, suite)
        for split in ("dev", "report")
        for suite in benchmark_suites.SUITES
    }
    seeds = [job["seed_base"] for job in plan["jobs"]]
    assert len(seeds) == len(set(seeds))
    assert {job["pair_count"] for job in plan["jobs"] if job["split"] == "dev"} == {100}
    assert {job["pair_count"] for job in plan["jobs"] if job["split"] == "report"} == {200}


def test_report_split_requires_explicit_confirmation(tmp_path: Path):
    with pytest.raises(SystemExit):
        benchmark_suites.parse_args(["--output-dir", str(tmp_path), "--split", "report"])


def test_plan_and_evaluator_command_bind_pipeline_ablation_controls(tmp_path: Path):
    args = benchmark_suites.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--dry-run",
            "--no-phase-calibration",
            "--evidence-channel",
            "gradient",
            "--no-spatial-residual",
            "--no-lattice-grouping",
            "--no-ambiguity-rule",
            "--subpixel-refinement",
            "none",
        ]
    )

    assert {
        key: benchmark_suites.build_plan(args)["configuration"][key]
        for key in evaluate.pipeline_configuration(args)
    } == evaluate.pipeline_configuration(args)
    assert benchmark_suites._pipeline_cli_args(args) == [
        "--no-phase-calibration",
        "--evidence-channel",
        "gradient",
        "--no-spatial-residual",
        "--no-lattice-grouping",
        "--no-ambiguity-rule",
        "--subpixel-refinement",
        "none",
    ]


def test_manifest_verification_checks_count_suite_and_unique_seeds(tmp_path: Path):
    seed_base = 1
    cfg = GeneratorConfig(image_size=64, supersample=1, suite="iid", difficulty="medium")
    records = [
        generate_pair(tmp_path, index, architecture, seed_base, cfg)
        for index, architecture in enumerate(("dram", "finfet"))
    ]
    path = write_manifest(tmp_path, records)
    job = {
        "pair_count": 2,
        "suite": "iid",
        "seed_base": seed_base,
        "expected_configuration": {
            "architecture": "both",
            "difficulty": "medium",
            "image_size": 64,
            "supersample": 1,
            "resampler": "area",
        },
    }

    assert benchmark_suites._verify_manifest(job, path) == records
    records[1]["architecture"] = "dram"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="architecture sequence"):
        benchmark_suites._verify_manifest(job, path)
    records[1]["architecture"] = "finfet"
    records[1]["seed"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        benchmark_suites._verify_manifest(job, path)


def test_manifest_verification_rejects_configuration_and_sidecar_drift(tmp_path: Path):
    cfg = GeneratorConfig(image_size=64, supersample=2, suite="geometry_ood", difficulty="medium")
    record = generate_pair(tmp_path, 0, "dram", 3, cfg)
    path = write_manifest(tmp_path, [record])
    job = {
        "pair_count": 1,
        "suite": "geometry_ood",
        "seed_base": 3,
        "expected_configuration": {
            "architecture": "dram",
            "difficulty": "medium",
            "image_size": 64,
            "supersample": 2,
            "resampler": "area",
        },
    }

    assert benchmark_suites._verify_manifest(job, path) == [record]
    wrong_difficulty = {**job, "expected_configuration": {**job["expected_configuration"], "difficulty": "hard"}}
    with pytest.raises(ValueError, match="difficulty mismatch"):
        benchmark_suites._verify_manifest(wrong_difficulty, path)
    tampered_record = deepcopy(record)
    tampered_record["distortion_parameters"]["search_resampler"] = "polyphase_hann"
    path.write_text(json.dumps(tampered_record) + "\n", encoding="utf-8")
    sidecar = tmp_path / "000000_dram.json"
    sidecar.write_text(json.dumps(tampered_record), encoding="utf-8")
    with pytest.raises(ValueError, match="search_resampler mismatch"):
        benchmark_suites._verify_manifest(job, path)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["center_x"] += 1.0
    sidecar.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar does not match"):
        benchmark_suites._verify_manifest(job, path)


def test_resume_rejects_report_after_input_or_code_binding_drift(tmp_path: Path):
    cfg = GeneratorConfig(image_size=64, supersample=1, suite="iid", difficulty="medium")
    record = generate_pair(tmp_path, 0, "dram", 5, cfg)
    manifest = write_manifest(tmp_path, [record])
    repo_root = Path(benchmark_suites.__file__).resolve().parent
    binding = evaluate.build_artifact_binding(manifest, [record], repo_root)
    report_path = tmp_path / "report.json"
    report = {
        "schema_version": 2,
        "manifest_sha256": binding["manifest_sha256"],
        "artifact_binding": deepcopy(binding),
        "dataset_record_count": 1,
        "evaluated_record_count": 1,
        "configuration": {
            "requested_method": "full",
            "top_k": 32,
            "scale_range": 0.006,
            "rotation_range": 3.0,
            "limit": None,
            "enable_phase_calibration": True,
            "periodic_evidence_channel": "structural",
            "enable_spatial_residual": True,
            "enable_lattice_grouping": True,
            "enable_ambiguity_rule": True,
            "subpixel_refinement": "parabolic",
        },
        "methods": {
            "full": {
                "samples": [{"id": record["id"]}],
                "metrics": {"all": {"count": 1}},
            }
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    job = {"report_path": str(report_path)}
    args = Namespace(
        resume=True,
        method="full",
        top_k=32,
        scale_range=0.006,
        rotation_range=3.0,
        phase_calibration=True,
        evidence_channel="structural",
        spatial_residual=True,
        lattice_grouping=True,
        ambiguity_rule=True,
        subpixel_refinement="parabolic",
    )

    assert benchmark_suites._evaluate(job, args, repo_root, manifest, [record]) == report
    args.subpixel_refinement = "none"
    with pytest.raises(ValueError, match="evaluation configuration mismatch"):
        benchmark_suites._evaluate(job, args, repo_root, manifest, [record])
    args.subpixel_refinement = "parabolic"
    report["artifact_binding"]["implementation_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation_sha256"):
        benchmark_suites._evaluate(job, args, repo_root, manifest, [record])
    report["artifact_binding"] = deepcopy(binding)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / record["search"]).write_bytes((tmp_path / record["reference"]).read_bytes())
    with pytest.raises(ValueError, match="input_images_sha256|dataset_sha256"):
        benchmark_suites._evaluate(job, args, repo_root, manifest, [record])


def test_ledger_records_measured_values_and_user_decision(tmp_path: Path):
    args = benchmark_suites.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "--experiment-id",
            "exp-001",
            "--hypothesis",
            "Gradient matching improves held-out accuracy.",
            "--decision",
            "Retain only if report criteria pass.",
        ]
    )
    plan = benchmark_suites.build_plan(args)
    actual = [
        {
            "split": "dev",
            "suite": "iid",
            "seed_base": 11,
            "sample_seeds": [11, 12],
            "manifest_sha256": "abc",
            "artifact_binding": {
                "dataset_sha256": "dataset",
                "implementation_sha256": "implementation",
                "git_commit": "commit",
                "input_images_sha256": {"a.png": "hash-a", "b.png": "hash-b"},
            },
            "methods": {
                "full": {
                    "metrics": {
                        "success_le_1px": 0.5,
                        "mean_error_px": 2.0,
                        "mean_runtime_ms": 3.0,
                        "count": 2,
                    },
                    "failure_counts": {"periodic ambiguity": 1},
                }
            },
        }
    ]

    entry = benchmark_suites.ledger_entry(
        args, plan, actual, tmp_path, tmp_path / "aggregate.json"
    )
    benchmark_suites.append_ledger(args.ledger, entry)
    persisted = json.loads(args.ledger.read_text(encoding="utf-8"))

    assert persisted["hypothesis"] == args.hypothesis
    assert persisted["decision"] == args.decision
    assert persisted["seed_set"]["dev:iid"]["sample_seeds"] == [11, 12]
    assert persisted["seed_set"]["dev:iid"]["dataset_sha256"] == "dataset"
    assert persisted["seed_set"]["dev:iid"]["input_image_count"] == 2
    assert persisted["accuracy"]["dev:iid"]["full"]["success_le_1px"] == 0.5
    assert persisted["runtime"]["dev:iid"]["full"]["mean_runtime_ms"] == 3.0
    assert persisted["failure_counts"]["dev:iid"]["full"] == {"periodic ambiguity": 1}
