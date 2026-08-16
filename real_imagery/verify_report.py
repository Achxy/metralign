#!/usr/bin/env python3
"""Independently verify real-imagery report bindings and aggregate arithmetic."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image

from real_imagery.evaluate_real_imagery import (
    CARINTHIA_SOURCE,
    SOURCES,
    carinthia_selection,
    implementation_sha256,
    protocol_bundle_sha256,
    source_rows,
    summarize_carinthia,
    summarize_digital,
    summarize_native,
)
from real_imagery.evaluate_registered_tem import (
    HERE as TEM_HERE,
    ROOT,
    SOURCE as TEM_SOURCE,
    discover_test_pairs,
    summarize as summarize_tem,
    tree_digest,
)
from real_imagery.protocol import digest_file
from real_imagery.protocol import DIGITAL_POSITIONS


KNOWN_METHODS = {"full", "baseline0"}
NATIVE_MAGNIFICATION_PAIRS = ((100, 50), (200, 100), (200, 50))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_configuration(row: dict) -> None:
    require(float(row["repeat_delta_px"]) >= 0.0, "negative repeat delta")
    configuration = row["configuration"]
    for key in (
        "method",
        "nominal_scale",
        "scale_range",
        "rotation_range",
        "enable_phase_calibration",
        "periodic_evidence_channel",
        "enable_spatial_residual",
        "enable_lattice_grouping",
        "enable_ambiguity_rule",
        "subpixel_refinement",
    ):
        require(key in configuration, f"record configuration omits {key}")
    if "method" in row:
        require(
            configuration["method"] == row["method"],
            "record method/configuration mismatch",
        )


def verify_record_errors(records: list[dict]) -> None:
    for row in records:
        completed = "error_px" in row
        failed = "error" in row
        require(
            completed != failed,
            f"record must contain exactly one result or error: {row.get('case_id')}",
        )
        if not completed:
            continue
        dx = float(row["prediction"][0]) - float(row["ground_truth"][0])
        dy = float(row["prediction"][1]) - float(row["ground_truth"][1])
        recomputed = math.hypot(dx, dy)
        require(
            math.isclose(recomputed, float(row["error_px"]), rel_tol=0.0, abs_tol=1e-12),
            f"error arithmetic mismatch: {row.get('case_id')}",
        )
        _verify_configuration(row)


def _verify_exact_coverage(
    records: list[dict], case_field: str, expected_cases: set[str], methods: set[str], label: str
) -> None:
    require(methods, f"{label} declares no methods")
    require(methods.issubset(KNOWN_METHODS), f"{label} declares an unknown method")
    keys = [(row.get(case_field), row.get("method")) for row in records]
    require(len(keys) == len(set(keys)), f"{label} has duplicate case/method records")
    expected = {(case, method) for case in expected_cases for method in methods}
    require(set(keys) == expected, f"{label} case/method coverage mismatch")
    require(len(records) == len(expected), f"{label} record count mismatch")


def _verify_fallback_counts(records: list[dict], summary: dict, methods: set[str]) -> None:
    for method in methods:
        require("fallback_count" in summary[method], f"{method} summary omits fallback_count")
        completed = [
            row
            for row in records
            if row["method"] == method and "prediction" in row
        ]
        actual = sum(
            "fallback" in row["diagnostics"]["pipeline_stages"] for row in completed
        )
        require(
            summary[method]["fallback_count"] == actual,
            f"{method} fallback count mismatch",
        )


def _verify_native_records(records: list[dict]) -> None:
    for row in records:
        completed = "prediction" in row
        failed = "error" in row
        require(completed != failed, f"native record must contain exactly one result or error: {row.get('pair_id')}")
        if not completed:
            continue
        _verify_configuration(row)
        proxy_accepted = bool(row["proxy_accepted"])
        has_disagreement = "proxy_disagreement_px" in row
        require(
            has_disagreement == proxy_accepted,
            f"native proxy-disagreement disclosure mismatch: {row.get('pair_id')}",
        )
        if proxy_accepted:
            target = row["feature_registration_proxy"]["reference_center_in_search_px"]
            recomputed = math.hypot(
                float(row["prediction"][0]) - float(target[0]),
                float(row["prediction"][1]) - float(target[1]),
            )
            require(
                math.isclose(
                    recomputed,
                    float(row["proxy_disagreement_px"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"native proxy-disagreement arithmetic mismatch: {row.get('pair_id')}",
            )


def verify_sem(
    report: dict,
    data_dir: Path,
    carinthia_archive: Path | None = None,
) -> dict:
    manifest = json.loads(SOURCES.read_text())
    require(report["source_manifest_sha256"] == digest_file(SOURCES), "SEM source manifest mismatch")
    require(
        report["artifact_binding"]["core_implementation_sha256"] == implementation_sha256(),
        "SEM core implementation fingerprint mismatch",
    )
    require(
        report["artifact_binding"]["protocol_bundle_sha256"] == protocol_bundle_sha256(),
        "SEM protocol bundle fingerprint mismatch",
    )
    rows = source_rows(manifest, data_dir)
    expected_sources = [
        {
            key: dataset[key]
            for key in (
                "id",
                "title",
                "doi",
                "landing_url",
                "license",
                "license_url",
                "authors",
                "category",
            )
        }
        for dataset in manifest["datasets"]
    ]
    require(report["sources"] == expected_sources, "SEM source metadata mismatch")
    expected_inputs = {
        (row["dataset"]["id"], row["area"], magnification): {
            "name": record["name"],
            "bytes": record["size"],
            "md5": record["md5"],
            "sha256": record["sha256"],
        }
        for row in rows
        for magnification, record in row["files"].items()
    }
    require(
        len(report["input_files"])
        == len({
            (item["dataset"], item["area"], item["magnification_k"])
            for item in report["input_files"]
        }),
        "SEM input list contains duplicate identities",
    )
    reported_inputs = {
        (item["dataset"], item["area"], item["magnification_k"]): {
            key: item[key] for key in ("name", "bytes", "md5", "sha256")
        }
        for item in report["input_files"]
    }
    require(reported_inputs == expected_inputs, "SEM input set or hash mismatch")
    digital = report["digital_crop_self_consistency"]["records"]
    native = report["native_multimagnification_agreement"]["records"]
    digital_summary = report["digital_crop_self_consistency"]["summary"]
    native_summary = report["native_multimagnification_agreement"]["summary"]
    digital_methods = set(digital_summary)
    native_methods = set(native_summary["methods"])
    require(digital_methods == native_methods, "SEM sections declare different method sets")
    expected_digital = {
        f"{row['dataset']['id']}/{row['area']}/digital-{position_index}"
        for row in rows
        for position_index in range(len(DIGITAL_POSITIONS))
    }
    expected_native = {
        f"{row['dataset']['id']}/{row['area']}/{reference_mag}k-to-{search_mag}k"
        for row in rows
        for reference_mag, search_mag in NATIVE_MAGNIFICATION_PAIRS
    }
    _verify_exact_coverage(
        digital, "case_id", expected_digital, digital_methods, "SEM digital"
    )
    _verify_exact_coverage(
        native, "pair_id", expected_native, native_methods, "SEM native"
    )
    source_by_area = {
        (row["dataset"]["id"], row["area"]): row for row in rows
    }
    for row in digital:
        source = source_by_area[(row["dataset"], row["area"])]
        require(
            row["category"] == source["dataset"]["category"],
            f"SEM digital category mismatch: {row['case_id']}",
        )
        require(row["source_magnification_k"] == 50, "SEM digital source magnification mismatch")
        require(
            row["source_sha256"] == source["files"][50]["sha256"],
            f"SEM digital source hash mismatch: {row['case_id']}",
        )
    for row in native:
        source = source_by_area[(row["dataset"], row["area"])]
        reference_mag = int(row["reference_magnification_k"])
        search_mag = int(row["search_magnification_k"])
        require(
            (reference_mag, search_mag) in NATIVE_MAGNIFICATION_PAIRS,
            f"SEM native magnification pair mismatch: {row['pair_id']}",
        )
        require(
            row["category"] == source["dataset"]["category"],
            f"SEM native category mismatch: {row['pair_id']}",
        )
        require(
            row["reference_sha256"] == source["files"][reference_mag]["sha256"],
            f"SEM native reference hash mismatch: {row['pair_id']}",
        )
        require(
            row["search_sha256"] == source["files"][search_mag]["sha256"],
            f"SEM native search hash mismatch: {row['pair_id']}",
        )
    verify_record_errors(digital)
    _verify_native_records(native)
    methods = sorted(digital_methods)
    _verify_fallback_counts(digital, digital_summary, digital_methods)
    _verify_fallback_counts(native, native_summary["methods"], native_methods)
    require(
        report["digital_crop_self_consistency"]["summary"]
        == summarize_digital(digital, methods),
        "SEM digital aggregate mismatch",
    )
    require(
        report["native_multimagnification_agreement"]["summary"]
        == summarize_native(native, methods),
        "SEM native aggregate mismatch",
    )
    carinthia_record_count = 0
    if "carinthia_semiconductor_sem_self_consistency" in report:
        require(carinthia_archive is not None, "Carinthia report requires --carinthia-archive")
        section = report["carinthia_semiconductor_sem_self_consistency"]
        carinthia_manifest = json.loads(CARINTHIA_SOURCE.read_text())
        require(
            section["source_manifest_sha256"] == digest_file(CARINTHIA_SOURCE),
            "Carinthia source manifest mismatch",
        )
        require(section["source"] == carinthia_manifest["dataset"], "Carinthia source mismatch")
        archive_binding, selected = carinthia_selection(
            carinthia_archive, carinthia_manifest
        )
        require(section["archive"] == archive_binding, "Carinthia archive binding mismatch")
        expected_inputs = [
            {key: value for key, value in source.items() if key != "pixels"}
            for source in selected
        ]
        require(section["input_members"] == expected_inputs, "Carinthia selected input mismatch")
        carinthia_records = section["records"]
        carinthia_summary = section["summary"]
        carinthia_methods = set(carinthia_summary)
        require(carinthia_methods == digital_methods, "Carinthia method set mismatch")
        expected_cases = {
            f"carinthia/label-{source['class_label']}/"
            f"{PurePosixPath(source['member']).stem}"
            for source in selected
        }
        _verify_exact_coverage(
            carinthia_records,
            "case_id",
            expected_cases,
            carinthia_methods,
            "Carinthia digital",
        )
        selected_by_member = {source["member"]: source for source in selected}
        for row in carinthia_records:
            source = selected_by_member[row["member"]]
            for field in (
                "class_label",
                "selection_rank_within_class",
                "member_bytes",
                "member_crc32",
                "member_sha256",
                "shape_yx",
                "dtype",
            ):
                require(row[field] == source[field], f"Carinthia row binding mismatch: {field}")
        verify_record_errors(carinthia_records)
        _verify_fallback_counts(
            carinthia_records, carinthia_summary, carinthia_methods
        )
        require(
            carinthia_summary == summarize_carinthia(carinthia_records, methods),
            "Carinthia aggregate mismatch",
        )
        carinthia_record_count = len(carinthia_records)
    else:
        require(carinthia_archive is None, "Carinthia archive supplied but report omits it")
    return {
        "input_file_count": len(reported_inputs),
        "record_count": len(digital) + len(native) + carinthia_record_count,
        "carinthia_record_count": carinthia_record_count,
    }


def verify_tem(report: dict, archive_path: Path) -> dict:
    require(report["source_manifest_sha256"] == digest_file(TEM_SOURCE), "TEM source manifest mismatch")
    manifest = json.loads(TEM_SOURCE.read_text())
    require(report["source"] == manifest["dataset"], "TEM source metadata mismatch")
    source_files = sorted((ROOT / "src" / "drift_sense").glob("*.py"))
    protocol_files = [TEM_HERE / "evaluate_registered_tem.py", TEM_HERE / "protocol.py", TEM_SOURCE]
    require(
        report["artifact_binding"]["core_implementation_sha256"]
        == tree_digest(source_files, b"metralign-registered-tem-core-v1"),
        "TEM core implementation fingerprint mismatch",
    )
    require(
        report["artifact_binding"]["protocol_bundle_sha256"]
        == tree_digest(protocol_files, b"metralign-registered-tem-protocol-v1"),
        "TEM protocol bundle fingerprint mismatch",
    )
    require(archive_path.stat().st_size == report["archive"]["bytes"], "TEM archive size mismatch")
    require(digest_file(archive_path, "md5") == report["archive"]["md5"], "TEM archive MD5 mismatch")
    require(digest_file(archive_path) == report["archive"]["sha256"], "TEM archive SHA-256 mismatch")
    expected_archive = manifest["archive"]
    require(report["archive"]["name"] == expected_archive["name"], "TEM archive name mismatch")
    require(report["archive"]["bytes"] == expected_archive["bytes"], "TEM pinned archive size mismatch")
    require(report["archive"]["md5"] == expected_archive["md5"], "TEM pinned archive MD5 mismatch")

    require(
        len(report["input_members"])
        == len({item["member"] for item in report["input_members"]}),
        "TEM input member list contains duplicates",
    )
    reported_inputs = {item["member"]: item for item in report["input_members"]}
    with ZipFile(archive_path) as archive:
        pairs = discover_test_pairs(archive)
        expected_members = {
            pair[key]
            for pair in pairs
            for key in ("gt_member", "low_member")
        }
        require(set(reported_inputs) == expected_members, "TEM selected member set mismatch")
        for member in sorted(expected_members, key=str.casefold):
            content = archive.read(member)
            info = archive.getinfo(member)
            require(info.file_size == reported_inputs[member]["bytes"], f"TEM member byte count mismatch: {member}")
            require(f"{info.CRC:08x}" == reported_inputs[member]["crc32"], f"TEM member CRC mismatch: {member}")
            require(sha256(content).hexdigest() == reported_inputs[member]["sha256"], f"TEM member hash mismatch: {member}")
            with Image.open(BytesIO(content)) as image:
                array = np.asarray(image)
                shape = list(array.shape)
            require(shape == reported_inputs[member]["shape_yx"], f"TEM member shape mismatch: {member}")
            require(str(array.dtype) == reported_inputs[member]["dtype"], f"TEM member dtype mismatch: {member}")
            require(float(np.min(array)) == reported_inputs[member]["intensity_min"], f"TEM member minimum mismatch: {member}")
            require(float(np.max(array)) == reported_inputs[member]["intensity_max"], f"TEM member maximum mismatch: {member}")
    require(len(pairs) == report["selection"]["eligible_pair_count"], "TEM pair count mismatch")
    records = report["records"]
    methods = set(report["summary"])
    expected_cases = {
        f"{pair['gt_member']}::{pair['low_member']}" for pair in pairs
    }
    _verify_exact_coverage(records, "case_id", expected_cases, methods, "TEM")
    pair_by_case = {
        f"{pair['gt_member']}::{pair['low_member']}": pair for pair in pairs
    }
    for row in records:
        pair = pair_by_case[row["case_id"]]
        require(row["gt_member"] == pair["gt_member"], f"TEM GT identity mismatch: {row['case_id']}")
        require(row["low_member"] == pair["low_member"], f"TEM Low identity mismatch: {row['case_id']}")
        require(row["sample"] == pair["sample"], f"TEM sample label mismatch: {row['case_id']}")
        require(
            row["low_index_within_gt"] == pair["low_index_within_gt"],
            f"TEM Low index mismatch: {row['case_id']}",
        )
        require(
            row["position_index"] == pair["position_index"],
            f"TEM position index mismatch: {row['case_id']}",
        )
        require(
            row["gt_sha256"] == reported_inputs[row["gt_member"]]["sha256"],
            f"TEM GT hash mismatch: {row['case_id']}",
        )
        require(
            row["low_sha256"] == reported_inputs[row["low_member"]]["sha256"],
            f"TEM Low hash mismatch: {row['case_id']}",
        )
    verify_record_errors(records)
    _verify_fallback_counts(records, report["summary"], methods)
    ordered_methods = sorted(methods)
    require(report["summary"] == summarize_tem(records, ordered_methods), "TEM aggregate mismatch")
    return {"input_member_count": len(reported_inputs), "pair_count": len(pairs), "record_count": len(records)}


def verify_plate(report_path: Path, plate_path: Path) -> dict:
    sidecar_path = plate_path.with_suffix(".json")
    require(sidecar_path.is_file(), f"missing plate sidecar: {sidecar_path}")
    sidecar = json.loads(sidecar_path.read_text())
    report_hash = digest_file(report_path)
    require(sidecar["report_sha256"] == report_hash, "plate sidecar report hash mismatch")
    require(sidecar["plate_sha256"] == digest_file(plate_path), "plate sidecar image hash mismatch")
    with Image.open(plate_path) as image:
        embedded_report_hash = image.info.get("report_sha256")
        embedded_protocol = image.info.get("protocol")
        embedded_license = image.info.get("license")
        embedded_source_doi = image.info.get("source_doi")
    report = json.loads(report_path.read_text())
    require(embedded_report_hash == report_hash, "PNG embedded report hash mismatch")
    require(embedded_protocol == report["protocol"], "PNG embedded protocol mismatch")
    require(embedded_license == "CC BY 4.0", "PNG embedded license mismatch")
    require(embedded_source_doi == sidecar["source_doi"], "PNG embedded source DOI mismatch")
    available = {row["case_id"] for row in completed_full_records_from_report(report)}
    selected = [row["case_id"] for row in sidecar["selected_records"]]
    require(len(selected) == 4 and len(set(selected)) == 4, "plate must bind four unique cases")
    require(set(selected).issubset(available), "plate sidecar references a case absent from report")
    return {"plate_sha256": sidecar["plate_sha256"], "selected_case_count": len(selected)}


def completed_full_records_from_report(report: dict) -> list[dict]:
    if report["protocol"] == "real-sem-self-consistency-v1":
        rows = report["digital_crop_self_consistency"]["records"]
    else:
        rows = report["records"]
    return [row for row in rows if row["method"] == "full" and "error_px" in row]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--sem-data-dir", type=Path)
    inputs.add_argument("--tem-archive", type=Path)
    parser.add_argument("--carinthia-archive", type=Path)
    parser.add_argument("--plate", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if args.sem_data_dir:
        require(report["protocol"] == "real-sem-self-consistency-v1", "wrong input kind for report")
        details = verify_sem(report, args.sem_data_dir, args.carinthia_archive)
    else:
        require(args.carinthia_archive is None, "Carinthia archive is only valid for SEM reports")
        require(report["protocol"] == "registered-real-minitem-crop-localization-v1", "wrong input kind for report")
        details = verify_tem(report, args.tem_archive)
    output = {
        "verified": True,
        "report": str(args.report),
        "report_sha256": digest_file(args.report),
        **details,
    }
    if args.plate:
        output["plate"] = verify_plate(args.report, args.plate)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
