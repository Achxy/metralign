import json
from pathlib import Path

import pytest

from make_independent_renderer_plate import select_cases, verify_plate_binding


def _sample(identifier: str, architecture: str, error: float) -> dict:
    return {"id": identifier, "architecture": architecture, "error": error}


def test_success_plate_selection_is_mechanical_and_excludes_failures():
    samples = [
        _sample("d0", "dram", 0.01),
        _sample("d1", "dram", 0.03),
        _sample("d2", "dram", 0.08),
        _sample("d3", "dram", 0.20),
        _sample("d4", "dram", 17.0),
        _sample("f0", "finfet", 0.02),
        _sample("f1", "finfet", 0.04),
        _sample("f2", "finfet", 0.09),
        _sample("f3", "finfet", 0.30),
        _sample("f4", "finfet", 9.0),
    ]

    selected = select_cases(samples, "success")

    assert [item["sample"]["id"] for item in selected] == ["d1", "f1", "d3", "f3"]
    assert [item["selection"] for item in selected] == [
        "median successful",
        "median successful",
        "P95 successful",
        "P95 successful",
    ]
    assert all(item["sample"]["error"] <= 1.0 for item in selected)


def test_success_plate_selection_uses_id_to_break_equal_distance():
    samples = [
        _sample("d_a", "dram", 0.1),
        _sample("d_b", "dram", 0.3),
        _sample("f_a", "finfet", 0.1),
        _sample("f_b", "finfet", 0.3),
    ]
    selected = select_cases(samples, "success")
    assert selected[0]["sample"]["id"] == "d_a"
    assert selected[1]["sample"]["id"] == "f_a"


def test_plate_sidecar_binds_report_and_plate_bytes(tmp_path: Path):
    report = tmp_path / "report.json"
    plate = tmp_path / "plate.png"
    sidecar = tmp_path / "plate.json"
    report.write_bytes(b"report-v1")
    plate.write_bytes(b"plate-v1")
    from hashlib import sha256

    sidecar.write_text(
        json.dumps(
            {
                "report_sha256": sha256(report.read_bytes()).hexdigest(),
                "plate_sha256": sha256(plate.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    assert verify_plate_binding(sidecar, report, plate)["report_sha256"]
    report.write_bytes(b"report-v2")
    with pytest.raises(ValueError, match="report hash mismatch"):
        verify_plate_binding(sidecar, report, plate)
    report.write_bytes(b"report-v1")
    plate.write_bytes(b"plate-v2")
    with pytest.raises(ValueError, match="plate hash mismatch"):
        verify_plate_binding(sidecar, report, plate)
