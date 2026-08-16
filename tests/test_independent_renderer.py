import ast
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import evaluate
import drift_sense.independent_renderer as independent_renderer
from drift_sense.independent_renderer import (
    HomogeneousProjection,
    IndependentRendererConfig,
    PRIMARY_PATH_MODULES,
    generate_independent_pair,
    verify_independent_suite,
    write_independent_suite,
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add("." * node.level + node.module)
    return modules


def test_source_has_no_primary_generator_imports():
    source = Path(independent_renderer.__file__)
    imported = _imported_modules(source)
    assert imported.isdisjoint(PRIMARY_PATH_MODULES)
    assert not any(module.startswith("drift_sense.") for module in imported)
    assert not any(module.startswith(".") for module in imported)


def test_homogeneous_projection_round_trip_and_jacobian():
    projection = HomogeneousProjection(
        64,
        64,
        (0.11, -0.001, -3.0, 0.002, 0.10, 7.0, 1e-5, -2e-5, 1.0),
    )
    physical = projection.sensor_to_physical(21.25, 47.75)
    recovered = projection.physical_to_sensor(float(physical[0]), float(physical[1]))
    assert recovered == pytest.approx((21.25, 47.75), abs=1e-10)
    jacobian = projection.local_jacobian(21.25, 47.75)
    epsilon = 1e-5
    dx = np.asarray(projection.sensor_to_physical(21.25 + epsilon, 47.75)) - np.asarray(physical)
    dy = np.asarray(projection.sensor_to_physical(21.25, 47.75 + epsilon)) - np.asarray(physical)
    finite_difference = np.column_stack((dx / epsilon, dy / epsilon))
    assert jacobian == pytest.approx(finite_difference, abs=1e-7)


def test_suite_is_deterministic_bound_and_evaluator_compatible(tmp_path: Path):
    config = IndependentRendererConfig(image_size=96, supersample=1)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_records = [
        generate_independent_pair(first, index, architecture, 991, config)
        for index, architecture in enumerate(("dram", "finfet"))
    ]
    second_records = [
        generate_independent_pair(second, index, architecture, 991, config)
        for index, architecture in enumerate(("dram", "finfet"))
    ]
    first_metadata = write_independent_suite(first, first_records, config)
    second_metadata = write_independent_suite(second, second_records, config)

    assert first_metadata["dataset_sha256"] == second_metadata["dataset_sha256"]
    assert verify_independent_suite(first) == first_metadata
    evaluate.validate_records(first, first_records, expected_image_size=96)
    assert first_metadata["code_separation"]["primary_path_imports"] == []
    assert first_metadata["generator_source_sha256"] == independent_renderer.generator_source_sha256()
    for record in first_records:
        assert np.hypot(record["center_x"] - 47.5, record["center_y"] - 47.5) > 1.0
        assert 0.096 < record["actual_scale"] < 0.104
        assert record["generator_provenance"]["primary_path_imports"] == []
        reference = np.asarray(Image.open(first / record["reference"]))
        search = np.asarray(Image.open(first / record["search"]))
        assert reference.shape == search.shape == (96, 96)
        assert not np.array_equal(reference, search)


def test_suite_verifier_rejects_image_tampering(tmp_path: Path):
    config = IndependentRendererConfig(image_size=64, supersample=1)
    record = generate_independent_pair(tmp_path, 0, "dram", 17, config)
    write_independent_suite(tmp_path, [record], config)
    path = tmp_path / record["search"]
    image = np.asarray(Image.open(path)).copy()
    image[0, 0] ^= np.uint8(1)
    Image.fromarray(image, mode="L").save(path)
    with pytest.raises(ValueError, match="input image hash mismatch"):
        verify_independent_suite(tmp_path)


def test_sidecars_match_manifest_records(tmp_path: Path):
    config = IndependentRendererConfig(image_size=64, supersample=1)
    records = [generate_independent_pair(tmp_path, 0, "finfet", 29, config)]
    write_independent_suite(tmp_path, records, config)
    manifest_record = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    sidecar_record = json.loads((tmp_path / f"{records[0]['id']}.json").read_text(encoding="utf-8"))
    assert manifest_record == sidecar_record == records[0]
