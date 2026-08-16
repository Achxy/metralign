from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from PIL import Image

from drift_sense.dataset import AUGMENTATIONS, GeneratorConfig, generate_pair, write_manifest
from drift_sense.distortions import AcquisitionParameters
from drift_sense.geometry import CaptureGeometry
from drift_sense.sem_render import render_capture
from drift_sense.sem_render_alt import _polyphase_sample


def test_pair_generation_and_ground_truth(tmp_path: Path):
    cfg = GeneratorConfig(image_size=96, supersample=1, suite="iid")
    record = generate_pair(tmp_path, 0, "dram", 17, cfg)
    write_manifest(tmp_path, [record])
    reference = np.asarray(Image.open(tmp_path / record["reference"]))
    search = np.asarray(Image.open(tmp_path / record["search"]))
    assert reference.shape == search.shape == (96, 96)
    assert 0 <= record["center_x"] < 96
    assert 0 <= record["center_y"] < 96
    assert record["center_x"] == record["center_pre_scan_x"] + record["center_scan_shift_x"]
    assert record["center_y"] == record["center_pre_scan_y"]
    assert record["template_rotation_deg"] == -record["rotation_deg"]
    search_geometry = CaptureGeometry(**record["search_geometry"])
    recovered_x, recovered_y = search_geometry.world_to_sensor(
        record["physical_center_x"], record["physical_center_y"]
    )
    assert abs(recovered_x - record["center_pre_scan_x"]) < 1e-9
    assert abs(recovered_y - record["center_pre_scan_y"]) < 1e-9
    assert record["reference_geometry"]["world_center_x"] != 0
    assert (tmp_path / "manifest.jsonl").is_file()


def test_cross_generator_uses_distinct_resampler(tmp_path: Path):
    cfg = GeneratorConfig(image_size=64, supersample=2, suite="cross_generator")
    record = generate_pair(tmp_path, 0, "finfet", 19, cfg)
    assert record["distortion_parameters"]["reference_resampler"] == "polyphase_kaiser"
    assert record["distortion_parameters"]["search_resampler"] == "polyphase_hann"
    assert record["distortion_parameters"]["reference_renderer"] == "alternate_polyphase_kaiser"
    assert record["distortion_parameters"]["search_renderer"] == "alternate_polyphase_hann"


def test_primary_captures_use_distinct_sampling_paths(tmp_path: Path):
    cfg = GeneratorConfig(image_size=64, supersample=2, suite="iid", resampler="area")
    record = generate_pair(tmp_path, 0, "dram", 23, cfg)
    params = record["distortion_parameters"]
    assert params["reference_resampler"] == "area"
    assert params["search_resampler"] == "lanczos"


def test_supersample_one_reports_no_resampling(tmp_path: Path):
    cfg = GeneratorConfig(image_size=64, supersample=1, suite="iid", resampler="lanczos")
    record = generate_pair(tmp_path, 0, "dram", 29, cfg)
    params = record["distortion_parameters"]
    assert params["reference_resampler"] == "none"
    assert params["search_resampler"] == "none"


def test_periodic_ambiguity_is_ideal_and_centered(tmp_path: Path):
    cfg = GeneratorConfig(image_size=96, supersample=1, suite="periodic_ambiguity")
    record = generate_pair(tmp_path, 0, "finfet", 37, cfg)
    assert record["distortion_parameters"]["geometry_variant"] == "ideal"
    assert abs(record["center_x"] - 47.5) < 0.1
    assert abs(record["center_y"] - 47.5) < 1e-12


def test_alternate_polyphase_sampling_preserves_pixel_center_phase():
    factor = 2
    size = 96
    x = (np.arange(size * factor, dtype=np.float32) + 0.5) / factor - 0.5
    ramp = np.broadcast_to(x, (size * factor, size * factor)).copy()
    for window in ("kaiser", "hann"):
        sampled = _polyphase_sample(ramp, factor, window)
        # Ignore filter transients at the boundary; a linear interior ramp must
        # remain registered to the declared output pixel-center coordinates.
        assert np.max(np.abs(sampled[20:-20, 20:-20] - np.arange(size)[None, 20:-20])) < 2e-3


def test_area_and_lanczos_sampling_paths_are_distinct():
    geometry = CaptureGeometry(64, 64, 0.0, 0.0, 1.0)
    acquisition = AcquisitionParameters(
        edge_strength=0,
        psf_sigma=0,
        poisson_peak=0,
        gaussian_sigma=0,
        slow_intensity_drift=0,
        scan_jitter_sigma=0,
    )
    area, _ = render_capture("dram", 31, geometry, acquisition, 32, 2, "area")
    lanczos, _ = render_capture("dram", 31, geometry, acquisition, 32, 2, "lanczos")
    assert area.shape == lanczos.shape == (64, 64)
    assert not np.allclose(area, lanczos)


def test_every_stress_component_is_independently_switchable(tmp_path: Path):
    neutral = {
        "edge_response": ("edge_strength", 0.0),
        "psf_blur": ("psf_sigma", 0.0),
        "poisson_noise": ("poisson_peak", 0.0),
        "gaussian_noise": ("gaussian_sigma", 0.0),
        "gain": ("gain", 1.0),
        "offset": ("offset", 0.0),
        "slow_drift": ("slow_intensity_drift", 0.0),
        "scan_jitter": ("scan_jitter_sigma", 0.0),
    }
    for index, name in enumerate(AUGMENTATIONS):
        cfg = GeneratorConfig(
            image_size=64,
            supersample=1,
            suite="iid",
            disabled_augmentations=(name,),
        )
        record = generate_pair(tmp_path, index, "dram", 101, cfg)
        assert record["distortion_parameters"]["disabled_augmentations"] == [name]
        if name in neutral:
            field, expected = neutral[name]
            assert record["noise_parameters"]["reference"][field] == expected
            assert record["noise_parameters"]["search"][field] == expected


def test_generator_config_is_validated_after_json_overrides(tmp_path: Path):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"image_size": 32, "supersample": 0}))
    result = subprocess.run(
        [
            sys.executable,
            "generate_dataset.py",
            "--architecture",
            "dram",
            "--num-pairs",
            "1",
            "--output-dir",
            str(tmp_path / "out"),
            "--config",
            str(config),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "invalid generator configuration" in result.stderr
