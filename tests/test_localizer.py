import numpy as np
from PIL import Image
from scipy import ndimage

import drift_sense.localizer as localizer_module
from drift_sense.dataset import GeneratorConfig, generate_pair
from drift_sense.localizer import LocalizationConfig, _resize, localize
from drift_sense.representations import PeriodicTransformEstimate


def test_baseline_recovers_nonperiodic_patch_center():
    rng = np.random.default_rng(5)
    reference = rng.normal(size=(100, 100)).astype(np.float32)
    template = _resize(reference, 0.1)
    search = rng.normal(0, 0.05, size=(100, 100)).astype(np.float32)
    search[40:50, 60:70] += template
    prediction = localize(reference, search, LocalizationConfig(method="baseline0"))
    assert np.hypot(prediction.x - 64.5, prediction.y - 44.5) < 0.8


def test_invalid_method_is_rejected():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    try:
        localize(image, image, LocalizationConfig(method="unknown"))
    except ValueError as exc:
        assert "unknown method" in str(exc)
    else:
        raise AssertionError("unknown method was accepted")


def test_full_uses_center_nearest_match_for_exact_periodicity():
    y, x = np.indices((1000, 1000), dtype=np.float32)
    reference = np.cos(2 * np.pi * x / 130) + np.cos(2 * np.pi * y / 150)
    search = np.cos(2 * np.pi * x / 13) + np.cos(2 * np.pi * y / 15)

    prediction = localize(
        reference,
        search,
        LocalizationConfig(method="full", top_k=8, rotation_range=0.0),
    )

    assert prediction.ambiguity_flag
    assert prediction.tied_count > 8
    assert prediction.selected_rotation_deg == 0.0
    assert prediction.score >= prediction.selected_score
    assert prediction.score_margin >= 0.0
    assert abs(prediction.score_margin - (prediction.score - prediction.runner_up_score)) < 1e-7
    assert np.hypot(prediction.x - 499.5, prediction.y - 499.5) < 15.0


def test_generated_periodic_suite_uses_nearest_valid_phase(tmp_path):
    cfg = GeneratorConfig(image_size=1000, supersample=2, suite="periodic_ambiguity")
    for index, architecture in enumerate(("dram", "finfet")):
        record = generate_pair(tmp_path, index, architecture, seed=41, cfg=cfg)
        reference = np.asarray(Image.open(tmp_path / record["reference"]).convert("L"), np.float32)
        search = np.asarray(Image.open(tmp_path / record["search"]).convert("L"), np.float32)

        prediction = localize(reference, search, LocalizationConfig(method="full"))
        error = np.hypot(
            prediction.x - record["center_x"],
            prediction.y - record["center_y"],
        )

        assert prediction.ambiguity_flag
        assert error < 1.0
        assert "residual" in prediction.channel_scores
        assert "residual_evidence" in prediction.ambiguity_evidence
        assert prediction.pipeline_stages["lattice_family_candidates"] is True


def test_full_uses_bounded_transform_fallback_when_phase_confidence_is_low(monkeypatch):
    called = []
    low_confidence = PeriodicTransformEstimate(
        scale=0.1,
        rotation_deg=0.0,
        pitch_x=10.0,
        pitch_y=15.0,
        x_vector=(0.1, 0.0),
        y_vector=(0.0, 1 / 15),
        axis_separable=False,
        confidence=0.0,
    )
    monkeypatch.setattr(
        localizer_module,
        "estimate_periodic_transform",
        lambda *args: low_confidence,
    )

    def fake_fallback(reference, search, cfg, pitch_x, pitch_y):
        called.append((cfg.scale_range, cfg.rotation_range, pitch_x, pitch_y))
        return 0.1, 0.0

    monkeypatch.setattr(localizer_module, "_bounded_residual_transform", fake_fallback)
    y, x = np.indices((400, 400), dtype=np.float32)
    reference = np.cos(2 * np.pi * x / 100) + np.cos(2 * np.pi * y / 150)
    search = np.cos(2 * np.pi * x / 10) + np.cos(2 * np.pi * y / 15)

    prediction = localize(reference, search, LocalizationConfig(method="full"))

    assert called == [(0.006, 3.0, 10.0, 15.0)]
    assert prediction.method == "full"
    assert prediction.spectral_confidence == 0.0
    assert prediction.selected_scale == 0.1
    assert prediction.selected_rotation_deg == 0.0
    assert prediction.runtime_ms > 0.0


def test_full_stage_controls_share_one_pipeline_and_are_diagnostic():
    ref_y, ref_x = np.indices((600, 600), dtype=np.float32)
    sea_y, sea_x = np.indices((300, 300), dtype=np.float32)
    reference = np.cos(2 * np.pi * ref_x / 100) + np.cos(2 * np.pi * ref_y / 150)
    search = np.cos(2 * np.pi * sea_x / 10) + np.cos(2 * np.pi * sea_y / 15)
    cfg = LocalizationConfig(
        method="full",
        rotation_range=0.0,
        enable_phase_calibration=False,
        periodic_evidence_channel="gradient",
        enable_spatial_residual=False,
        enable_lattice_grouping=False,
        enable_ambiguity_rule=False,
        subpixel_refinement="none",
    )

    prediction = localize(reference, search, cfg)

    assert prediction.method == "full"
    assert prediction.pipeline_stages == {
        "phase_calibration": False,
        "evidence_channel": "gradient",
        "spatial_residual": False,
        "lattice_family_candidates": False,
        "ambiguity_rule": False,
        "subpixel_refinement": "none",
    }
    assert not prediction.ambiguity_flag
    assert "residual" in prediction.channel_scores
