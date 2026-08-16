"""Deterministic dataset generation and manifest handling."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .distortions import AcquisitionParameters
from .geometry import CaptureGeometry
from .sem_render import render_capture, to_uint8
from .sem_render_alt import render_capture_alternate


SUITE_SEED_OFFSETS = {
    "iid": 0,
    "high_noise": 100_000,
    "geometry_ood": 200_000,
    "transform_ood": 300_000,
    "periodic_ambiguity": 400_000,
    "scan_distortion": 500_000,
    "cross_generator": 600_000,
}

ACQUISITION_AUGMENTATIONS = (
    "edge_response",
    "psf_blur",
    "poisson_noise",
    "gaussian_noise",
    "gain",
    "offset",
    "slow_drift",
    "scan_jitter",
)
GEOMETRY_AUGMENTATIONS = (
    "rotation",
    "scale_error",
    "anisotropy",
    "geometric_drift",
    "pitch_variation",
    "center_variation",
    "width_variation",
    "edge_roughness",
)
AUGMENTATIONS = ACQUISITION_AUGMENTATIONS + GEOMETRY_AUGMENTATIONS


@dataclass
class GeneratorConfig:
    image_size: int = 1000
    nominal_scale: float = 0.1
    supersample: int = 2
    suite: str = "iid"
    difficulty: str = "medium"
    resampler: str = "area"
    disabled_augmentations: tuple[str, ...] = ()


def _acquisition_params(
    rng: np.random.Generator,
    suite: str,
    is_search: bool,
    difficulty: str,
    disabled_augmentations: tuple[str, ...] = (),
) -> AcquisitionParameters:
    disabled = set(disabled_augmentations)
    worse = 1.0 if is_search else 0.68
    noise_mult = 1.0
    jitter_mult = 1.0
    if suite == "high_noise":
        noise_mult = 1.8
    if suite == "scan_distortion":
        jitter_mult = 3.0
    if suite == "cross_generator":
        noise_mult = 1.25
        jitter_mult = 1.35
    difficulty_factor = {"easy": 0.60, "medium": 1.0, "hard": 1.45}[difficulty]
    noise_mult *= difficulty_factor
    jitter_mult *= difficulty_factor
    peak_base = (120.0 if is_search else 230.0) / noise_mult
    sampled = {
        "edge_strength": float(rng.uniform(0.12, 0.28)),
        "psf_sigma": float(rng.uniform(0.55, 1.0) * worse),
        "poisson_peak": float(rng.uniform(0.8, 1.25) * peak_base),
        "gaussian_sigma": float(rng.uniform(0.010, 0.025) * worse * noise_mult),
        "gain": float(rng.uniform(0.88, 1.12)),
        "offset": float(rng.uniform(-0.035, 0.035)),
        "slow_intensity_drift": float(rng.uniform(-0.055, 0.055)),
        "scan_jitter_sigma": float(rng.uniform(0.12, 0.34) * jitter_mult),
        "scan_jitter_correlation": float(rng.uniform(3.0, 10.0)),
    }
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
    for component, (field, value) in neutral.items():
        if component in disabled:
            sampled[field] = value
    return AcquisitionParameters(**sampled)


def _geometry_params(
    rng: np.random.Generator,
    cfg: GeneratorConfig,
    target_search_xy: tuple[float, float],
) -> tuple[CaptureGeometry, CaptureGeometry]:
    size = cfg.image_size
    suite = cfg.suite
    disabled = set(cfg.disabled_augmentations)
    difficulty_factor = {"easy": 0.60, "medium": 1.0, "hard": 1.45}[cfg.difficulty]
    scale_error = 0.0 if "scale_error" in disabled else (0.008 if suite != "transform_ood" else 0.025) * difficulty_factor
    rotation = 0.0 if "rotation" in disabled else (0.35 if suite != "transform_ood" else 1.15) * difficulty_factor
    anisotropy = 0.0 if "anisotropy" in disabled else (0.0015 if suite != "transform_ood" else 0.006) * difficulty_factor
    drift = 0.0 if "geometric_drift" in disabled else (0.45 if suite != "scan_distortion" else 1.8) * difficulty_factor
    search = CaptureGeometry(
        width=size,
        height=size,
        world_center_x=0.0,
        world_center_y=0.0,
        pixel_size=float(1.0 + rng.uniform(-scale_error, scale_error)),
        rotation_deg=float(rng.uniform(-rotation, rotation)),
        anisotropy=float(rng.uniform(-anisotropy, anisotropy)),
        drift_linear=float(rng.uniform(-drift, drift)),
        drift_quadratic=float(rng.uniform(-drift * 0.4, drift * 0.4)),
    )
    tx, ty = target_search_xy
    target_wx, target_wy = search.sensor_to_world(np.asarray(tx), np.asarray(ty))
    reference = CaptureGeometry(
        width=size,
        height=size,
        world_center_x=float(target_wx),
        world_center_y=float(target_wy),
        pixel_size=float(cfg.nominal_scale * (1.0 + rng.uniform(-scale_error, scale_error))),
        rotation_deg=float(rng.uniform(-rotation, rotation)),
        anisotropy=float(rng.uniform(-anisotropy, anisotropy)),
        drift_linear=float(rng.uniform(-drift, drift)),
        drift_quadratic=float(rng.uniform(-drift * 0.4, drift * 0.4)),
    )
    return reference, search


def generate_pair(
    output_dir: Path,
    index: int,
    architecture: str,
    seed: int,
    cfg: GeneratorConfig,
) -> dict:
    """Generate a pair from one continuous wafer field and independent captures."""
    suite_offset = SUITE_SEED_OFFSETS.get(cfg.suite)
    if suite_offset is None:
        raise ValueError(f"unknown suite: {cfg.suite}")
    sample_seed = int(seed + suite_offset + index * 104729)
    rng = np.random.default_rng(sample_seed)
    size = cfg.image_size
    # Keep the full reference footprint inside search. Ambiguity suite favors the
    # center where repeated sites create deliberately small score margins.
    margin = max(0.12 * size, 0.5 * size * cfg.nominal_scale + 8)
    if cfg.suite == "periodic_ambiguity":
        target = ((size - 1.0) / 2.0, (size - 1.0) / 2.0)
    else:
        target = tuple(rng.uniform(margin, size - 1 - margin, size=2))
    reference_geometry, search_geometry = _geometry_params(rng, cfg, target)
    disabled = tuple(cfg.disabled_augmentations)
    ref_acq = _acquisition_params(rng, cfg.suite, False, cfg.difficulty, disabled)
    search_acq = _acquisition_params(rng, cfg.suite, True, cfg.difficulty, disabled)
    disabled_variations = tuple(name for name in disabled if name in GEOMETRY_AUGMENTATIONS)
    geometry_variant = (
        "ood"
        if cfg.suite == "geometry_ood"
        else "ideal"
        if cfg.suite == "periodic_ambiguity"
        else "default"
    )

    if cfg.suite == "cross_generator":
        reference, ref_shifts = render_capture_alternate(
            architecture,
            sample_seed,
            reference_geometry,
            ref_acq,
            sample_seed + 1,
            cfg.supersample,
            geometry_variant,
            "kaiser",
            disabled_variations,
        )
        search, search_shifts = render_capture_alternate(
            architecture,
            sample_seed,
            search_geometry,
            search_acq,
            sample_seed + 2,
            cfg.supersample,
            geometry_variant,
            "hann",
            disabled_variations,
        )
        reference_renderer = "alternate_polyphase_kaiser"
        search_renderer = "alternate_polyphase_hann"
    else:
        search_resampler = "lanczos" if cfg.resampler == "area" else "area"
        reference, ref_shifts = render_capture(
            architecture,
            sample_seed,
            reference_geometry,
            ref_acq,
            sample_seed + 1,
            cfg.supersample,
            cfg.resampler,
            geometry_variant,
            disabled_variations,
        )
        search, search_shifts = render_capture(
            architecture,
            sample_seed,
            search_geometry,
            search_acq,
            sample_seed + 2,
            cfg.supersample,
            search_resampler,
            geometry_variant,
            disabled_variations,
        )
        reference_renderer = f"primary_{cfg.resampler}"
        search_renderer = f"primary_{search_resampler}"

    target_wx = reference_geometry.world_center_x
    target_wy = reference_geometry.world_center_y
    center_pre_scan_x, center_pre_scan_y = search_geometry.world_to_sensor(target_wx, target_wy)
    center_scan_shift_x = float(
        np.interp(center_pre_scan_y, np.arange(size), search_shifts)
    )
    center_x = center_pre_scan_x + center_scan_shift_x
    center_y = center_pre_scan_y

    stem = f"{index:06d}_{architecture}"
    reference_name = f"{stem}_reference.png"
    search_name = f"{stem}_search.png"
    Image.fromarray(to_uint8(reference), mode="L").save(output_dir / reference_name)
    Image.fromarray(to_uint8(search), mode="L").save(output_dir / search_name)
    record = {
        "id": stem,
        "architecture": architecture,
        "suite": cfg.suite,
        "difficulty": cfg.difficulty,
        "seed": sample_seed,
        "reference": reference_name,
        "search": search_name,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "physical_center_x": float(target_wx),
        "physical_center_y": float(target_wy),
        "center_pre_scan_x": float(center_pre_scan_x),
        "center_pre_scan_y": float(center_pre_scan_y),
        "center_scan_shift_x": center_scan_shift_x,
        "nominal_scale": cfg.nominal_scale,
        "image_size": cfg.image_size,
        "actual_scale": float(reference_geometry.pixel_size / search_geometry.pixel_size),
        # Relative sensor-to-world rotation and the corresponding image-space
        # template warp have opposite signs under the image y-down convention.
        "rotation_deg": float(reference_geometry.rotation_deg - search_geometry.rotation_deg),
        "template_rotation_deg": float(search_geometry.rotation_deg - reference_geometry.rotation_deg),
        "reference_geometry": reference_geometry.to_dict(),
        "search_geometry": search_geometry.to_dict(),
        "noise_parameters": {
            "reference": ref_acq.to_dict(),
            "search": search_acq.to_dict(),
        },
        "distortion_parameters": {
            "geometry_variant": geometry_variant,
            "reference_row_shift_rms": float(np.sqrt(np.mean(ref_shifts**2))),
            "search_row_shift_rms": float(np.sqrt(np.mean(search_shifts**2))),
            "reference_resampler": (
                "none"
                if cfg.supersample == 1
                else cfg.resampler
                if cfg.suite != "cross_generator"
                else "polyphase_kaiser"
            ),
            "search_resampler": (
                "none"
                if cfg.supersample == 1
                else search_resampler
                if cfg.suite != "cross_generator"
                else "polyphase_hann"
            ),
            "reference_renderer": reference_renderer,
            "search_renderer": search_renderer,
            "supersample": cfg.supersample,
            "disabled_augmentations": list(disabled),
        },
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_manifest(output_dir: Path, records: Iterable[dict]) -> Path:
    path = output_dir / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def load_manifest(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
