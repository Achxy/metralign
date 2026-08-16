"""Standalone held-out renderer for generator-transfer evaluation.

This module deliberately has no dependency on the primary synthetic path:
``architectures``, ``geometry``, ``distortions``, ``sem_render``,
``sem_render_alt``, and ``dataset`` are not imported.  It defines its own
physical coordinate convention, homogeneous sensor projection, layout fields,
detector response, noise ordering, manifest writer, and artifact verifier.

The separation is a code-provenance claim, not a claim that synthetic images
are independent evidence of performance on physical SEM/TEM instruments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


SUITE_NAME = "independent_renderer"
SCHEMA_VERSION = 1
SAMPLE_SEED_STRIDE = 130_363
PRIMARY_PATH_MODULES = (
    "drift_sense.architectures",
    "drift_sense.geometry",
    "drift_sense.distortions",
    "drift_sense.sem_render",
    "drift_sense.sem_render_alt",
    "drift_sense.dataset",
)


@dataclass(frozen=True)
class IndependentRendererConfig:
    """Configuration for the disjoint held-out capture model."""

    image_size: int = 1000
    nominal_scale: float = 0.1
    supersample: int = 2
    difficulty: str = "medium"

    def validate(self) -> None:
        if self.image_size < 64:
            raise ValueError("image_size must be at least 64")
        if self.supersample not in {1, 2, 3, 4}:
            raise ValueError("supersample must be between 1 and 4")
        if not 0.05 <= self.nominal_scale <= 0.25:
            raise ValueError("nominal_scale must be between 0.05 and 0.25")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty must be easy, medium, or hard")


@dataclass(frozen=True)
class HomogeneousProjection:
    """A self-contained sensor-pixel to physical-nanometre homography."""

    width: int
    height: int
    matrix: tuple[float, ...]
    convention: str = "sensor_xy_down_to_physical_nm_xy"

    def _array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)

    def sensor_to_physical(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        h = self._array()
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        denominator = h[2, 0] * x_array + h[2, 1] * y_array + h[2, 2]
        px = (h[0, 0] * x_array + h[0, 1] * y_array + h[0, 2]) / denominator
        py = (h[1, 0] * x_array + h[1, 1] * y_array + h[1, 2]) / denominator
        return px, py

    def physical_to_sensor(self, px: float, py: float) -> tuple[float, float]:
        inverse = np.linalg.inv(self._array())
        point = inverse @ np.asarray([px, py, 1.0], dtype=np.float64)
        point /= point[2]
        return float(point[0]), float(point[1])

    def local_jacobian(self, x: float, y: float) -> np.ndarray:
        """Return the physical-nm per sensor-pixel Jacobian at one point."""
        h = self._array()
        denominator = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        numerators = np.asarray(
            [
                h[0, 0] * x + h[0, 1] * y + h[0, 2],
                h[1, 0] * x + h[1, 1] * y + h[1, 2],
            ]
        )
        jacobian = np.empty((2, 2), dtype=np.float64)
        for axis in range(2):
            derivative_denominator = h[2, axis]
            derivatives_numerator = h[:2, axis]
            jacobian[:, axis] = (
                derivatives_numerator * denominator
                - numerators * derivative_denominator
            ) / (denominator * denominator)
        return jacobian

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["matrix"] = list(self.matrix)
        return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_source_sha256() -> str:
    """Fingerprint the complete standalone implementation source."""
    return _sha256_file(Path(__file__))


def _splitmix64(value: np.ndarray | int) -> np.ndarray:
    """Integer mixer used for persistent per-feature process variation."""
    x = np.asarray(value, dtype=np.uint64)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _feature_noise(
    index_x: np.ndarray, index_y: np.ndarray, seed: int, channel: int
) -> np.ndarray:
    """Persistent feature noise in [-1, 1] without the primary sine hash."""
    ux = np.asarray(index_x, dtype=np.int64).astype(np.uint64)
    uy = np.asarray(index_y, dtype=np.int64).astype(np.uint64)
    with np.errstate(over="ignore"):
        key = (
            ux * np.uint64(0xD6E8FEB86659FD93)
            ^ uy * np.uint64(0xA5A3564E27F8862F)
            ^ np.uint64(seed & ((1 << 64) - 1))
            ^ np.uint64(channel) * np.uint64(0x8CB92BA72F3D8DD7)
        )
    mixed = _splitmix64(key)
    top = (mixed >> np.uint64(11)).astype(np.float64)
    return (top * (1.0 / float(1 << 53)) * 2.0 - 1.0).astype(np.float32)


def _smooth_transition(value: np.ndarray, half_width: np.ndarray | float) -> np.ndarray:
    width = np.maximum(np.asarray(half_width, dtype=np.float32), 1e-4)
    return 0.5 - 0.5 * np.tanh(value / width)


def _physical_value_noise(
    px: np.ndarray,
    py: np.ndarray,
    seed: int,
    channel: int,
    spacing_nm: float,
) -> np.ndarray:
    """Continuous physical-space process field from a disjoint integer grid."""
    grid_x = np.floor(px / spacing_nm).astype(np.int64)
    grid_y = np.floor(py / spacing_nm).astype(np.int64)
    fraction_x = (px / spacing_nm - grid_x).astype(np.float32)
    fraction_y = (py / spacing_nm - grid_y).astype(np.float32)
    fraction_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x)
    fraction_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y)
    n00 = _feature_noise(grid_x, grid_y, seed, channel)
    n10 = _feature_noise(grid_x + 1, grid_y, seed, channel)
    n01 = _feature_noise(grid_x, grid_y + 1, seed, channel)
    n11 = _feature_noise(grid_x + 1, grid_y + 1, seed, channel)
    lower = n00 + fraction_x * (n10 - n00)
    upper = n01 + fraction_x * (n11 - n01)
    return (lower + fraction_y * (upper - lower)).astype(np.float32)


def _seed_parameters(seed: int, architecture: str) -> dict[str, float]:
    """Sample global process values with NumPy SeedSequence, not primary hashes."""
    label = 0x4452414D if architecture == "dram" else 0x46494E46
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, label])))
    if architecture == "dram":
        return {
            "pitch_x": float(rng.uniform(11.2, 12.6)),
            "pitch_y": float(rng.uniform(13.4, 15.2)),
            "radius_x": float(rng.uniform(3.1, 3.8)),
            "radius_y": float(rng.uniform(3.4, 4.2)),
            "phase_x": float(rng.uniform(-5.0, 5.0)),
            "phase_y": float(rng.uniform(-5.0, 5.0)),
        }
    return {
        "fin_pitch": float(rng.uniform(8.2, 9.5)),
        "gate_pitch": float(rng.uniform(15.4, 18.2)),
        "fin_width": float(rng.uniform(2.5, 3.3)),
        "gate_width": float(rng.uniform(4.3, 5.6)),
        "phase_x": float(rng.uniform(-4.0, 4.0)),
        "phase_y": float(rng.uniform(-7.0, 7.0)),
    }


def _dram_layout(px: np.ndarray, py: np.ndarray, seed: int) -> np.ndarray:
    """Brick-array landing pads with discrete defects and CMP relief."""
    parameters = _seed_parameters(seed, "dram")
    pitch_x = parameters["pitch_x"]
    pitch_y = parameters["pitch_y"]
    shifted_y = py - parameters["phase_y"]
    row = np.floor(shifted_y / pitch_y + 0.5).astype(np.int64)
    stagger = np.where((row & 1) == 0, -0.22, 0.28) * pitch_x
    shifted_x = px - parameters["phase_x"] - stagger
    column = np.floor(shifted_x / pitch_x + 0.5).astype(np.int64)

    center_dx = 0.62 * _feature_noise(column, row, seed, 1)
    center_dy = 0.54 * _feature_noise(column, row, seed, 2)
    radius_x = parameters["radius_x"] * (
        1.0 + 0.13 * _feature_noise(column, row, seed, 3)
    )
    radius_y = parameters["radius_y"] * (
        1.0 + 0.12 * _feature_noise(column, row, seed, 4)
    )
    local_x = shifted_x - column * pitch_x - center_dx
    local_y = shifted_y - row * pitch_y - center_dy
    ellipse = np.sqrt((local_x / radius_x) ** 2 + (local_y / radius_y) ** 2) - 1.0
    pad = _smooth_transition(ellipse, 0.10)

    # Cell dose, ellipticity, and asymmetric landing-pad response are
    # persistent layout properties. Their amplitude is intentionally above the
    # search detector's shot-noise floor so a physical site remains
    # identifiable after the 10:1 change in sampling density.
    cell_dose = 0.72 + 0.28 * (
        0.5 + 0.5 * _feature_noise(column, row, seed, 7)
    )
    asymmetry = 1.0 + 0.16 * _feature_noise(column, row, seed, 8) * np.clip(
        local_x / np.maximum(radius_x, 1e-3), -1.0, 1.0
    )
    neighbourhood_dose = 0.88 + 0.12 * (
        0.5 + 0.5 * _feature_noise(column // 3, row // 3, seed, 9)
    )
    pad *= cell_dose * asymmetry * neighbourhood_dose

    defect_key = _feature_noise(column // 5, row // 5, seed, 5)
    missing = (defect_key > 0.965) & (((column + 2 * row) % 5) == 0)
    pad = np.where(missing, 0.18 * pad, pad)
    bridge = _smooth_transition(np.abs(local_y) - 0.55, 0.13) * (
        _feature_noise(column, row, seed, 6) > 0.90
    )
    wordline = 0.5 + 0.5 * np.cos(2.0 * np.pi * shifted_y / (2.0 * pitch_y))
    cmp_wave = np.sin(0.014 * px + 0.009 * py + 0.4 * np.sin(0.006 * py))
    # Lithography dose/CMP variation is sampled in physical coordinates rather
    # than sensor coordinates. It therefore persists across magnification while
    # supplying realistic non-periodic site evidence.
    process_field = _physical_value_noise(px, py, seed, 20, 43.0)
    grain_field = _physical_value_noise(px, py, seed, 21, 19.0)
    pad *= 1.0 + 0.08 * grain_field
    yield_field = (
        0.11
        + 0.72 * pad
        + 0.07 * wordline
        + 0.07 * bridge
        + 0.035 * cmp_wave
        + 0.065 * process_field
    )
    return np.clip(yield_field, 0.0, 1.0).astype(np.float32)


def _finfet_layout(px: np.ndarray, py: np.ndarray, seed: int) -> np.ndarray:
    """Gate/fin layers with line-edge wandering, cuts, and sparse contacts."""
    parameters = _seed_parameters(seed, "finfet")
    fin_pitch = parameters["fin_pitch"]
    gate_pitch = parameters["gate_pitch"]
    shifted_x = px - parameters["phase_x"]
    shifted_y = py - parameters["phase_y"]
    fin_index = np.floor(shifted_x / fin_pitch + 0.5).astype(np.int64)
    gate_index = np.floor(shifted_y / gate_pitch + 0.5).astype(np.int64)

    fin_phase = np.pi * _feature_noise(fin_index, np.zeros_like(fin_index), seed, 10)
    gate_phase = np.pi * _feature_noise(np.zeros_like(gate_index), gate_index, seed, 11)
    fin_wander = 0.24 * np.sin(2.0 * np.pi * py / 73.0 + fin_phase)
    gate_wander = 0.27 * np.sin(2.0 * np.pi * px / 91.0 + gate_phase)
    segment_noise = _feature_noise(fin_index, gate_index, seed, 12)
    fin_width = parameters["fin_width"] * (1.0 + 0.075 * segment_noise)
    gate_width = parameters["gate_width"] * (
        1.0 + 0.065 * _feature_noise(fin_index, gate_index, seed, 13)
    )
    fin_distance = np.abs(shifted_x - fin_index * fin_pitch - fin_wander) - fin_width / 2
    gate_distance = np.abs(shifted_y - gate_index * gate_pitch - gate_wander) - gate_width / 2
    fin = _smooth_transition(fin_distance, 0.30)
    gate = _smooth_transition(gate_distance, 0.36)

    block_x = np.floor(shifted_x / (6.0 * fin_pitch)).astype(np.int64)
    block_y = np.floor(shifted_y / (4.0 * gate_pitch)).astype(np.int64)
    cut = (_feature_noise(block_x, block_y, seed, 14) > 0.91) & (
        np.abs((shifted_y / gate_pitch) % 4.0 - 2.0) < 0.22
    )
    gate = np.where(cut, 0.08 * gate, gate)
    contact_select = _feature_noise(fin_index // 2, gate_index // 2, seed, 15) > 0.47
    contact_distance = np.hypot(
        (shifted_x - fin_index * fin_pitch) / 2.4,
        (shifted_y - gate_index * gate_pitch) / 2.8,
    ) - 1.0
    contact = _smooth_transition(contact_distance, 0.12) * contact_select
    yield_field = 0.09 + 0.39 * fin + 0.25 * gate + 0.17 * fin * gate + 0.12 * contact
    return np.clip(yield_field, 0.0, 1.0).astype(np.float32)


def _layout(architecture: str, px: np.ndarray, py: np.ndarray, seed: int) -> np.ndarray:
    if architecture == "dram":
        return _dram_layout(px, py, seed)
    if architecture == "finfet":
        return _finfet_layout(px, py, seed)
    raise ValueError(f"unsupported architecture: {architecture!r}")


def _projection(
    size: int,
    physical_center: tuple[float, float],
    nm_per_pixel: float,
    rotation_deg: float,
    aspect: float,
    shear: float,
    perspective: tuple[float, float],
) -> HomogeneousProjection:
    center = (size - 1.0) / 2.0
    theta = np.deg2rad(rotation_deg)
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0, 0, 1]],
        dtype=np.float64,
    )
    sensor_scale = np.asarray(
        [[nm_per_pixel * (1.0 + aspect), nm_per_pixel * shear, 0.0],
         [0.0, nm_per_pixel * (1.0 - aspect), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projective = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective[0], perspective[1], 1.0]],
        dtype=np.float64,
    )
    recenter = np.asarray(
        [[1.0, 0.0, -center], [0.0, 1.0, -center], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translate = np.asarray(
        [[1.0, 0.0, physical_center[0]], [0.0, 1.0, physical_center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = translate @ rotation @ sensor_scale @ projective @ recenter
    return HomogeneousProjection(size, size, tuple(float(value) for value in matrix.flat))


def _sensor_grid(size: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(size * supersample, dtype=np.float32) + 0.5) / supersample - 0.5
    return np.meshgrid(axis, axis)


def _capture_parameters(
    sample_seed: int, size: int, nominal_scale: float, difficulty: str
) -> tuple[HomogeneousProjection, HomogeneousProjection, tuple[float, float], dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([sample_seed, 0xC4A7])))
    difficulty_scale = {"easy": 0.55, "medium": 1.0, "hard": 1.45}[difficulty]
    margin = max(0.13 * size, 0.5 * size * nominal_scale + 9.0)
    target_sensor = tuple(float(value) for value in rng.uniform(margin, size - 1 - margin, 2))
    search_nm_per_pixel = float(rng.uniform(0.96, 1.04))
    reference_nm_per_pixel = float(
        search_nm_per_pixel
        * nominal_scale
        * (1.0 + rng.uniform(-0.020, 0.020) * difficulty_scale)
    )
    search_rotation = float(rng.uniform(-0.40, 0.40) * difficulty_scale)
    reference_rotation = float(rng.uniform(-0.40, 0.40) * difficulty_scale)
    projection_parameters = {
        "search": {
            "nm_per_pixel": search_nm_per_pixel,
            "rotation_deg": search_rotation,
            "aspect": float(rng.uniform(-0.0015, 0.0015) * difficulty_scale),
            "shear": float(rng.uniform(-0.0012, 0.0012) * difficulty_scale),
            "perspective": tuple(
                float(value)
                for value in rng.uniform(-1.1e-6, 1.1e-6, 2) * difficulty_scale
            ),
        },
        "reference": {
            "nm_per_pixel": reference_nm_per_pixel,
            "rotation_deg": reference_rotation,
            "aspect": float(rng.uniform(-0.0015, 0.0015) * difficulty_scale),
            "shear": float(rng.uniform(-0.0012, 0.0012) * difficulty_scale),
            "perspective": tuple(
                float(value)
                for value in rng.uniform(-1.1e-6, 1.1e-6, 2) * difficulty_scale
            ),
        },
    }
    search = _projection(size, (0.0, 0.0), **projection_parameters["search"])
    physical_target_arrays = search.sensor_to_physical(*target_sensor)
    physical_target = (float(physical_target_arrays[0]), float(physical_target_arrays[1]))
    reference = _projection(size, physical_target, **projection_parameters["reference"])
    return reference, search, target_sensor, projection_parameters


def _reduce_reference(image: np.ndarray, size: int, supersample: int) -> np.ndarray:
    """Integrating detector with a raised-cosine intra-pixel aperture."""
    if supersample == 1:
        return image
    axis = np.sin(np.pi * (np.arange(supersample) + 0.5) / supersample)
    weights = np.outer(axis, axis).astype(np.float32)
    weights /= np.sum(weights)
    blocks = image.reshape(size, supersample, size, supersample)
    return np.einsum("iajb,ab->ij", blocks, weights, optimize=True).astype(np.float32)


def _reduce_search(image: np.ndarray, size: int, supersample: int) -> np.ndarray:
    """Charge-spread detector followed by phase-centred point sampling."""
    if supersample == 1:
        return image
    spread = ndimage.gaussian_filter(image, sigma=0.38 * supersample, mode="mirror")
    start = (supersample - 1.0) / 2.0
    coordinates = start + supersample * np.arange(size, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return ndimage.map_coordinates(
        spread, (yy, xx), order=3, mode="reflect", prefilter=True
    ).astype(np.float32)


def _detector_response(
    latent: np.ndarray,
    capture_seed: int,
    role: str,
    difficulty: str,
) -> np.ndarray:
    """Separate detector/charging/noise model with role-specific ordering."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([capture_seed, 0xD37EC7])))
    stress = {"easy": 0.65, "medium": 1.0, "hard": 1.40}[difficulty]
    if role == "reference":
        blurred = ndimage.gaussian_filter(latent, sigma=(0.62, 0.48), mode="mirror")
        edge_x = ndimage.sobel(blurred, axis=1, mode="mirror")
        edge_y = ndimage.sobel(blurred, axis=0, mode="mirror")
        response = blurred + 0.025 * np.hypot(edge_x, edge_y)
        gamma = 0.92
        electron_count = 260.0 / stress
        read_noise = 0.008 * stress
    else:
        blurred = ndimage.gaussian_filter(latent, sigma=(0.86, 1.08), mode="reflect")
        laplacian = ndimage.laplace(blurred, mode="reflect")
        response = blurred - 0.035 * laplacian
        gamma = 1.08
        electron_count = 145.0 / stress
        read_noise = 0.014 * stress
    response = np.clip(response, 0.0, 1.0) ** gamma

    height, width = response.shape
    coarse = rng.normal(size=(9, 9)).astype(np.float32)
    charging = ndimage.zoom(coarse, (height / 9.0, width / 9.0), order=3, mode="reflect")
    charging = charging[:height, :width]
    charging -= float(np.mean(charging))
    charging /= max(float(np.std(charging)), 1e-6)
    response = response * (1.0 + 0.018 * stress * charging)
    row_phase = rng.uniform(0.0, 2.0 * np.pi)
    row_gain = 1.0 + 0.007 * stress * np.sin(
        np.arange(height, dtype=np.float32) * (2.0 * np.pi / 37.0) + row_phase
    )
    response *= row_gain[:, None]

    counts = rng.poisson(np.clip(response, 0.0, 1.2) * electron_count).astype(np.float32)
    response = counts / electron_count
    response += rng.normal(0.0, read_noise, response.shape).astype(np.float32)
    gain = float(rng.uniform(0.91, 1.09))
    offset = float(rng.uniform(-0.028, 0.028))
    return np.clip(response * gain + offset, 0.0, 1.0).astype(np.float32)


def _render_capture(
    architecture: str,
    layout_seed: int,
    projection: HomogeneousProjection,
    capture_seed: int,
    supersample: int,
    role: str,
    difficulty: str,
) -> np.ndarray:
    x, y = _sensor_grid(projection.width, supersample)
    physical_x, physical_y = projection.sensor_to_physical(x, y)
    latent = _layout(architecture, physical_x, physical_y, layout_seed)
    if role == "reference":
        sampled = _reduce_reference(latent, projection.width, supersample)
    else:
        sampled = _reduce_search(latent, projection.width, supersample)
    return _detector_response(sampled, capture_seed, role, difficulty)


def _orientation_deg(projection: HomogeneousProjection, x: float, y: float) -> float:
    jacobian = projection.local_jacobian(x, y)
    return float(np.rad2deg(np.arctan2(jacobian[1, 0], jacobian[0, 0])))


def _local_scale(projection: HomogeneousProjection, x: float, y: float) -> float:
    return float(np.sqrt(abs(np.linalg.det(projection.local_jacobian(x, y)))))


def generate_independent_pair(
    output_dir: Path,
    index: int,
    architecture: str,
    seed: int,
    config: IndependentRendererConfig,
) -> dict[str, Any]:
    """Generate one pair and sidecar without using the primary generator path."""
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    if architecture not in {"dram", "finfet"}:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    sample_seed = int(seed + index * SAMPLE_SEED_STRIDE)
    reference_projection, search_projection, declared_target, parameters = _capture_parameters(
        sample_seed,
        config.image_size,
        config.nominal_scale,
        config.difficulty,
    )
    layout_seed = sample_seed + 17
    reference_seed = sample_seed + 29
    search_seed = sample_seed + 43
    reference = _render_capture(
        architecture,
        layout_seed,
        reference_projection,
        reference_seed,
        config.supersample,
        "reference",
        config.difficulty,
    )
    search = _render_capture(
        architecture,
        layout_seed,
        search_projection,
        search_seed,
        config.supersample,
        "search",
        config.difficulty,
    )
    target_physical = reference_projection.sensor_to_physical(
        (config.image_size - 1.0) / 2.0,
        (config.image_size - 1.0) / 2.0,
    )
    center_x, center_y = search_projection.physical_to_sensor(
        float(target_physical[0]), float(target_physical[1])
    )
    if np.hypot(center_x - declared_target[0], center_y - declared_target[1]) > 1e-8:
        raise RuntimeError("projection round-trip changed declared ground truth")

    reference_center = (config.image_size - 1.0) / 2.0
    actual_scale = _local_scale(
        reference_projection, reference_center, reference_center
    ) / _local_scale(search_projection, center_x, center_y)
    physical_rotation = _orientation_deg(
        reference_projection, reference_center, reference_center
    ) - _orientation_deg(search_projection, center_x, center_y)
    stem = f"{index:06d}_{architecture}"
    reference_name = f"{stem}_reference.png"
    search_name = f"{stem}_search.png"
    Image.fromarray(np.clip(np.rint(reference * 255.0), 0, 255).astype(np.uint8), mode="L").save(
        output_dir / reference_name
    )
    Image.fromarray(np.clip(np.rint(search * 255.0), 0, 255).astype(np.uint8), mode="L").save(
        output_dir / search_name
    )
    source_hash = generator_source_sha256()
    configuration_hash = _sha256_bytes(_canonical_json(asdict(config)))
    image_hashes = {
        reference_name: _sha256_file(output_dir / reference_name),
        search_name: _sha256_file(output_dir / search_name),
    }
    record: dict[str, Any] = {
        "id": stem,
        "architecture": architecture,
        "suite": SUITE_NAME,
        "difficulty": config.difficulty,
        "seed": sample_seed,
        "reference": reference_name,
        "search": search_name,
        "center_x": center_x,
        "center_y": center_y,
        "physical_center_x_nm": float(target_physical[0]),
        "physical_center_y_nm": float(target_physical[1]),
        "nominal_scale": config.nominal_scale,
        "actual_scale": actual_scale,
        "rotation_deg": physical_rotation,
        "template_rotation_deg": -physical_rotation,
        "image_size": config.image_size,
        "reference_geometry": reference_projection.to_dict(),
        "search_geometry": search_projection.to_dict(),
        "projection_parameters": parameters,
        "capture_parameters": {
            "layout_seed": layout_seed,
            "reference_capture_seed": reference_seed,
            "search_capture_seed": search_seed,
            "reference_detector": "raised_cosine_integrating_detector",
            "search_detector": "charge_spread_cubic_detector",
            "supersample": config.supersample,
        },
        "generator_provenance": {
            "name": "metralign_standalone_layout_capture",
            "version": SCHEMA_VERSION,
            "module": "drift_sense.independent_renderer",
            "source_sha256": source_hash,
            "configuration_sha256": configuration_hash,
            "primary_path_imports": [],
            "excluded_primary_modules": list(PRIMARY_PATH_MODULES),
            "coordinate_model": "physical_nm_homography",
            "layout_model": "standalone_signed_distance_layout",
            "reference_sampling": "raised_cosine_pixel_integration",
            "search_sampling": "gaussian_charge_spread_cubic_sampling",
            "claim_scope": "code-path separation only; not external experimental evidence",
        },
        "image_sha256": image_hashes,
    }
    # Return the same JSON-domain value written to sidecars/manifests. This
    # prevents tuple/list representation drift from masquerading as provenance
    # drift when records are reloaded.
    record = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    (output_dir / f"{stem}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


def _dataset_sha256(manifest_hash: str, image_hashes: dict[str, str]) -> str:
    digest = sha256()
    digest.update(b"metralign-independent-renderer-v1\0")
    digest.update(bytes.fromhex(manifest_hash))
    for name, value in sorted(image_hashes.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def write_independent_suite(
    output_dir: Path,
    records: Iterable[dict[str, Any]],
    config: IndependentRendererConfig,
) -> dict[str, Any]:
    """Write manifest, provenance metadata, and a machine-checkable hash ledger."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        raise ValueError("cannot write an empty independent suite")
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest_hash = _sha256_file(manifest_path)
    image_hashes = {
        name: value
        for row in rows
        for name, value in row["image_sha256"].items()
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "record_count": len(rows),
        "architectures": sorted({str(row["architecture"]) for row in rows}),
        "configuration": asdict(config),
        "configuration_sha256": _sha256_bytes(_canonical_json(asdict(config))),
        "generator_source": "src/drift_sense/independent_renderer.py",
        "generator_source_sha256": generator_source_sha256(),
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "input_images_sha256": dict(sorted(image_hashes.items())),
        "dataset_sha256": _dataset_sha256(manifest_hash, image_hashes),
        "code_separation": {
            "primary_path_imports": [],
            "excluded_primary_modules": list(PRIMARY_PATH_MODULES),
            "shared_contract_only": [
                "dram/finfet category labels",
                "reference/search grayscale PNG interface",
                "0.1 nominal scale task",
                "evaluation manifest field names",
            ],
        },
        "scope": (
            "Held-out synthetic generator-transfer evidence. It does not replace "
            "evaluation on independently acquired microscope imagery."
        ),
    }
    metadata_path = output_dir / "suite-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    ledger_paths = [
        manifest_path,
        metadata_path,
        *(output_dir / name for name in image_hashes),
        *(output_dir / f"{row['id']}.json" for row in rows),
    ]
    ledger = "".join(
        f"{_sha256_file(path)}  {path.name}\n"
        for path in sorted(ledger_paths, key=lambda item: item.name)
    )
    (output_dir / "SHA256SUMS").write_text(ledger, encoding="utf-8")
    return metadata


def verify_independent_suite(output_dir: Path) -> dict[str, Any]:
    """Recompute every provenance binding and reject any detached artifact."""
    metadata_path = output_dir / "suite-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("suite") != SUITE_NAME:
        raise ValueError("suite metadata has the wrong suite name")
    if metadata.get("generator_source_sha256") != generator_source_sha256():
        raise ValueError("generator source hash mismatch")
    configuration = metadata.get("configuration")
    if metadata.get("configuration_sha256") != _sha256_bytes(_canonical_json(configuration)):
        raise ValueError("configuration hash mismatch")
    manifest_path = output_dir / str(metadata["manifest"])
    manifest_hash = _sha256_file(manifest_path)
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("manifest hash mismatch")
    image_hashes = metadata.get("input_images_sha256")
    if not isinstance(image_hashes, dict) or not image_hashes:
        raise ValueError("metadata has no input image hashes")
    for name, expected in image_hashes.items():
        if _sha256_file(output_dir / name) != expected:
            raise ValueError(f"input image hash mismatch: {name}")
    if metadata.get("dataset_sha256") != _dataset_sha256(manifest_hash, image_hashes):
        raise ValueError("dataset hash mismatch")
    ledger_lines = (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in ledger_lines:
        expected, name = line.split("  ", 1)
        if _sha256_file(output_dir / name) != expected:
            raise ValueError(f"SHA256SUMS mismatch: {name}")
    return metadata
