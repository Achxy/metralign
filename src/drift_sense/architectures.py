"""Continuous procedural wafer geometries with persistent process variation."""

from __future__ import annotations

import numpy as np


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int, channel: float = 0.0) -> np.ndarray:
    """Deterministic hash in [-1, 1], indexed in physical cell coordinates."""
    value = np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7 + channel * 19.19)
    return 2.0 * np.mod(value * 43758.5453123, 1.0) - 1.0


def _smooth_step(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray) -> np.ndarray:
    den = np.maximum(np.asarray(edge1) - np.asarray(edge0), 1e-6)
    t = np.clip((x - edge0) / den, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _global_variation(seed: int, channel: float) -> float:
    return float(_hash2(np.asarray(0), np.asarray(0), seed, channel))


def render_dram_geometry(
    wx: np.ndarray,
    wy: np.ndarray,
    seed: int,
    geometry_variant: str = "default",
    disabled_variations: tuple[str, ...] = (),
) -> np.ndarray:
    """DRAM-like staggered contact array evaluated in world coordinates."""
    disabled = set(disabled_variations)
    variation_scale = 0.0 if geometry_variant == "ideal" else 1.0
    spread = (
        0.0
        if "pitch_variation" in disabled
        else (0.035 if geometry_variant == "ood" else 0.010) * variation_scale
    )
    pitch_x = 13.0 * (1.0 + spread * _global_variation(seed, 21.0))
    pitch_y = 15.0 * (1.0 + spread * _global_variation(seed, 22.0))
    radius_nominal = 3.65 * (
        1.0
        + (0.0 if "width_variation" in disabled else variation_scale)
        * (0.10 if geometry_variant == "ood" else 0.025)
        * _global_variation(seed, 23.0)
    )
    row = np.floor(wy / pitch_y + 0.5).astype(np.int32)
    row_offset = (row & 1) * (0.5 * pitch_x)
    col = np.floor((wx - row_offset) / pitch_x + 0.5).astype(np.int32)

    # Persistent cell-level variation: every capture sees the same center and radius.
    center_scale = 0.0 if "center_variation" in disabled else variation_scale
    width_scale = 0.0 if "width_variation" in disabled else variation_scale
    roughness_scale = 0.0 if "edge_roughness" in disabled else variation_scale
    cx_shift = center_scale * 0.46 * _hash2(col, row, seed, 1.0)
    cy_shift = center_scale * 0.40 * _hash2(col, row, seed, 2.0)
    radius = radius_nominal * (
        1.0 + width_scale * 0.075 * _hash2(col, row, seed, 3.0)
    )
    cx = col * pitch_x + row_offset + cx_shift
    cy = row * pitch_y + cy_shift
    dx, dy = wx - cx, wy - cy
    angle = np.arctan2(dy, dx)
    # Low-amplitude boundary variation stands in for correlated edge roughness.
    rough = roughness_scale * 0.10 * np.sin(
        5.0 * angle + 2.3 * _hash2(col, row, seed, 4.0)
    )
    dist = np.hypot(dx, dy)
    contact = 1.0 - _smooth_step(radius + rough - 0.45, radius + rough + 0.45, dist)

    # Word/bit-line relief provides two reciprocal-lattice directions.
    line_x = 0.12 * (0.5 + 0.5 * np.cos(2.0 * np.pi * wx / pitch_x))
    line_y = 0.09 * (0.5 + 0.5 * np.cos(2.0 * np.pi * wy / pitch_y))
    return np.clip(0.12 + contact * 0.68 + line_x + line_y, 0.0, 1.0)


def render_finfet_geometry(
    wx: np.ndarray,
    wy: np.ndarray,
    seed: int,
    geometry_variant: str = "default",
    disabled_variations: tuple[str, ...] = (),
) -> np.ndarray:
    """FinFET-like orthogonal fin/gate geometry with persistent width variation."""
    disabled = set(disabled_variations)
    variation_scale = 0.0 if geometry_variant == "ideal" else 1.0
    spread = (
        0.0
        if "pitch_variation" in disabled
        else (0.040 if geometry_variant == "ood" else 0.010) * variation_scale
    )
    fin_pitch = 10.0 * (1.0 + spread * _global_variation(seed, 24.0))
    gate_pitch = 18.0 * (1.0 + spread * _global_variation(seed, 25.0))
    fin_width_nominal = 3.15 * (
        1.0
        + (0.0 if "width_variation" in disabled else variation_scale)
        * (0.10 if geometry_variant == "ood" else 0.025)
        * _global_variation(seed, 26.0)
    )
    gate_width_nominal = 5.1 * (
        1.0
        + (0.0 if "width_variation" in disabled else variation_scale)
        * (0.10 if geometry_variant == "ood" else 0.025)
        * _global_variation(seed, 27.0)
    )
    fin_idx = np.floor(wx / fin_pitch + 0.5).astype(np.int32)
    gate_idx = np.floor(wy / gate_pitch + 0.5).astype(np.int32)
    # Width and center vary along each line as well as from line to line. This
    # gives persistent 2-D line-edge/line-width roughness instead of assigning
    # one width to an infinitely long ideal line.
    fin_line = _hash2(fin_idx, np.zeros_like(fin_idx), seed, 7.0)
    gate_line = _hash2(np.zeros_like(gate_idx), gate_idx, seed, 8.0)
    fin_segment = _hash2(fin_idx, gate_idx, seed, 9.0)
    gate_segment = _hash2(fin_idx, gate_idx, seed, 10.0)
    center_scale = 0.0 if "center_variation" in disabled else variation_scale
    width_scale = 0.0 if "width_variation" in disabled else variation_scale
    roughness_scale = 0.0 if "edge_roughness" in disabled else variation_scale
    fin_center = fin_idx * fin_pitch + center_scale * (
        0.18 * fin_line + roughness_scale * 0.20 * fin_segment
    )
    gate_center = gate_idx * gate_pitch + center_scale * (
        0.18 * gate_line + roughness_scale * 0.22 * gate_segment
    )
    fin_width = fin_width_nominal * (
        1.0 + width_scale * (0.045 * fin_line + roughness_scale * 0.055 * fin_segment)
    )
    gate_width = gate_width_nominal * (
        1.0 + width_scale * (0.040 * gate_line + roughness_scale * 0.050 * gate_segment)
    )
    fin = 1.0 - _smooth_step(fin_width / 2 - 0.35, fin_width / 2 + 0.35, np.abs(wx - fin_center))
    gate = 1.0 - _smooth_step(gate_width / 2 - 0.40, gate_width / 2 + 0.40, np.abs(wy - gate_center))
    image = 0.10 + 0.46 * fin + 0.23 * gate + 0.18 * fin * gate
    return np.clip(image, 0.0, 1.0)


def render_architecture(
    name: str,
    wx: np.ndarray,
    wy: np.ndarray,
    seed: int,
    geometry_variant: str = "default",
    disabled_variations: tuple[str, ...] = (),
) -> np.ndarray:
    if name == "dram":
        return render_dram_geometry(wx, wy, seed, geometry_variant, disabled_variations)
    if name == "finfet":
        return render_finfet_geometry(wx, wy, seed, geometry_variant, disabled_variations)
    raise ValueError(f"unsupported architecture: {name!r}")
