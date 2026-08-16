"""Independent SEM capture renderer."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from .architectures import render_architecture
from .distortions import AcquisitionParameters, apply_acquisition
from .geometry import CaptureGeometry, sensor_grid


def _downsample_area(image: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return image.reshape(h, factor, w, factor).mean(axis=(1, 3))


def _downsample_lanczos(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    im = Image.fromarray(np.clip(image * 65535.0, 0, 65535).astype(np.uint16))
    return np.asarray(im.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 65535.0


def render_capture(
    architecture: str,
    wafer_seed: int,
    geometry: CaptureGeometry,
    acquisition: AcquisitionParameters,
    acquisition_seed: int,
    supersample: int = 2,
    resampler: str = "area",
    geometry_variant: str = "default",
    disabled_variations: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    x, y = sensor_grid(geometry.width, geometry.height, supersample)
    wx, wy = geometry.sensor_to_world(x, y)
    latent = render_architecture(
        architecture, wx, wy, wafer_seed, geometry_variant, disabled_variations
    ).astype(np.float32)
    shape = (geometry.height, geometry.width)
    if supersample > 1:
        if resampler == "area":
            latent = _downsample_area(latent, supersample, shape)
        elif resampler == "lanczos":
            latent = _downsample_lanczos(latent, shape)
        else:
            raise ValueError(f"unsupported resampler: {resampler}")
    elif latent.shape != shape:
        latent = ndimage.zoom(latent, (shape[0] / latent.shape[0], shape[1] / latent.shape[1]), order=1)
    return apply_acquisition(latent, acquisition, np.random.default_rng(acquisition_seed))


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
