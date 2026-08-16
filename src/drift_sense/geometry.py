"""Coordinate transforms shared by rendering and ground-truth generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class CaptureGeometry:
    """Map output sensor coordinates to continuous wafer/world coordinates.

    ``pixel_size`` is expressed in world units per output pixel.  Drift is a
    smooth scan-dependent x displacement in sensor pixels and is zero at the
    image center, which makes the optical center an invariant reference point.
    """

    width: int
    height: int
    world_center_x: float
    world_center_y: float
    pixel_size: float
    rotation_deg: float = 0.0
    anisotropy: float = 0.0
    drift_linear: float = 0.0
    drift_quadratic: float = 0.0

    @property
    def cx(self) -> float:
        return (self.width - 1.0) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1.0) / 2.0

    def drift_x(self, centered_y: np.ndarray | float) -> np.ndarray | float:
        yn = np.asarray(centered_y) / max(self.height / 2.0, 1.0)
        return self.drift_linear * yn + self.drift_quadratic * yn * np.abs(yn)

    def sensor_to_world(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(y, dtype=np.float64) - self.cy
        u = np.asarray(x, dtype=np.float64) - self.cx + self.drift_x(v)
        sx = self.pixel_size * (1.0 + self.anisotropy)
        sy = self.pixel_size * (1.0 - self.anisotropy)
        theta = np.deg2rad(self.rotation_deg)
        ct, st = np.cos(theta), np.sin(theta)
        wx = self.world_center_x + ct * (u * sx) - st * (v * sy)
        wy = self.world_center_y + st * (u * sx) + ct * (v * sy)
        return wx, wy

    def world_to_sensor(self, wx: float, wy: float) -> tuple[float, float]:
        """Analytic inverse of :meth:`sensor_to_world`."""
        theta = np.deg2rad(self.rotation_deg)
        ct, st = np.cos(theta), np.sin(theta)
        dx, dy = wx - self.world_center_x, wy - self.world_center_y
        local_x = ct * dx + st * dy
        local_y = -st * dx + ct * dy
        sx = self.pixel_size * (1.0 + self.anisotropy)
        sy = self.pixel_size * (1.0 - self.anisotropy)
        v = local_y / sy
        u = local_x / sx - self.drift_x(v)
        return float(u + self.cx), float(v + self.cy)

    def to_dict(self) -> dict:
        return asdict(self)


def sensor_grid(width: int, height: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-center grid for supersampled area integration."""
    ss = int(supersample)
    x = (np.arange(width * ss, dtype=np.float32) + 0.5) / ss - 0.5
    y = (np.arange(height * ss, dtype=np.float32) + 0.5) / ss - 0.5
    return np.meshgrid(x, y)
