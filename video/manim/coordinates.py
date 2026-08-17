"""Exact pixel-to-scene coordinate mapping for evidence overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageCoordinateMap:
    pixel_width: int
    pixel_height: int
    scene_left: float
    scene_right: float
    scene_bottom: float
    scene_top: float

    def point(self, x_pixel: float, y_pixel: float) -> np.ndarray:
        if self.pixel_width <= 1 or self.pixel_height <= 1:
            raise ValueError("pixel dimensions must exceed one")
        x = self.scene_left + x_pixel / (self.pixel_width - 1) * (
            self.scene_right - self.scene_left
        )
        y = self.scene_top - y_pixel / (self.pixel_height - 1) * (
            self.scene_top - self.scene_bottom
        )
        return np.array([x, y, 0.0])

    @classmethod
    def from_mobject(cls, image, pixel_width: int, pixel_height: int) -> "ImageCoordinateMap":
        return cls(
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            scene_left=float(image.get_left()[0]),
            scene_right=float(image.get_right()[0]),
            scene_bottom=float(image.get_bottom()[1]),
            scene_top=float(image.get_top()[1]),
        )

