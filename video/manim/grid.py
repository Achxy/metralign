"""One rigid 12-column layout shared by every film scene."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MasterGrid:
    left: float = -6.42
    right: float = 6.42
    top: float = 3.58
    bottom: float = -3.58
    columns: int = 12
    gutter: float = 0.18

    @property
    def column_width(self) -> float:
        return ((self.right - self.left) - self.gutter * (self.columns - 1)) / self.columns

    def col(self, index: int) -> float:
        if not 0 <= index <= self.columns:
            raise ValueError(f"column index outside 0..{self.columns}: {index}")
        if index == self.columns:
            return self.right
        return self.left + index * (self.column_width + self.gutter)

    def span(self, start: int, end: int) -> tuple[float, float]:
        if end <= start:
            raise ValueError("grid span end must exceed start")
        x0 = self.col(start)
        x1 = self.col(end) - (self.gutter if end < self.columns else 0.0)
        return x0, x1

    def center(self, start: int, end: int, y: float = 0.0) -> np.ndarray:
        x0, x1 = self.span(start, end)
        return np.array([(x0 + x1) / 2.0, y, 0.0])

    def width(self, start: int, end: int) -> float:
        x0, x1 = self.span(start, end)
        return x1 - x0

    def row(self, fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("row fraction must lie within 0..1")
        return self.top + fraction * (self.bottom - self.top)


def _resolved_grid() -> MasterGrid:
    locator = os.environ.get("METRALIGN_RESOLVED_FILM")
    if not locator or not Path(locator).is_file():
        return MasterGrid()
    try:
        resolved = json.loads(Path(locator).read_text(encoding="utf-8"))
        frame = resolved["film"]["frame"]
        grid = resolved["theme"]["grid"]
        pixel_width = float(frame["width"])
        pixel_height = float(frame["height"])
        scene_height = 8.0
        scene_width = scene_height * pixel_width / pixel_height
        horizontal_unit = scene_width / pixel_width
        vertical_unit = scene_height / pixel_height
        margin_px = float(grid["outer_margin_px"])
        return MasterGrid(
            left=-scene_width / 2.0 + margin_px * horizontal_unit,
            right=scene_width / 2.0 - margin_px * horizontal_unit,
            top=scene_height / 2.0 - margin_px * vertical_unit,
            bottom=-scene_height / 2.0 + margin_px * vertical_unit,
            columns=int(grid["columns"]),
            gutter=float(grid["gutter_px"]) * horizontal_unit,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ZeroDivisionError):
        return MasterGrid()


GRID = _resolved_grid()
