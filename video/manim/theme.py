"""Canonical visual constants loaded from the resolved film manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class FilmTheme:
    background: str = "#090909"
    black_secondary: str = "#141414"
    graphite: str = "#222222"
    primary_text: str = "#F1F0EC"
    muted_text: str = "#9A9994"
    rule: str = "#343434"
    prediction: str = "#D65A3A"
    ground_truth: str = "#7EA6C9"
    display_font: str = "DIN Condensed"
    body_font: str = "Arial"
    mono_font: str = "Menlo"


def _resolved_theme() -> FilmTheme:
    locator = os.environ.get("METRALIGN_RESOLVED_FILM")
    if not locator or not Path(locator).is_file():
        return FilmTheme()
    try:
        resolved = json.loads(Path(locator).read_text(encoding="utf-8"))
        colors = resolved["theme"]["colors"]
        fonts = resolved["theme"]["fonts"]
        return FilmTheme(
            background=colors["background"],
            black_secondary=colors["black_secondary"],
            graphite=colors["graphite"],
            primary_text=colors["primary_text"],
            muted_text=colors["muted_text"],
            rule=colors["rule"],
            prediction=colors["prediction"],
            ground_truth=colors["ground_truth"],
            display_font=fonts["display"],
            body_font=fonts["body"],
            mono_font=fonts["mono"],
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return FilmTheme()


THEME = _resolved_theme()
