"""Canonical visual constants loaded from the resolved film manifest."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class FilmTheme:
    background: str = "#0B0B0E"
    black_secondary: str = "#15151A"
    graphite: str = "#23242A"
    primary_text: str = "#F3F1EB"
    muted_text: str = "#AAA8A2"
    rule: str = "#34353D"
    prediction: str = "#E36A45"
    ground_truth: str = "#67A9CE"
    display_font: str = "Avenir Next"
    body_font: str = "Avenir Next"
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
