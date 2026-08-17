"""Load hash-bound evidence records and raster assets for Manim scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_repo_path(locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else REPO_ROOT / path


def exported_asset(index: dict[str, Any], name: str) -> Path:
    assets = index.get("assets", {})
    entry = assets.get(name) if isinstance(assets, dict) else None
    if isinstance(entry, str):
        return resolve_repo_path(entry)
    if isinstance(entry, dict):
        locator = entry.get("path") or entry.get("file")
        if isinstance(locator, str):
            return resolve_repo_path(locator)
    candidate = VIDEO_ROOT / "evidence" / "exported" / name
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")


def metric_value(resolved: dict[str, Any], name: str) -> Any:
    return resolved["resolved_evidence"]["metrics"][name]["value"]


def sample_value(resolved: dict[str, Any], name: str) -> dict[str, Any]:
    value = resolved["resolved_evidence"]["samples"][name]["value"]
    if not isinstance(value, dict):
        raise ValueError(f"resolved sample {name} is not a mapping")
    return value

