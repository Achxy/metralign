from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytest.importorskip("manim")

from manim import MathTex

from video.manim.base_scene import FilmScene
from video.manim.components import equation, label, section_header


ROOT = Path(__file__).resolve().parents[2]


def test_equations_use_semantic_latex() -> None:
    rendered = equation(
        r"R_x(x,y)",
        "=",
        r"I(x,y)",
        "-",
        r"\frac{1}{2}",
        color_map={r"R_x": "#67A9CE"},
    )
    assert isinstance(rendered, MathTex)
    assert len(rendered.submobjects) >= 5


def test_labels_preserve_authored_case_and_headers_stay_in_safe_band() -> None:
    authored = label("Synthetic benchmark example")
    assert authored.original_text == "Synthetic benchmark example"

    header = section_header("05", "RETAIN ALTERNATIVES")
    assert header.get_top()[1] <= 3.3
    assert header.get_bottom()[1] >= 2.55
    assert header.get_left()[0] >= -6.45
    assert header.get_right()[0] <= 6.45


def test_transition_default_is_short_and_not_coupled_to_narration_length() -> None:
    signature = inspect.signature(FilmScene.cue)
    assert signature.parameters["run_time"].default is None
    source = inspect.getsource(FilmScene.cue)
    assert "else 0.48" in source
    assert "audio *" not in source


def test_judge_facing_scene_source_excludes_internal_audit_stamps() -> None:
    scene_source = (ROOT / "video/manim/scenes/film.py").read_text(encoding="utf-8")
    for forbidden in (
        "synthetic_case_stamp",
        "SEALED SYNTHETIC",
        "SOURCE-BOUND MAXIMA",
        "000002_dram",
        "DEMI BOLD",
    ):
        assert forbidden not in scene_source


def test_live_capture_uses_presentation_safe_input_names() -> None:
    command = (ROOT / "video/evidence/exported/live_command.txt").read_text(
        encoding="utf-8"
    )
    assert "--reference reference.png" in command
    assert "--search search.png" in command
    assert "000002" not in command
    assert "dram" not in command.lower()
