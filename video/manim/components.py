"""Small factual composition primitives; no decorative UI components."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Cross,
    Group,
    ImageMobject,
    Line,
    Rectangle,
    Text,
    VGroup,
)

from video.manim.grid import GRID
from video.manim.theme import THEME


def text(
    value: str,
    *,
    size: float = 30,
    color: str | None = None,
    font: str | None = None,
    weight: str = "NORMAL",
) -> Text:
    return Text(
        value,
        font=font or THEME.body_font,
        font_size=size,
        color=color or THEME.primary_text,
        weight=weight,
        disable_ligatures=True,
    )


def label(value: str, *, size: float = 18, color: str | None = None) -> Text:
    return text(
        value.upper(),
        size=size,
        color=color or THEME.muted_text,
        font=THEME.display_font,
        weight="BOLD",
    )


def section_header(index: str, title: str) -> VGroup:
    number = label(index, size=16, color=THEME.muted_text)
    heading = text(title.upper(), size=27, font=THEME.display_font, weight="BOLD")
    number.move_to([GRID.left + number.width / 2, GRID.top - 0.16, 0])
    heading.next_to(number, RIGHT, buff=0.3).align_to(number, UP)
    rule = Line(
        [GRID.left, GRID.top - 0.56, 0],
        [GRID.right, GRID.top - 0.56, 0],
        color=THEME.rule,
        stroke_width=1.0,
    )
    return VGroup(number, heading, rule)


def fit_image(path: str | Path, width: float, height: float) -> ImageMobject:
    image = ImageMobject(str(path))
    scale = min(width / image.width, height / image.height)
    image.scale(scale)
    return image


def image_panel(
    path: str | Path,
    *,
    width: float,
    height: float,
    panel_label: str | None = None,
    border: bool = True,
) -> Group:
    image = fit_image(path, width, height)
    objects = [image]
    if border:
        objects.append(
            Rectangle(
                width=image.width,
                height=image.height,
                color=THEME.rule,
                stroke_width=1.0,
                fill_opacity=0.0,
            ).move_to(image)
        )
    if panel_label:
        panel_text = label(panel_label, size=15)
        panel_text.next_to(image, UP, buff=0.12).align_to(image, LEFT)
        objects.append(panel_text)
    return Group(*objects)


def registration_cross(
    point: np.ndarray,
    *,
    color: str,
    size: float = 0.14,
    stroke_width: float = 2.0,
) -> VGroup:
    horizontal = Line(LEFT * size, RIGHT * size, color=color, stroke_width=stroke_width)
    vertical = Line(DOWN * size, UP * size, color=color, stroke_width=stroke_width)
    return VGroup(horizontal, vertical).move_to(point)


def metric_block(value: str, caption: str, *, width: float) -> VGroup:
    number = text(value, size=56, font=THEME.display_font, weight="BOLD")
    copy = label(caption, size=16)
    copy.next_to(number, DOWN, buff=0.12).align_to(number, LEFT)
    rule = Line(LEFT * width / 2, RIGHT * width / 2, color=THEME.rule, stroke_width=1)
    rule.next_to(copy, DOWN, buff=0.18).align_to(number, LEFT)
    return VGroup(number, copy, rule)


def scope_note(value: str, *, width: float = 5.7) -> Text:
    note = text(value, size=17, color=THEME.muted_text)
    note.set_width(min(width, note.width))
    return note
