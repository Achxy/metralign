"""Precise composition primitives for the Metralign explanatory film."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Group,
    ImageMobject,
    Line,
    MathTex,
    Rectangle,
    TexTemplate,
    Text,
    VGroup,
)

from video.manim.grid import GRID
from video.manim.theme import THEME


_HOMEBREW_TEXLIVE_SHARE = Path("/opt/homebrew/opt/texlive/share")
if _HOMEBREW_TEXLIVE_SHARE.is_dir():
    # Homebrew packages dvisvgm separately from TeX Live, so its compiled
    # kpathsea prefix does not discover the TeX tree without these roots.
    os.environ.setdefault(
        "TEXMFCNF", str(_HOMEBREW_TEXLIVE_SHARE / "texmf-dist" / "web2c")
    )
    os.environ.setdefault("TEXMFROOT", str(_HOMEBREW_TEXLIVE_SHARE))
    os.environ.setdefault("TEXMFDIST", str(_HOMEBREW_TEXLIVE_SHARE / "texmf-dist"))
    os.environ.setdefault("TEXMFLOCAL", str(_HOMEBREW_TEXLIVE_SHARE / "texmf-local"))


# Native LaTeX + dvisvgm preserve Manim's semantic term identifiers, which are
# required for token-aware equation coloring and TransformMatchingTex.
MATH_TEMPLATE = TexTemplate(tex_compiler="latex", output_format=".dvi")
MATH_TEMPLATE.add_to_preamble(r"\usepackage{amsmath,amssymb,bm}")


def text(
    value: str,
    *,
    size: float = 30,
    color: str | None = None,
    font: str | None = None,
    weight: str = "NORMAL",
    line_spacing: float = -1,
) -> Text:
    """Render ordinary language with the film's humanist text face."""

    return Text(
        value,
        font=font or THEME.body_font,
        font_size=size,
        color=color or THEME.primary_text,
        weight=weight,
        line_spacing=line_spacing,
        disable_ligatures=False,
    )


def label(value: str, *, size: float = 18, color: str | None = None) -> Text:
    """Render a restrained scope or data label."""

    return text(
        value,
        size=size,
        color=color or THEME.muted_text,
        weight="MEDIUM",
    )


def equation(
    *parts: str,
    size: float = 54,
    color: str | None = None,
    isolate: list[str] | None = None,
    color_map: dict[str, str] | None = None,
) -> MathTex:
    """Render source notation as actual LaTeX, split for semantic transforms."""

    result = MathTex(
        *parts,
        font_size=size,
        color=color or THEME.primary_text,
        tex_template=MATH_TEMPLATE,
        substrings_to_isolate=isolate,
    )
    if color_map:
        result.set_color_by_tex_to_color_map(color_map)
    return result


def section_header(index: str, title: str) -> VGroup:
    """A collision-proof chapter marker with a protected title band."""

    number = label(index, size=15, color=THEME.ground_truth)
    heading = text(title.title(), size=35, weight="SEMIBOLD")
    number.move_to([GRID.left + number.width / 2, GRID.top - 0.30, 0])
    heading.next_to(number, RIGHT, buff=0.28).align_to(number, DOWN)
    rule = Line(
        [heading.get_left()[0], GRID.top - 0.57, 0],
        [min(heading.get_right()[0] + 0.45, GRID.right), GRID.top - 0.57, 0],
        color=THEME.rule,
        stroke_width=1.2,
    )
    return VGroup(number, heading, rule)


def fit_image(path: str | Path, width: float, height: float) -> ImageMobject:
    image = ImageMobject(str(path))
    image.scale(min(width / image.width, height / image.height))
    return image


def crop_image(
    path: str | Path,
    box: tuple[float, float, float, float],
    *,
    width: float,
    height: float,
) -> ImageMobject:
    """Display an authentic normalized crop without writing a derived asset."""

    left, top, right, bottom = box
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError(f"normalized crop must lie inside the source image: {box}")
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"))
    y0, y1 = round(top * array.shape[0]), round(bottom * array.shape[0])
    x0, x1 = round(left * array.shape[1]), round(right * array.shape[1])
    image = ImageMobject(array[y0:y1, x0:x1])
    image.scale(min(width / image.width, height / image.height))
    return image


def image_panel(
    path: str | Path,
    *,
    width: float,
    height: float,
    panel_label: str | None = None,
    border: bool = True,
) -> Group:
    return framed_image(
        fit_image(path, width, height), panel_label=panel_label, border=border
    )


def framed_image(
    image: ImageMobject,
    *,
    panel_label: str | None = None,
    border: bool = True,
    label_size: float = 18,
) -> Group:
    """Attach a label without allowing it to invade the protected title band."""

    objects: list = [image]
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
        panel_text = label(panel_label, size=label_size)
        panel_text.next_to(image, UP, buff=0.16).align_to(image, LEFT)
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


def metric_statement(value: str, caption: str, *, accent: str | None = None) -> VGroup:
    """A direct-labelled number, intentionally free of card chrome."""

    number = text(value, size=64, color=accent, weight="SEMIBOLD")
    copy = text(caption, size=22, color=THEME.muted_text)
    copy.next_to(number, DOWN, buff=0.12).align_to(number, LEFT)
    return VGroup(number, copy)


def metric_block(value: str, caption: str, *, width: float) -> VGroup:
    """Compatibility wrapper retained while scenes migrate to direct labels."""

    group = metric_statement(value, caption)
    rule = Line(LEFT * width / 2, RIGHT * width / 2, color=THEME.rule, stroke_width=1)
    rule.next_to(group, DOWN, buff=0.18).align_to(group, LEFT)
    group.add(rule)
    return group


def scope_note(value: str, *, width: float = 5.7) -> Text:
    note = text(value, size=19, color=THEME.muted_text)
    if note.width > width:
        note.scale_to_fit_width(width)
    return note


def fit_to_safe_frame(mobject, *, buffer: float = 0.08):
    """Scale a composed object only when it exceeds the explicit safe frame."""

    max_width = GRID.right - GRID.left - 2 * buffer
    max_height = GRID.top - GRID.bottom - 2 * buffer
    if mobject.width > max_width:
        mobject.scale_to_fit_width(max_width)
    if mobject.height > max_height:
        mobject.scale_to_fit_height(max_height)
    return mobject
