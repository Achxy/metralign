"""Source-bound vector graphics used by the Metralign film."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    Axes,
    Dot,
    Line,
    Rectangle,
    VGroup,
)

from video.manim.components import label, text
from video.manim.evidence import REPO_ROOT
from video.manim.theme import THEME


def frozen_errors() -> list[float]:
    """Read every archived error directly from the seven sealed reports."""

    errors: list[float] = []
    for report_path in sorted((REPO_ROOT / "results" / "frozen" / "reports").glob("*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        errors.extend(float(sample["error"]) for sample in report["methods"]["full"]["samples"])
    if len(errors) != 1400:
        raise ValueError(f"sealed report population changed: {len(errors)}")
    return errors


def suite_dot_rows(suites: list[dict], *, width: float = 10.9) -> VGroup:
    """Render all 1,400 outcomes as seven rows of 200 source-bound dots."""

    rows = VGroup()
    dot_span = width - 3.65
    display_names = {
        "iid": "IID",
        "high_noise": "High noise",
        "geometry_ood": "Geometry OOD",
        "transform_ood": "Transform OOD",
        "periodic_ambiguity": "Periodic ambiguity",
        "scan_distortion": "Scan distortion",
        "cross_generator": "Alternate renderer",
    }
    for record in suites:
        count = int(record["pair_count"])
        success = int(record["within_1px_count"])
        suite_id = str(record["suite"])
        name = display_names.get(suite_id, suite_id.replace("_", " "))
        row_label = text(name, size=21)
        dots = VGroup(
            *[
                Dot(
                    radius=0.015,
                    color=THEME.ground_truth if index < success else THEME.prediction,
                    fill_opacity=0.95,
                    stroke_width=0,
                )
                for index in range(count)
            ]
        ).arrange(RIGHT, buff=max(0.004, (dot_span - 2 * count * 0.015) / max(count - 1, 1)))
        dots.stretch_to_fit_width(dot_span)
        count_label = text(
            f"{success}/{count}",
            size=20,
            color=THEME.ground_truth if success == count else THEME.prediction,
            weight="MEDIUM",
        )
        row = VGroup(row_label, dots, count_label)
        row_label.move_to(LEFT * (width / 2 - row_label.width / 2))
        dots.next_to(row_label, RIGHT, buff=0.35)
        count_label.next_to(dots, RIGHT, buff=0.22)
        rows.add(row)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=0.38)
    return rows


def empirical_cdf(*, width: float = 9.6, height: float = 4.5) -> tuple[VGroup, Axes]:
    """Create a vector CDF for the sealed errors, with the >1 px tail explicit."""

    errors = np.sort(np.asarray(frozen_errors(), dtype=float))
    in_range = errors[errors <= 1.0]
    y = np.arange(1, len(in_range) + 1, dtype=float) / len(errors) * 100.0
    axes = Axes(
        x_range=[0, 1.0, 0.2],
        y_range=[0, 100, 25],
        x_length=width,
        y_length=height,
        tips=False,
        axis_config={"color": THEME.rule, "stroke_width": 1.2},
    )
    points = [axes.c2p(float(xv), float(yv)) for xv, yv in zip(in_range, y)]
    curve = VGroup(
        *[
            Line(a, b, color=THEME.ground_truth, stroke_width=3.2)
            for a, b in zip(points[:-1], points[1:])
        ]
    )
    ticks = VGroup()
    for value in [0.0, 0.5, 1.0]:
        mark = label(f"{value:g} px", size=16)
        mark.next_to(axes.c2p(value, 0), DOWN, buff=0.12)
        ticks.add(mark)
    for value in [0, 50, 100]:
        mark = label(f"{value}%", size=16)
        mark.next_to(axes.c2p(0, value), LEFT, buff=0.12)
        ticks.add(mark)
    return VGroup(axes, curve, ticks), axes


def comparison_bars(
    classic_records: list[dict],
    *,
    metralign_rate: float,
    width: float = 11.0,
) -> tuple[VGroup, VGroup, VGroup]:
    """Build direct-labelled vector bars; audit-table typography is intentionally absent."""

    left_x = -width / 2 + 3.25
    bar_width = width - 5.15
    metralign = _comparison_row(
        "Metralign",
        metralign_rate,
        left_x=left_x,
        bar_width=bar_width,
        color=THEME.ground_truth,
    )
    best_record = next(record for record in classic_records if record["is_best"])
    best = _comparison_row(
        f"Best of six · {best_record['display_name']}",
        float(best_record["within_1px_rate_display"]),
        left_x=left_x,
        bar_width=bar_width,
        color=THEME.prediction,
    )
    others = VGroup(
        *[
            _comparison_row(
                str(record["display_name"]),
                float(record["within_1px_rate_display"]),
                left_x=left_x,
                bar_width=bar_width,
                color=THEME.muted_text,
                compact=True,
            )
            for record in classic_records
            if not record["is_best"]
        ]
    ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    return metralign, best, others


def _comparison_row(
    name: str,
    rate: float,
    *,
    left_x: float,
    bar_width: float,
    color: str,
    compact: bool = False,
) -> VGroup:
    name_label = text(name, size=18 if compact else 23, color=THEME.primary_text)
    name_label.move_to([left_x - 0.25 - name_label.width / 2, 0, 0])
    track = Line(
        [left_x, 0, 0],
        [left_x + bar_width, 0, 0],
        color=THEME.rule,
        stroke_width=5 if compact else 8,
    )
    value_width = max(0.012, bar_width * rate / 100.0)
    value = Rectangle(
        width=value_width,
        height=0.08 if compact else 0.13,
        fill_color=color,
        fill_opacity=1.0,
        stroke_width=0,
    ).move_to([left_x + value_width / 2, 0, 0])
    number = text(
        f"{rate:.2f}%",
        size=18 if compact else 24,
        color=color,
        weight="MEDIUM",
    )
    number.move_to([left_x + bar_width + 0.35 + number.width / 2, 0, 0])
    return VGroup(name_label, track, value, number)
