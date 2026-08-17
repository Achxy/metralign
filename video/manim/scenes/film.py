"""Twelve evidence-bound explanatory scenes for the Metralign film."""

from __future__ import annotations

import csv
import math

import numpy as np
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Arrow,
    Axes,
    Circle,
    Circumscribe,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    Group,
    GrowFromCenter,
    Indicate,
    LaggedStart,
    Line,
    ManimColor,
    Rectangle,
    ReplacementTransform,
    Restore,
    ShowIncreasingSubsets,
    Square,
    Transform,
    TransformFromCopy,
    TransformMatchingTex,
    VGroup,
    Write,
    interpolate_color,
)

from drift_sense.candidates import top_k_candidates
from video.manim.base_scene import FilmScene
from video.manim.charts import comparison_bars, empirical_cdf, suite_dot_rows
from video.manim.components import (
    crop_image,
    equation,
    fit_image,
    framed_image,
    image_panel,
    label,
    metric_statement,
    registration_cross,
    scope_note,
    section_header,
    text,
)
from video.manim.coordinates import ImageCoordinateMap
from video.manim.evidence import load_json
from video.manim.grid import GRID
from video.manim.theme import THEME


BODY_TOP = 2.42
BODY_BOTTOM = -2.55
BODY_CENTER = (BODY_TOP + BODY_BOTTOM) / 2
BODY_HEIGHT = BODY_TOP - BODY_BOTTOM


def _float(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _xy(sample: dict, key: str) -> tuple[float, float]:
    value = sample[key]
    return float(value[0]), float(value[1])


def _panel_image(panel):
    return panel[0]


def image_mapper(image) -> ImageCoordinateMap:
    height, width = image.pixel_array.shape[:2]
    return ImageCoordinateMap.from_mobject(image, width, height)


def phase_drift_panel(reference_path, search_path) -> VGroup:
    """Vector plot of the measured selected-Y phase traces."""

    def rows(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                {
                    "x": float(row["scan_coordinate_px"]),
                    "phase": float(row["unwrapped_phase_rad"]),
                    "fit": float(row["fitted_phase_rad"]),
                    "used": row["fit_used"] == "1",
                }
                for row in csv.DictReader(handle)
            ]

    reference = rows(reference_path)
    search = rows(search_path)
    all_rows = reference + search
    x_min = min(row["x"] for row in all_rows)
    x_max = max(row["x"] for row in all_rows)
    phases = [row["phase"] for row in all_rows if row["used"]]
    y_min = math.floor(min(phases) * 2.0) / 2.0
    y_max = math.ceil(max(phases) * 2.0) / 2.0
    axes = Axes(
        x_range=[x_min, x_max, x_max - x_min],
        y_range=[y_min, y_max, 0.5],
        x_length=6.15,
        y_length=3.65,
        tips=False,
        axis_config={"color": THEME.rule, "stroke_width": 1.2},
    )

    def dots(source, color):
        used = [row for row in source if row["used"]]
        stride = max(1, len(used) // 150)
        return VGroup(
            *[
                Dot(
                    axes.c2p(row["x"], row["phase"]),
                    radius=0.014,
                    color=color,
                    fill_opacity=0.5,
                    stroke_width=0,
                )
                for row in used[::stride]
            ]
        )

    def fit(source, color):
        return Line(
            axes.c2p(source[0]["x"], source[0]["fit"]),
            axes.c2p(source[-1]["x"], source[-1]["fit"]),
            color=color,
            stroke_width=3,
        )

    title = text("Measured Y-carrier phase drift", size=24, weight="SEMIBOLD")
    title.next_to(axes, UP, buff=0.2).align_to(axes, LEFT)
    x_label = label("Scan coordinate / px", size=16)
    x_label.next_to(axes, DOWN, buff=0.12)
    legend = VGroup(
        Dot(radius=0.035, color=THEME.ground_truth),
        label("reference", size=15, color=THEME.ground_truth),
        Dot(radius=0.035, color=THEME.prediction),
        label("search", size=15, color=THEME.prediction),
    ).arrange(RIGHT, buff=0.14)
    legend.next_to(axes, UP, buff=0.22).align_to(axes, RIGHT)
    return VGroup(
        axes,
        dots(reference, THEME.ground_truth),
        dots(search, THEME.prediction),
        fit(reference, THEME.ground_truth),
        fit(search, THEME.prediction),
        title,
        x_label,
        legend,
    )


class OpenScene(FilmScene):
    scene_id = "open"

    def construct(self) -> None:
        project = self.resolved["project"]
        sample = self.sample("success_iid")
        search = fit_image(self.asset("frozen_success_search"), 7.1, 7.1)
        search.move_to([3.62, 0.05, 0]).set_opacity(0.56)
        mapper = image_mapper(search)
        gt = _xy(sample, "ground_truth")
        scale = float(sample["diagnostics"]["selected_scale"])
        roi = Rectangle(
            width=search.width * scale,
            height=search.height * scale,
            color=THEME.ground_truth,
            stroke_width=2.6,
        ).move_to(mapper.point(*gt))
        reference = fit_image(self.asset("frozen_success_reference"), 2.45, 2.45)
        reference.move_to([-4.83, -1.05, 0])
        reference_border = Rectangle(
            width=reference.width,
            height=reference.height,
            color=THEME.ground_truth,
            stroke_width=2.0,
        ).move_to(reference)
        mark = fit_image(self.asset("metralign_mark"), 0.72, 0.72)
        mark.move_to([-5.68, 2.92, 0])
        title = text(project["name"], size=68, weight="SEMIBOLD")
        title.move_to([-4.48, 2.05, 0]).align_to(reference, LEFT)
        subtitle = text(
            "Absolute-site localization\nunder periodic ambiguity",
            size=30,
            color=THEME.primary_text,
            line_spacing=0.92,
        )
        subtitle.next_to(title, DOWN, buff=0.34).align_to(title, LEFT)
        scope = text(
            "Synthetic benchmark example",
            size=18,
            color=THEME.muted_text,
        )
        scope.next_to(reference, DOWN, buff=0.28).align_to(reference, LEFT)
        event = label(
            f"{project['event']} · {project['submission_context']}",
            size=15,
        )
        event.move_to([GRID.left + event.width / 2, -3.05, 0])

        self.cue_sequence(
            "000_open_a",
            [
                ([FadeIn(search), FadeIn(reference), Create(reference_border)], 0.45),
                ([TransformFromCopy(reference_border, roi)], 0.75),
                ([FadeIn(mark), Write(title)], 0.65),
            ],
        )
        self.cue("000_open_b", [FadeIn(subtitle), FadeIn(scope), FadeIn(event)], run_time=0.35)
        self.pad_to_resolved_duration()


class ProblemScene(FilmScene):
    scene_id = "problem"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("01", self.scene_record["title"]))
        reference = fit_image(self.asset("frozen_success_reference"), 2.75, 2.75)
        search = fit_image(self.asset("frozen_success_search"), 5.05, 5.05)
        reference.move_to([-4.55, 0.20, 0])
        search.move_to([2.55, -0.05, 0])
        reference_caption = text("Fine reference field", size=22)
        reference_caption.next_to(reference, DOWN, buff=0.18).align_to(reference, LEFT)
        search_caption = text("Wider periodic search", size=22)
        search_caption.next_to(search, DOWN, buff=0.18).align_to(search, LEFT)
        mapper = image_mapper(search)
        candidates = load_json(self.exported("candidates.json"))["candidates"][:14]
        markers = VGroup(
            *[
                Circle(
                    radius=0.055,
                    color=THEME.muted_text,
                    stroke_width=1.35,
                    fill_opacity=0,
                ).move_to(mapper.point(*record["center_xy"]))
                for record in candidates
            ]
        )
        gt = _xy(sample, "ground_truth")
        pred = _xy(sample, "prediction")
        gt_cross = registration_cross(mapper.point(*gt), color=THEME.ground_truth, size=0.11)
        pred_cross = registration_cross(mapper.point(*pred), color=THEME.prediction, size=0.16)
        answer = VGroup(
            text("One absolute coordinate", size=30, weight="SEMIBOLD"),
            text(
                f"({_float(pred[0])}, {_float(pred[1])}) px",
                size=26,
                color=THEME.prediction,
                font=THEME.mono_font,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        answer.move_to([-4.55, -2.10, 0]).align_to(reference, LEFT)
        arrow = Arrow(reference.get_right(), search.get_left(), color=THEME.ground_truth, buff=0.28)

        self.cue("010_problem_a", [FadeIn(reference), FadeIn(search), FadeIn(reference_caption), FadeIn(search_caption), Create(arrow)])
        self.cue(
            "010_problem_b",
            [LaggedStart(*[GrowFromCenter(marker) for marker in markers], lag_ratio=0.07)],
            run_time=0.75,
        )
        self.cue("010_problem_c", [FadeIn(gt_cross), FadeIn(pred_cross), FadeIn(answer)], run_time=0.45)
        self.pad_to_resolved_duration()


class BaselineScene(FilmScene):
    scene_id = "baseline"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("02", self.scene_record["title"]))
        search = fit_image(self.asset("frozen_success_search"), 5.0, 5.0)
        score = fit_image(self.exported("baseline_score_map.png"), 5.0, 5.0)
        search.move_to([-3.22, -0.08, 0])
        score.move_to(search)
        map_array = np.load(self.exported("baseline_score_map.npy"), allow_pickle=False)
        maxima = top_k_candidates(map_array, k=16, nms_radius=13)
        mapper = ImageCoordinateMap.from_mobject(score, map_array.shape[1], map_array.shape[0])
        rings = VGroup(
            *[
                Circle(
                    radius=0.055 if index else 0.085,
                    color=THEME.ground_truth if index == 0 else THEME.muted_text,
                    stroke_width=2.0 if index == 0 else 1.2,
                ).move_to(mapper.point(record.x, record.y))
                for index, record in enumerate(maxima)
            ]
        )
        explanation = VGroup(
            text("Many local answers", size=40, weight="SEMIBOLD"),
            text("The correlation map repeats\nwith the lattice phase.", size=28, line_spacing=0.9),
            Line(LEFT * 2.2, RIGHT * 2.2, color=THEME.rule, stroke_width=1.2),
            text(
                "Local similarity can identify\na phase family — not the site.",
                size=25,
                color=THEME.muted_text,
                line_spacing=0.9,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.27)
        explanation.move_to([3.65, -0.05, 0]).align_to([1.6, 0, 0], LEFT)
        map_caption = label("Normalized template correlation", size=17)
        map_caption.next_to(score, DOWN, buff=0.18).align_to(score, LEFT)

        self.cue_sequence(
            "020_baseline_a",
            [
                ([FadeIn(search)], 0.35),
                ([FadeTransform(search, score), FadeIn(map_caption)], 0.65),
                ([LaggedStart(*[GrowFromCenter(ring) for ring in rings], lag_ratio=0.05)], 0.85),
            ],
        )
        self.cue("020_baseline_b", [FadeIn(explanation)], run_time=0.45)
        self.pad_to_resolved_duration()


class CalibrationScene(FilmScene):
    scene_id = "calibration"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("03", self.scene_record["title"]))
        source = fit_image(self.asset("frozen_success_search"), 5.15, 5.15)
        fft = fit_image(self.exported("fft_search.png"), 5.15, 5.15)
        source.move_to([-3.25, -0.08, 0])
        fft.move_to(source)
        phase = phase_drift_panel(
            self.exported("phase_drift_reference_y.csv"),
            self.exported("phase_drift_search_y.csv"),
        )
        phase.scale(0.79).move_to([-2.58, -0.08, 0])
        transform = load_json(self.exported("phase_transform.json"))
        pitch_x, pitch_y = transform["clipped_pitch_xy_px"]
        estimate = transform["estimate"]
        values = VGroup(
            metric_statement(f"{estimate['scale']:.6f}", "sampling ratio", accent=THEME.ground_truth),
            metric_statement(f"{estimate['rotation_deg']:.3f}°", "bounded rotation", accent=THEME.prediction),
            text(
                f"real-space pitch  {pitch_x:.3f} × {pitch_y:.3f} px",
                size=19,
                color=THEME.primary_text,
            ),
            text(
                f"spectral confidence  {estimate['confidence']:.3f}",
                size=20,
                color=THEME.muted_text,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        values[0].scale(0.72)
        values[1].scale(0.72)
        values.move_to([3.85, -0.1, 0]).align_to([1.48, 0, 0], LEFT)
        transform_arrow = Arrow([-0.42, 0, 0], [1.35, 0, 0], color=THEME.ground_truth, buff=0.15)

        self.cue_sequence(
            "030_phase_a",
            [
                ([FadeIn(source)], 0.35),
                ([FadeTransform(source, fft)], 0.75),
                ([Circumscribe(fft, color=THEME.ground_truth, fade_out=True)], 0.65),
            ],
        )
        self.cue_sequence(
            "030_phase_b",
            [
                ([FadeOut(fft), FadeIn(phase), Create(transform_arrow)], 0.55),
                ([FadeIn(values[0]), FadeIn(values[1])], 0.45),
                ([FadeIn(values[2:])], 0.35),
            ],
        )
        self.cue("030_phase_c", [Indicate(values, color=THEME.ground_truth)], run_time=0.7)
        self.pad_to_resolved_duration()


class DifferenceScene(FilmScene):
    scene_id = "difference"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("04", self.scene_record["title"]))
        formula_x = equation(
            r"R_x(x,y)",
            "=",
            r"I(x,y)",
            "-",
            r"\frac{1}{2}\left[",
            r"I(x-p_x,y)",
            "+",
            r"I(x+p_x,y)",
            r"\right]",
            size=48,
            color_map={r"R_x": THEME.ground_truth, r"p_x": THEME.prediction},
        )
        formula_y = equation(
            r"R_y(x,y)",
            "=",
            r"I(x,y)",
            "-",
            r"\frac{1}{2}\left[",
            r"I(x,y-p_y)",
            "+",
            r"I(x,y+p_y)",
            r"\right]",
            size=48,
            color_map={r"R_y": THEME.ground_truth, r"p_y": THEME.prediction},
        )
        formula_x.move_to([0, 1.92, 0])
        formula_y.move_to(formula_x)
        original = fit_image(self.exported("matched_reference_template.png"), 3.15, 2.75)
        ref_x = fit_image(self.exported("period_difference_reference.png"), 3.15, 2.75)
        search_x = fit_image(self.exported("period_difference_search_crop.png"), 3.15, 2.75)
        original.move_to([-4.25, -0.42, 0])
        ref_x.move_to([0, -0.42, 0])
        search_x.move_to([4.25, -0.42, 0])
        captions = VGroup(
            text("Matched reference", size=21),
            text("Reference residual", size=21, color=THEME.ground_truth),
            text("Search residual", size=21, color=THEME.prediction),
        )
        for caption, image in zip(captions, [original, ref_x, search_x]):
            caption.next_to(image, DOWN, buff=0.15).align_to(image, LEFT)
        self.cue_sequence(
            "040_difference_a",
            [
                ([FadeIn(original), FadeIn(captions[0])], 0.35),
                ([Write(formula_x)], 0.9),
                ([FadeIn(ref_x, shift=RIGHT * 0.18), FadeIn(captions[1])], 0.6),
            ],
        )
        self.cue_sequence(
            "040_difference_b",
            [
                ([TransformMatchingTex(
                    formula_x,
                    formula_y,
                    key_map={r"R_x": r"R_y", r"p_x": r"p_y"},
                )], 0.9),
                ([FadeIn(search_x, shift=RIGHT * 0.18), FadeIn(captions[2])], 0.55),
                ([Circumscribe(Group(ref_x, search_x), color=THEME.prediction)], 0.65),
            ],
        )
        self.cue("040_difference_c", [])
        self.pad_to_resolved_duration()


class CandidateScene(FilmScene):
    scene_id = "candidates"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("05", self.scene_record["title"]))
        search = fit_image(self.asset("frozen_success_search"), 5.0, 5.0)
        search.move_to([-3.35, -0.08, 0])
        mapper = image_mapper(search)
        records = load_json(self.exported("candidates.json"))["candidates"][:16]
        rings = VGroup()
        rank_labels = VGroup()
        for record in records:
            point = mapper.point(*record["center_xy"])
            selected = bool(record["selected"])
            ring = Circle(
                radius=0.075 if selected else 0.045,
                color=THEME.prediction if selected else THEME.muted_text,
                stroke_width=2.4 if selected else 1.1,
            ).move_to(point)
            rings.add(ring)
            if int(record["rank"]) <= 3:
                rank = label(str(record["rank"]), size=14, color=ring.color)
                rank.next_to(ring, UP, buff=0.04)
                rank_labels.add(rank)
        diagnostics = sample["diagnostics"]
        score_delta = float(diagnostics["score"]) - float(diagnostics["runner_up_score"])
        statements = VGroup(
            text("Supported peaks", size=38, weight="SEMIBOLD"),
            text(
                f"best score  {diagnostics['score']:.6f}",
                size=24,
                color=THEME.ground_truth,
                font=THEME.mono_font,
            ),
            text(
                f"margin      {score_delta:.6f}",
                size=24,
                color=THEME.primary_text,
                font=THEME.mono_font,
            ),
            text(
                f"residual support  {diagnostics['ambiguity_evidence']['residual_evidence']:.3f}",
                size=23,
                font=THEME.mono_font,
            ),
            Line(LEFT * 2.2, RIGHT * 2.2, color=THEME.rule, stroke_width=1.1),
            text(
                "A center or stage prior may break a tie\nonly when residual support is low\nor the transform is unstable.",
                size=23,
                color=THEME.muted_text,
                line_spacing=0.85,
            ),
            label(
                "No ambiguity flag" if not diagnostics["ambiguity_flag"] else "Ambiguity flagged",
                size=17,
                color=THEME.ground_truth,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        statements.move_to([3.75, -0.12, 0]).align_to([1.15, 0, 0], LEFT)
        image_caption = text("Candidate locations", size=18, color=THEME.muted_text)
        image_caption.next_to(search, DOWN, buff=0.16).align_to(search, LEFT)

        self.cue_sequence(
            "050_candidates_a",
            [
                ([FadeIn(search), FadeIn(image_caption)], 0.35),
                ([LaggedStart(*[GrowFromCenter(ring) for ring in rings], lag_ratio=0.05)], 0.85),
                ([FadeIn(rank_labels), FadeIn(statements[:4])], 0.45),
            ],
        )
        self.cue("050_candidates_b", [FadeIn(statements[4:6])], run_time=0.4)
        self.cue("050_candidates_c", [FadeIn(statements[6]), Indicate(rings[0], color=THEME.prediction)], run_time=0.65)
        self.pad_to_resolved_duration()


class RefinementScene(FilmScene):
    scene_id = "refinement"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("06", self.scene_record["title"]))
        refinement = load_json(self.exported("refinement_patch.json"))
        scores = np.asarray(refinement["score_values"], dtype=float)
        low_color = ManimColor(THEME.graphite)
        high_color = ManimColor(THEME.ground_truth)
        cells = VGroup()
        for row in range(scores.shape[0]):
            for column in range(scores.shape[1]):
                normalized = (scores[row, column] - scores.min()) / (scores.max() - scores.min())
                color = interpolate_color(low_color, high_color, float(normalized))
                cell = Square(
                    side_length=0.78,
                    fill_color=color,
                    fill_opacity=0.96,
                    stroke_color=THEME.background,
                    stroke_width=1.8,
                )
                cell.move_to([(column - 2) * 0.78, (2 - row) * 0.78, 0])
                cells.add(cell)
        cells.move_to([-2.75, -0.05, 0])
        frame = Rectangle(
            width=cells.width,
            height=cells.height,
            color=THEME.rule,
            stroke_width=1.4,
        ).move_to(cells)
        peak_origin = cells.get_center()
        coarse = refinement["integer_peak_top_left_xy"]
        refined = refinement["refined_top_left_xy"]
        delta = np.array([float(refined[0]) - float(coarse[0]), -(float(refined[1]) - float(coarse[1])), 0]) * 0.78
        coarse_mark = registration_cross(peak_origin, color=THEME.primary_text, size=0.24, stroke_width=2.4)
        refined_mark = registration_cross(peak_origin + delta, color=THEME.prediction, size=0.24, stroke_width=2.8)
        trajectory = Arrow(
            peak_origin,
            peak_origin + delta,
            color=THEME.prediction,
            stroke_width=3,
            buff=0.02,
            max_tip_length_to_length_ratio=0.35,
        )
        pred = _xy(sample, "prediction")
        gt = _xy(sample, "ground_truth")
        details = VGroup(
            text("Parabolic fit", size=40, weight="SEMIBOLD"),
            equation(r"\hat{\delta}=-\frac{b}{2a}", size=50, color_map={r"\hat{\delta}": THEME.prediction}),
            text(
                f"integer peak  ({coarse[0]:.0f}, {coarse[1]:.0f})",
                size=23,
                font=THEME.mono_font,
            ),
            text(
                f"refined      ({pred[0]:.3f}, {pred[1]:.3f})",
                size=23,
                color=THEME.prediction,
                font=THEME.mono_font,
            ),
            text(
                f"ground truth ({gt[0]:.3f}, {gt[1]:.3f})",
                size=23,
                color=THEME.ground_truth,
                font=THEME.mono_font,
            ),
            text(
                f"error  {sample['error_px']:.3f} px",
                size=42,
                weight="SEMIBOLD",
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        details.move_to([3.45, -0.05, 0]).align_to([0.8, 0, 0], LEFT)

        self.cue_sequence(
            "060_refine_a",
            [
                ([LaggedStart(*[FadeIn(cell) for cell in cells], lag_ratio=0.02), Create(frame)], 0.65),
                ([FadeIn(coarse_mark), FadeIn(details[:3])], 0.4),
            ],
        )
        self.cue_sequence(
            "060_refine_b",
            [
                ([Create(trajectory), Transform(coarse_mark, refined_mark)], 0.75),
                ([FadeIn(details[3:])], 0.4),
                ([Circumscribe(details[-1], color=THEME.ground_truth)], 0.55),
            ],
        )
        self.pad_to_resolved_duration()


class LiveInferenceScene(FilmScene):
    scene_id = "live_inference"

    def construct(self) -> None:
        sample = self.sample("success_iid")
        self.add(section_header("07", self.scene_record["title"]))
        terminal = fit_image(self.exported("terminal_capture.png"), 11.4, 4.9)
        terminal.move_to([0, -0.1, 0])
        capture_label = label("Captured command and stdout", size=17)
        capture_label.next_to(terminal, DOWN, buff=0.16).align_to(terminal, LEFT)
        live = self.sample("live_inference")
        stdout = (live.get("stdout") or live.get("coordinate_stdout") or "").strip()
        proof = text(stdout, size=40, font=THEME.mono_font, color=THEME.prediction)
        proof.move_to([0, 1.75, 0])
        output = fit_image(self.exported("live_overlay.png"), 6.1, 5.0)
        output.move_to([0, -0.15, 0])
        output_label = text("The two stdout values map directly to the search coordinate", size=24)
        output_label.next_to(output, DOWN, buff=0.16)

        self.cue("070_live_a", [FadeIn(terminal), FadeIn(capture_label)], run_time=0.35)
        self.cue_sequence(
            "070_live_b",
            [
                ([FadeOut(terminal), FadeOut(capture_label), FadeIn(proof)], 0.3),
                ([FadeOut(proof), FadeIn(output, shift=UP * 0.12)], 0.75),
                ([FadeIn(output_label)], 0.3),
            ],
        )
        self.pad_to_resolved_duration()


class EvaluationScene(FilmScene):
    scene_id = "evaluation"

    def construct(self) -> None:
        self.add(section_header("08", self.scene_record["title"]))
        plot_data = load_json(self.exported("benchmark_plot_data.json"))
        suite_rows = suite_dot_rows(plot_data["suites"], width=11.6)
        suite_rows.scale(0.93).move_to([0, -0.08, 0])
        suite_heading = VGroup(
            text("Seven fixed suites", size=34, weight="SEMIBOLD"),
            text("Each dot represents one pair · orange marks error above 1 px", size=18, color=THEME.muted_text),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        suite_heading.move_to([GRID.left + suite_heading.width / 2, 2.25, 0])
        count = metric_statement("1,398 / 1,400", "predictions within 1 px", accent=THEME.ground_truth)
        count.move_to([-3.85, 0.75, 0]).align_to([-5.6, 0, 0], LEFT)
        rate = text("99.86%", size=28, color=THEME.ground_truth, weight="SEMIBOLD")
        rate.next_to(count, DOWN, buff=0.36).align_to(count, LEFT)
        cdf, axes = empirical_cdf(width=6.0, height=3.65)
        cdf.move_to([2.75, -0.22, 0])
        median_line = DashedLine(
            axes.c2p(0.083, 0),
            axes.c2p(0.083, 100),
            color=THEME.ground_truth,
            stroke_width=2,
        )
        p95_line = DashedLine(
            axes.c2p(0.319, 0),
            axes.c2p(0.319, 100),
            color=THEME.prediction,
            stroke_width=2,
        )
        cdf.add(median_line, p95_line)
        cdf_labels = VGroup(
            text("median  0.083 px", size=21, color=THEME.ground_truth),
            text("P95     0.319 px", size=21, color=THEME.prediction),
            text("mean wall time  239 ms", size=21),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        cdf_labels.move_to([-4.4, -1.02, 0]).align_to(count, LEFT)
        external = self.evidence_index["metrics"]["external_baselines"]
        metralign, best, others = comparison_bars(
            external["classic_adapters"],
            metralign_rate=float(self.metric("frozen_within_1px_rate")),
            width=11.2,
        )
        metralign.move_to([0, 1.48, 0])
        best.move_to([0, 0.55, 0])
        others.move_to([0, -1.1, 0])
        comparison_title = text("Fixed comparison on the same 1,400 pairs", size=30, weight="SEMIBOLD")
        comparison_title.move_to([GRID.left + comparison_title.width / 2, 2.25, 0])
        comparison_scope = label("Six fixed classic adapters · all-pair success rate", size=16)
        comparison_scope.next_to(comparison_title, DOWN, buff=0.12).align_to(comparison_title, LEFT)
        xfeat = VGroup(
            text("Additional development comparison", size=19, color=THEME.prediction),
            text("Official XFeat* + USAC_MAGSAC", size=33, weight="SEMIBOLD"),
            metric_statement("0 / 1,400", "locations within 5 px", accent=THEME.prediction),
            text("1,079 / 1,400 returned a homography", size=24, color=THEME.muted_text),
            text(
                "Different task assumptions; included as a scoped comparison, not a general benchmark verdict.",
                size=20,
                color=THEME.muted_text,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.26)
        xfeat[2].scale(0.82)
        xfeat.move_to([0, -0.1, 0])

        self.cue(
            "080_results_a",
            [FadeIn(suite_heading), LaggedStart(*[FadeIn(row) for row in suite_rows], lag_ratio=0.08)],
            run_time=0.85,
        )
        self.cue(
            "080_results_b",
            [FadeOut(suite_rows), FadeOut(suite_heading), FadeIn(count), FadeIn(rate)],
            run_time=0.28,
        )
        self.cue(
            "080_results_c",
            [FadeOut(rate), FadeIn(cdf), FadeIn(cdf_labels)],
            run_time=0.45,
        )
        self.cue_sequence(
            "080_results_d",
            [
                ([FadeOut(count), FadeOut(cdf), FadeOut(cdf_labels), FadeIn(comparison_title), FadeIn(comparison_scope)], 0.28),
                ([FadeIn(metralign), FadeIn(best)], 0.45),
                ([LaggedStart(*[FadeIn(row) for row in others], lag_ratio=0.08)], 0.7),
            ],
        )
        self.cue(
            "080_results_e",
            [FadeOut(comparison_title), FadeOut(comparison_scope), FadeOut(metralign), FadeOut(best), FadeOut(others), FadeIn(xfeat)],
            run_time=0.28,
        )
        self.pad_to_resolved_duration()


class TransferScene(FilmScene):
    scene_id = "transfer"

    def construct(self) -> None:
        self.add(section_header("09", self.scene_record["title"]))
        real_path = self.asset("real_microscopy_plate")
        independent_path = self.asset("independent_renderer_plate")
        sem = crop_image(real_path, (0.02, 0.08, 0.49, 0.36), width=11.0, height=4.05)
        tem = crop_image(real_path, (0.02, 0.65, 0.49, 0.94), width=11.0, height=4.05)
        sem.move_to([0, -0.5, 0])
        tem.move_to(sem)
        real_heading = text("Acquired microscopy", size=32, weight="SEMIBOLD")
        real_heading.move_to([GRID.left + real_heading.width / 2, 2.28, 0])
        real_scope = text(
            "Representative successful cases from independently published datasets",
            size=20,
            color=THEME.muted_text,
        )
        real_scope.next_to(real_heading, DOWN, buff=0.1).align_to(real_heading, LEFT)
        sem_note = label(
            f"SEM digital crops: {int(self.metric('sem_digital_within_1px_count'))} of {int(self.metric('sem_digital_pair_count'))} within 1 px",
            size=17,
        )
        tem_note = label(
            f"Registered TEM: {int(self.metric('registered_tem_within_1px_count'))} of {int(self.metric('registered_tem_pair_count'))} within 1 px",
            size=17,
        )
        sem_note.move_to([GRID.right - sem_note.width / 2, 2.22, 0])
        tem_note.move_to([GRID.right - tem_note.width / 2, 2.22, 0])
        dram = crop_image(independent_path, (0.02, 0.105, 0.49, 0.414), width=11.0, height=4.05)
        finfet = crop_image(independent_path, (0.50, 0.105, 0.98, 0.414), width=11.0, height=4.05)
        dram.move_to([0, -0.5, 0])
        finfet.move_to(dram)
        independent_heading = text("Separately implemented renderer", size=32, weight="SEMIBOLD")
        independent_heading.move_to([GRID.left + independent_heading.width / 2, 2.28, 0])
        independent_scope = text(
            "Different capture code · shared benchmark geometry assumptions",
            size=20,
            color=THEME.muted_text,
        )
        independent_scope.next_to(independent_heading, DOWN, buff=0.1).align_to(independent_heading, LEFT)
        independent_note = label(
            f"Independent renderer: {int(self.metric('independent_within_1px_count'))} of {int(self.metric('independent_pair_count'))} within {float(self.metric('independent_primary_threshold_px')):g} px",
            size=17,
            color=THEME.ground_truth,
        )
        independent_note.move_to([GRID.right - independent_note.width / 2, 2.22, 0])

        self.cue_sequence(
            "090_transfer_a",
            [
                ([FadeIn(real_heading), FadeIn(real_scope), FadeIn(sem), FadeIn(sem_note)], 0.3),
                ([], 3.6),
                ([FadeTransform(sem, tem), FadeTransform(sem_note, tem_note)], 0.55),
            ],
        )
        self.cue_sequence(
            "090_transfer_b",
            [
                ([FadeOut(tem), FadeOut(tem_note), FadeOut(real_heading), FadeOut(real_scope), FadeIn(independent_heading), FadeIn(independent_scope), FadeIn(dram), FadeIn(independent_note)], 0.3),
                ([], 2.25),
                ([FadeTransform(dram, finfet)], 0.55),
            ],
        )
        self.pad_to_resolved_duration()


class FailureScene(FilmScene):
    scene_id = "failure"

    def construct(self) -> None:
        self.add(section_header("10", self.scene_record["title"]))
        failure = self.sample("failure_scan")
        reference = fit_image(self.asset("frozen_failure_reference"), 2.45, 2.45)
        search = fit_image(self.asset("frozen_failure_search"), 5.15, 5.15)
        reference.move_to([-4.75, 1.00, 0])
        search.move_to([2.25, -0.05, 0])
        mapper = image_mapper(search)
        gt = _xy(failure, "ground_truth")
        pred = _xy(failure, "prediction")
        gt_point = mapper.point(*gt)
        pred_point = mapper.point(*pred)
        gt_mark = registration_cross(gt_point, color=THEME.ground_truth, size=0.16, stroke_width=2.7)
        pred_mark = registration_cross(pred_point, color=THEME.prediction, size=0.16, stroke_width=2.7)
        displacement = Line(gt_point, pred_point, color=THEME.prediction, stroke_width=2.2)
        displacement_label = text(
            f"{failure['error_px']:.3f} px",
            size=24,
            color=THEME.prediction,
            weight="SEMIBOLD",
        )
        displacement_label.next_to(displacement.get_center(), UP, buff=0.12)
        facts = VGroup(
            text("Ambiguity detected", size=29, weight="SEMIBOLD", color=THEME.prediction),
            text(
                f"{int(failure['diagnostics']['tied_count']):,} plausible locations",
                size=22,
                font=THEME.mono_font,
            ),
            text(
                f"residual support  {failure['diagnostics']['ambiguity_evidence']['residual_evidence']:.3f}",
                size=20,
                font=THEME.mono_font,
            ),
            text(
                "Weak site-specific evidence leaves\nmany plausible absolute positions.",
                size=21,
                color=THEME.muted_text,
                line_spacing=0.9,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts.move_to([-4.75, -1.43, 0]).align_to(reference, LEFT)
        legend = VGroup(
            label("truth", size=15, color=THEME.ground_truth),
            label("returned", size=15, color=THEME.prediction),
        ).arrange(RIGHT, buff=0.35)
        legend.next_to(search, DOWN, buff=0.16).align_to(search, LEFT)

        self.cue_sequence(
            "100_failure_a",
            [
                ([FadeIn(reference), FadeIn(search), FadeIn(facts[:2])], 0.35),
                ([FadeIn(gt_mark), FadeIn(pred_mark), Create(displacement), FadeIn(displacement_label), FadeIn(legend)], 0.65),
            ],
        )
        self.cue("100_failure_b", [FadeIn(facts[2:]), Indicate(displacement, color=THEME.prediction)], run_time=0.65)
        self.pad_to_resolved_duration()


class ReproducibilityScene(FilmScene):
    scene_id = "reproducibility"

    def construct(self) -> None:
        project = self.resolved["project"]
        repository = project["repository_url"]
        website = project["website_url"]
        self.add(section_header("11", self.scene_record["title"]))
        commands = VGroup(
            text(f"$ git clone {repository}.git", size=25, font=THEME.mono_font),
            text("$ python -m pip install -c constraints.txt .", size=25, font=THEME.mono_font),
            text("$ metralign --reference reference.png --search search.png", size=25, font=THEME.mono_font),
            text("$ python evaluate.py --data-dir data/smoke --method full", size=25, font=THEME.mono_font),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.43)
        commands.move_to([0, 0.45, 0])
        evidence_line = text(
            "generation · inference · evaluation · reports · input hashes",
            size=24,
            color=THEME.muted_text,
        )
        evidence_line.next_to(commands, DOWN, buff=0.6).align_to(commands, LEFT)
        mark = fit_image(self.asset("metralign_mark"), 1.05, 1.05)
        name = text(f"{project['name']} {project['version']}", size=52, weight="SEMIBOLD")
        team = text(project["team"]["display_name"], size=24, color=THEME.muted_text)
        repo = text(repository.removeprefix("https://"), size=27, font=THEME.mono_font)
        site = text(website.removeprefix("https://").rstrip("/"), size=22, color=THEME.muted_text)
        identity = Group(mark, name, team, repo, site).arrange(DOWN, buff=0.25)
        identity.move_to([0, 0.2, 0])
        partners = Group(
            fit_image(self.asset("semi_organiser_mark"), 1.55, 0.48),
            fit_image(self.asset("applied_materials_partner_mark"), 1.2, 0.48),
        ).arrange(RIGHT, buff=1.25)
        partner_labels = VGroup(
            label("organiser", size=14),
            label("Drift-Sense industry partner", size=14),
        )
        for partner_label, partner in zip(partner_labels, partners):
            partner_label.next_to(partner, UP, buff=0.1)
        partner_group = Group(partners, partner_labels)
        partner_group.move_to([0, -2.43, 0])

        self.cue(
            "110_end_a",
            [LaggedStart(*[FadeIn(command, shift=UP * 0.08) for command in commands], lag_ratio=0.13), FadeIn(evidence_line)],
            run_time=0.85,
        )
        self.cue(
            "110_end_b",
            [FadeOut(commands), FadeOut(evidence_line), FadeIn(identity), FadeIn(partner_group)],
            run_time=0.28,
        )
        self.pad_to_resolved_duration()
