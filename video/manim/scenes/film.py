"""Twelve independently renderable shots for the Metralign technical film."""

from __future__ import annotations

import csv
import math

from manim import (
    Axes,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    ReplacementTransform,
    Transform,
    VGroup,
)

from video.manim.base_scene import FilmScene
from video.manim.components import (
    fit_image,
    image_panel,
    label,
    metric_block,
    registration_cross,
    scope_note,
    section_header,
    text,
)
from video.manim.coordinates import ImageCoordinateMap
from video.manim.evidence import load_json
from video.manim.grid import GRID
from video.manim.theme import THEME


def _float(value, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _xy(sample: dict, key: str) -> tuple[float, float]:
    value = sample[key]
    return float(value[0]), float(value[1])


def _panel_image(panel):
    return panel[0]


def _case_id(sample: dict) -> str:
    value = sample.get("id") or sample.get("case_id")
    return str(value).upper() if value else "BOUND CASE"


def synthetic_case_stamp(sample: dict) -> VGroup:
    """Small, persistent scope disclosure for the synthetic method trace."""

    architecture = str(sample.get("architecture", "")).upper() or "ARCHITECTURE"
    stamp = label(
        f"SEALED SYNTHETIC · {architecture} · {_case_id(sample)}",
        size=12,
    )
    stamp.move_to(
        [GRID.right - stamp.width / 2, GRID.top - 0.16, 0]
    )
    return VGroup(stamp)


def phase_drift_panel(reference_path, search_path) -> VGroup:
    """Plot the measured Y-carrier phase samples and fitted drift lines."""

    def read_rows(path):
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

    reference = read_rows(reference_path)
    search = read_rows(search_path)
    all_rows = reference + search
    x_min = min(row["x"] for row in all_rows)
    x_max = max(row["x"] for row in all_rows)
    used_phases = [row["phase"] for row in all_rows if row["used"]]
    y_min = math.floor(min(used_phases) * 2.0) / 2.0
    y_max = math.ceil(max(used_phases) * 2.0) / 2.0

    axes = Axes(
        x_range=[x_min, x_max, x_max - x_min],
        y_range=[y_min, y_max, 0.5],
        x_length=3.18,
        y_length=2.42,
        tips=False,
        axis_config={"color": THEME.rule, "stroke_width": 1.0},
    )

    def measured_points(rows, color):
        used = [row for row in rows if row["used"]]
        stride = max(1, len(used) // 100)
        return VGroup(
            *[
                Dot(
                    axes.c2p(row["x"], row["phase"]),
                    radius=0.010,
                    color=color,
                    fill_opacity=0.48,
                    stroke_width=0,
                )
                for row in used[::stride]
            ]
        )

    def fit_line(rows, color):
        return Line(
            axes.c2p(rows[0]["x"], rows[0]["fit"]),
            axes.c2p(rows[-1]["x"], rows[-1]["fit"]),
            color=color,
            stroke_width=2.1,
        )

    reference_points = measured_points(reference, THEME.ground_truth)
    search_points = measured_points(search, THEME.prediction)
    reference_fit = fit_line(reference, THEME.ground_truth)
    search_fit = fit_line(search, THEME.prediction)
    reference_label = label("REFERENCE", size=11, color=THEME.ground_truth)
    search_label = label("SEARCH", size=11, color=THEME.prediction)
    reference_at = reference[int(0.68 * (len(reference) - 1))]
    search_at = search[int(0.68 * (len(search) - 1))]
    reference_label.next_to(
        axes.c2p(reference_at["x"], reference_at["fit"]), UP, buff=0.06
    )
    search_label.next_to(
        axes.c2p(search_at["x"], search_at["fit"]), DOWN, buff=0.06
    )
    x_label = label("SCAN COORDINATE / PX", size=10)
    x_label.next_to(axes, DOWN, buff=0.09)
    title = label("Y-CARRIER PHASE DRIFT", size=15)
    title.next_to(axes, UP, buff=0.12).align_to(axes, LEFT)
    border = Rectangle(
        width=3.55,
        height=3.28,
        color=THEME.rule,
        stroke_width=1.0,
        fill_opacity=0.0,
    ).move_to(axes)
    return VGroup(
        axes,
        reference_points,
        search_points,
        reference_fit,
        search_fit,
        reference_label,
        search_label,
        x_label,
        title,
        border,
    )


class OpenScene(FilmScene):
    scene_id = "open"

    def construct(self) -> None:
        project = self.resolved["project"]
        success = self.sample("success_iid")
        plate = fit_image(self.asset("frozen_success_search"), 5.7, 6.65)
        plate.move_to(GRID.center(7, 12, -0.08))
        plate.set_opacity(0.86)
        plate_rule = Rectangle(
            width=plate.width,
            height=plate.height,
            color=THEME.rule,
            stroke_width=1,
        ).move_to(plate)
        plate_scope = label(
            f"SEALED SYNTHETIC · {str(success['architecture']).upper()} · {_case_id(success)}",
            size=12,
        )
        plate_scope.next_to(plate, UP, buff=0.12).align_to(plate, LEFT)

        mark = fit_image(self.asset("metralign_mark"), 1.18, 1.18)
        mark.move_to(GRID.center(0, 2, 1.78))
        title = text(project["name"].upper(), size=86, font=THEME.display_font, weight="BOLD")
        title.move_to(GRID.center(0, 7, 0.56)).align_to([GRID.left, 0, 0], LEFT)
        subtitle = text(
            project["technical_title"].replace(" under ", "\nunder "),
            size=31,
            color=THEME.primary_text,
        )
        subtitle.move_to(GRID.center(0, 7, -0.65)).align_to(title, LEFT)
        event = label(
            f"{project['event']} · {project['submission_context']}",
            size=16,
        )
        event.move_to(GRID.center(0, 7, -2.25)).align_to(title, LEFT)
        method = label("DETERMINISTIC · TRAINING-FREE · CPU-ONLY", size=15)
        method.next_to(event, DOWN, buff=0.18).align_to(event, LEFT)
        edge_rule = Line(
            [GRID.col(7) - 0.18, GRID.top, 0],
            [GRID.col(7) - 0.18, GRID.bottom, 0],
            color=THEME.rule,
            stroke_width=1,
        )

        self.cue(
            "000_open_a",
            [
                FadeIn(plate),
                Create(plate_rule),
                Create(edge_rule),
                FadeIn(plate_scope),
                FadeIn(mark),
                FadeIn(title),
            ],
        )
        self.cue("000_open_b", [FadeIn(subtitle), FadeIn(event), FadeIn(method)])
        self.pad_to_resolved_duration()


class ProblemScene(FilmScene):
    scene_id = "problem"

    def construct(self) -> None:
        success = self.sample("success_iid")
        header = VGroup(
            section_header("01", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        reference = image_panel(
            self.asset("frozen_success_reference"), width=3.8, height=4.5, panel_label="REFERENCE / FINE FIELD"
        )
        search = image_panel(
            self.asset("frozen_success_search"), width=5.45, height=5.45, panel_label="SEARCH / WIDE FIELD"
        )
        reference.move_to(GRID.center(0, 4, -0.22))
        search.move_to(GRID.center(6, 12, -0.22))
        search_image = _panel_image(search)
        gt = _xy(success, "ground_truth")
        pred = _xy(success, "prediction")
        pixel_height, pixel_width = search_image.pixel_array.shape[:2]
        mapper = ImageCoordinateMap.from_mobject(search_image, pixel_width, pixel_height)
        relative_field_width = float(success["diagnostics"]["selected_scale"])
        roi = Rectangle(
            width=search_image.width * relative_field_width,
            height=search_image.height * relative_field_width,
            color=THEME.ground_truth,
            stroke_width=2,
        ).move_to(mapper.point(*gt))
        physical = label("APPROX. REFERENCE FOV", size=13, color=THEME.ground_truth)
        physical.next_to(roi, UP, buff=0.1)
        gt_cross = registration_cross(mapper.point(*gt), color=THEME.ground_truth, size=0.1)
        pred_cross = registration_cross(mapper.point(*pred), color=THEME.prediction, size=0.14)
        coordinate = text(
            f"PRED  ({_float(pred[0])}, {_float(pred[1])}) px",
            size=21,
            font=THEME.mono_font,
            color=THEME.prediction,
        )
        coordinate.move_to(GRID.center(6, 12, -3.25))
        absolute = text("ONE ABSOLUTE COORDINATE", size=35, font=THEME.display_font, weight="BOLD")
        absolute.set_width(min(absolute.width, GRID.width(0, 5)))
        absolute.move_to(GRID.center(0, 5, -3.0))

        self.add(header)
        self.cue("010_problem_a", [FadeIn(reference), FadeIn(search), Create(roi), FadeIn(physical)])
        candidate_overlay = fit_image(self.exported("candidate_overlay.png"), search_image.width, search_image.height)
        candidate_overlay.move_to(search_image)
        self.cue(
            "010_problem_b",
            [FadeOut(roi), FadeOut(physical), FadeIn(candidate_overlay)],
        )
        self.cue(
            "010_problem_c",
            [FadeIn(gt_cross), FadeIn(pred_cross), FadeIn(coordinate), FadeIn(absolute)],
        )
        self.pad_to_resolved_duration()


class BaselineScene(FilmScene):
    scene_id = "baseline"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("02", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        score_map = image_panel(
            self.exported("baseline_score_map.png"),
            width=5.8,
            height=5.45,
            panel_label="NORMALIZED TEMPLATE CORRELATION",
        )
        overlay = image_panel(
            self.exported("baseline_candidate_overlay.png"),
            width=5.8,
            height=5.45,
            panel_label="PERIODIC LOCAL MAXIMA / ACTUAL MAP",
        )
        score_map.move_to(GRID.center(0, 6, -0.25))
        overlay.move_to(GRID.center(6, 12, -0.25))
        note = scope_note(
            "Local similarity identifies a phase family; it does not establish the absolute site.",
            width=10.8,
        )
        note.move_to(GRID.center(1, 11, -3.28))
        self.cue("020_baseline_a", [FadeIn(score_map)])
        self.cue("020_baseline_b", [FadeIn(overlay), FadeIn(note)])
        self.pad_to_resolved_duration()


class CalibrationScene(FilmScene):
    scene_id = "calibration"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("03", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        source = image_panel(
            self.asset("frozen_success_search"), width=3.65, height=3.65, panel_label="SEARCH"
        )
        fft_search = image_panel(
            self.exported("fft_search.png"), width=3.65, height=3.65, panel_label="LOG-MAGNITUDE FFT"
        )
        fft_reference = image_panel(
            self.exported("fft_reference.png"), width=3.65, height=3.65, panel_label="REFERENCE FFT"
        )
        phase_drift = phase_drift_panel(
            self.exported("phase_drift_reference_y.csv"),
            self.exported("phase_drift_search_y.csv"),
        )
        phase_transform = load_json(self.exported("phase_transform.json"))
        source.move_to(GRID.center(0, 4, 0.15))
        phase_drift.move_to(source)
        fft_search.move_to(GRID.center(4, 8, 0.15))
        fft_reference.move_to(GRID.center(8, 12, 0.15))
        diagnostics = success["diagnostics"]
        measured = label(f"MEASURED ON CASE {_case_id(success)}", size=15)
        transform_values = VGroup(
            text(
                f"scale  {_float(diagnostics['selected_scale'], 6)}",
                size=22,
                font=THEME.mono_font,
            ),
            text(
                f"rotation  {_float(diagnostics['selected_rotation_deg'], 3)}°",
                size=22,
                font=THEME.mono_font,
            ),
        ).arrange(RIGHT, buff=0.7)
        pitch_x, pitch_y = phase_transform["clipped_pitch_xy_px"]
        period_and_confidence = VGroup(
            text(
                f"period  {_float(pitch_x, 3)} × {_float(pitch_y, 3)} px",
                size=20,
                font=THEME.mono_font,
            ),
            text(
                f"confidence  {_float(diagnostics['spectral_confidence'], 3)}",
                size=20,
                font=THEME.mono_font,
            ),
        ).arrange(RIGHT, buff=0.7)
        values = VGroup(measured, transform_values, period_and_confidence).arrange(
            DOWN, buff=0.12
        )
        values.move_to(GRID.center(0, 12, -2.55))

        self.cue("030_phase_a", [FadeIn(source), FadeIn(fft_search)])
        self.cue(
            "030_phase_b",
            [FadeOut(source), FadeIn(phase_drift), FadeIn(fft_reference), FadeIn(values)],
        )
        self.cue("030_phase_c")
        self.pad_to_resolved_duration()


class DifferenceScene(FilmScene):
    scene_id = "difference"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("04", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        original = image_panel(
            self.exported("matched_reference_template.png"), width=3.75, height=4.2, panel_label="MATCHED REFERENCE"
        )
        ref_difference = image_panel(
            self.exported("period_difference_reference.png"),
            width=3.75,
            height=4.2,
            panel_label="REFERENCE RESIDUAL",
        )
        search_difference = image_panel(
            self.exported("period_difference_search_crop.png"),
            width=3.75,
            height=4.2,
            panel_label="SEARCH RESIDUAL",
        )
        original.move_to(GRID.center(0, 4, -0.05))
        ref_difference.move_to(GRID.center(4, 8, -0.05))
        search_difference.move_to(GRID.center(8, 12, -0.05))
        operator = text(
            "R_x = I(x,y) − ½[I(x−p_x,y) + I(x+p_x,y)]",
            size=26,
            font=THEME.mono_font,
        )
        operator.move_to(GRID.center(0, 12, -2.9))
        note = label(
            "X CHANNEL SHOWN · Y ANALOGOUS · CORRELATION SCORES FUSED",
            size=14,
        )
        note.next_to(operator, DOWN, buff=0.18)

        self.cue("040_difference_a", [FadeIn(original), FadeIn(ref_difference), FadeIn(operator)])
        self.cue("040_difference_b", [FadeIn(search_difference)])
        self.cue("040_difference_c", [FadeIn(note)])
        self.pad_to_resolved_duration()


class CandidateScene(FilmScene):
    scene_id = "candidates"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("05", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        overlay = image_panel(
            self.exported("candidate_overlay.png"),
            width=6.2,
            height=5.8,
            panel_label="SUPPORTED LOCAL MAXIMA",
        )
        overlay.move_to(GRID.center(0, 7, -0.15))
        d = success["diagnostics"]
        diagnostics = VGroup(
            label("BOUND DIAGNOSTICS", size=16),
            text(f"top score       {_float(d['score'], 6)}", size=20, font=THEME.mono_font),
            text(f"runner-up       {_float(d['runner_up_score'], 6)}", size=20, font=THEME.mono_font),
            text(f"residual support {_float(d['ambiguity_evidence']['residual_evidence'], 6)}", size=20, font=THEME.mono_font),
            text(f"tied candidates  {int(d['tied_count'])}", size=20, font=THEME.mono_font),
            text(
                f"ambiguous         {str(bool(d['ambiguity_flag'])).upper()}",
                size=20,
                font=THEME.mono_font,
                color=THEME.primary_text,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
        diagnostics.move_to(GRID.center(8, 12, 0.35)).align_to([GRID.col(8), 0, 0], LEFT)
        gate = scope_note(
            "Tie prior enabled only with low residual support or transform instability.",
            width=4.15,
        )
        gate.move_to(GRID.center(8, 12, -2.15)).align_to(diagnostics, LEFT)
        rule = Line(
            [GRID.col(8), -1.72, 0],
            [GRID.right, -1.72, 0],
            color=THEME.rule,
            stroke_width=1,
        )

        self.cue("050_candidates_a", [FadeIn(overlay), FadeIn(diagnostics[:4])])
        self.cue("050_candidates_b", [Create(rule), FadeIn(gate)])
        self.cue("050_candidates_c", [FadeIn(diagnostics[4:])])
        self.pad_to_resolved_duration()


class RefinementScene(FilmScene):
    scene_id = "refinement"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("06", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        patch = image_panel(
            self.exported("refinement_patch.png"), width=6.45, height=5.65, panel_label="LOCAL SCORE SURFACE / ACTUAL PEAK"
        )
        patch.move_to(GRID.center(0, 7, -0.18))
        pred = _xy(success, "prediction")
        gt = _xy(success, "ground_truth")
        coarse = success.get("coarse_prediction", [round(pred[0]), round(pred[1])])
        detail = VGroup(
            label(f"SEALED IID CASE {_case_id(success)}", size=16),
            text(
                f"integer  ({_float(coarse[0], 0)}, {_float(coarse[1], 0)})",
                size=22,
                font=THEME.mono_font,
            ),
            text(
                f"refined  ({_float(pred[0])}, {_float(pred[1])})",
                size=22,
                font=THEME.mono_font,
                color=THEME.prediction,
            ),
            text(
                f"truth    ({_float(gt[0])}, {_float(gt[1])})",
                size=22,
                font=THEME.mono_font,
                color=THEME.ground_truth,
            ),
            text(
                f"error     {_float(success['error_px'])} px",
                size=34,
                font=THEME.display_font,
                weight="BOLD",
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        detail.set_width(min(detail.width, GRID.width(8, 12)))
        detail.move_to(GRID.center(8, 12, 0.0)).align_to([GRID.col(8), 0, 0], LEFT)
        refinement = load_json(self.exported("refinement_patch.json"))
        patch_origin_x, patch_origin_y = refinement["score_patch_top_left_xy"]
        coarse_x, coarse_y = refinement["integer_peak_top_left_xy"]
        refined_x, refined_y = refinement["refined_top_left_xy"]
        patch_image = _panel_image(patch)
        raster_height, raster_width = patch_image.pixel_array.shape[:2]
        score_height = len(refinement["score_values"])
        score_width = len(refinement["score_values"][0])
        patch_mapper = ImageCoordinateMap.from_mobject(
            patch_image,
            raster_width,
            raster_height,
        )

        def score_cell_center(index: float, cell_count: int, raster_size: int) -> float:
            return (index + 0.5) * raster_size / cell_count - 0.5

        coarse_point = patch_mapper.point(
            score_cell_center(
                float(coarse_x) - float(patch_origin_x), score_width, raster_width
            ),
            score_cell_center(
                float(coarse_y) - float(patch_origin_y), score_height, raster_height
            ),
        )
        refined_point = patch_mapper.point(
            score_cell_center(
                float(refined_x) - float(patch_origin_x), score_width, raster_width
            ),
            score_cell_center(
                float(refined_y) - float(patch_origin_y), score_height, raster_height
            ),
        )
        coarse_mark = registration_cross(coarse_point, color=THEME.muted_text, size=0.2)
        refined_mark = registration_cross(refined_point, color=THEME.prediction, size=0.2)

        self.cue("060_refine_a", [FadeIn(patch), FadeIn(detail[:2]), FadeIn(coarse_mark)])
        self.cue(
            "060_refine_b",
            [Transform(coarse_mark, refined_mark), FadeIn(detail[2:])],
        )
        self.pad_to_resolved_duration()


class LiveInferenceScene(FilmScene):
    scene_id = "live_inference"

    def construct(self) -> None:
        success = self.sample("success_iid")
        self.add(
            section_header("07", self.scene_record["title"]),
            synthetic_case_stamp(success),
        )
        terminal = image_panel(
            self.exported("terminal_capture.png"),
            width=11.9,
            height=5.75,
            panel_label="REAL COMMAND / CAPTURED STDOUT",
        )
        terminal.move_to(GRID.center(0, 12, -0.18))
        overlay = image_panel(
            self.exported("live_overlay.png"), width=6.15, height=5.75, panel_label="OUTPUT MAPPED TO SEARCH COORDINATES"
        )
        overlay.move_to(GRID.center(6, 12, -0.18))
        terminal_left = terminal.copy()
        terminal_left.scale(0.55).move_to(GRID.center(0, 6, -0.18))
        live = self.sample("live_inference")
        output = live.get("stdout") or live.get("coordinate_stdout") or ""
        proof = text(output.strip(), size=27, font=THEME.mono_font, color=THEME.prediction)
        proof.move_to(GRID.center(0, 6, -3.25))

        self.cue("070_live_a", [FadeIn(terminal)])
        self.cue(
            "070_live_b",
            [ReplacementTransform(terminal, terminal_left), FadeIn(overlay), FadeIn(proof)],
        )
        self.pad_to_resolved_duration()


class EvaluationScene(FilmScene):
    scene_id = "evaluation"

    def construct(self) -> None:
        self.add(section_header("08", self.scene_record["title"]))
        plot = image_panel(
            self.exported("evaluation_error_distribution.png"),
            width=6.15,
            height=4.45,
            panel_label=f"ERROR DISTRIBUTION / {int(self.metric('frozen_pair_count')):,} SEALED PAIRS",
        )
        plot.move_to(GRID.center(0, 7, -0.05))
        pair_count = int(self.metric("frozen_pair_count"))
        within = int(self.metric("frozen_within_1px_count"))
        threshold = float(self.metric("frozen_primary_threshold_px"))
        rate = float(self.metric("frozen_within_1px_rate"))
        if rate <= 1.0:
            rate *= 100.0
        blocks = VGroup(
            metric_block(f"{within:,}/{pair_count:,}", f"WITHIN {threshold:g} PX", width=4.3),
            metric_block(f"{_float(self.metric('frozen_median_error_px'))} PX", "MEDIAN ERROR", width=4.3),
            metric_block(f"{_float(self.metric('frozen_p95_error_px'))} PX", "P95 ERROR", width=4.3),
            metric_block(f"{_float(self.metric('frozen_mean_runtime_ms'), 0)} MS", "MEAN LOCALIZATION WALL TIME", width=4.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        blocks.move_to(GRID.center(8, 12, -0.25)).align_to([GRID.col(8), 0, 0], LEFT)
        suite_table = image_panel(
            self.exported("evaluation_suite_strip.png"),
            width=11.75,
            height=5.05,
            panel_label="PER-SUITE SEALED RESULTS / EXACT COUNTS",
        )
        suite_table.move_to(GRID.center(0, 12, -0.2))
        primary_scope = label(
            f"SEALED SYNTHETIC REPORT · {rate:.2f}% WITHIN {threshold:g} PX",
            size=15,
        )
        primary_scope.move_to(GRID.center(0, 12, -3.2))
        comparison = fit_image(
            self.exported("external_comparison_table.png"),
            11.75,
            5.65,
        )
        comparison.move_to(GRID.center(0, 12, -0.16))

        self.cue("080_results_a", [FadeIn(suite_table)])
        self.cue(
            "080_results_b",
            [FadeOut(suite_table), FadeIn(plot), FadeIn(blocks[:2])],
        )
        self.cue("080_results_c", [FadeIn(blocks[2:]), FadeIn(primary_scope)])
        self.cue(
            "080_results_d",
            [
                FadeOut(plot),
                FadeOut(blocks),
                FadeOut(primary_scope),
                FadeIn(comparison),
            ],
        )
        self.cue("080_results_e")
        self.pad_to_resolved_duration()


class TransferScene(FilmScene):
    scene_id = "transfer"

    def construct(self) -> None:
        self.add(section_header("09", self.scene_record["title"]))
        real = image_panel(
            self.asset("real_microscopy_plate"),
            width=5.8,
            height=4.75,
            panel_label="ACQUIRED MICROSCOPY / MECHANICALLY SELECTED SUCCESSES",
        )
        independent = image_panel(
            self.asset("independent_renderer_plate"),
            width=5.8,
            height=4.75,
            panel_label="SEPARATE RENDERER / MECHANICALLY SELECTED SUCCESSES",
        )
        real.move_to(GRID.center(0, 6, 0.12))
        independent.move_to(GRID.center(6, 12, 0.12))
        real_note = VGroup(
            label(
                f"DIGITAL-CROP SEM · {int(self.metric('sem_digital_within_1px_count'))}/{int(self.metric('sem_digital_pair_count'))} ≤ 1 PX",
                size=12,
            ),
            label(
                f"CARINTHIA SEM · {int(self.metric('carinthia_within_1px_count'))}/{int(self.metric('carinthia_pair_count'))} ≤ 1 PX · {int(self.metric('carinthia_fallback_count'))} FALLBACKS",
                size=12,
            ),
            label(
                f"REGISTERED TEM · {int(self.metric('registered_tem_within_1px_count'))}/{int(self.metric('registered_tem_pair_count'))} ≤ 1 PX · {int(self.metric('registered_tem_fallback_count'))} FALLBACKS",
                size=12,
            ),
            label("DEVELOPMENT TRANSFER · NO MICROSCOPE-STAGE ACCURACY CLAIM", size=11),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        real_note.move_to(GRID.center(0, 6, -3.03))
        independent_threshold = float(self.metric("independent_primary_threshold_px"))
        independent_note = VGroup(
            label(
                f"{int(self.metric('independent_within_1px_count'))}/{int(self.metric('independent_pair_count'))} WITHIN {independent_threshold:g} PX",
                size=13,
            ),
            label("INDEPENDENT CODE · SHARED TASK-LEVEL GEOMETRY ASSUMPTIONS", size=11),
            label("DEVELOPMENT TRANSFER CHECK · NOT PART OF THE SEALED CLAIM", size=11),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        independent_note.move_to(GRID.center(6, 12, -3.18))

        self.cue("090_transfer_a", [FadeIn(real), FadeIn(independent)])
        self.cue("090_transfer_b", [FadeIn(real_note), FadeIn(independent_note)])
        self.pad_to_resolved_duration()


class FailureScene(FilmScene):
    scene_id = "failure"

    def construct(self) -> None:
        self.add(section_header("10", self.scene_record["title"]))
        reference = image_panel(
            self.asset("frozen_failure_reference"), width=3.45, height=4.45, panel_label="REFERENCE"
        )
        overlay = image_panel(
            self.exported("failure_overlay.png"), width=6.35, height=5.8, panel_label="SEARCH / TRUTH AND RETURNED LOCATION"
        )
        reference.move_to(GRID.center(0, 4, -0.1))
        overlay.move_to(GRID.center(5, 12, -0.1))
        failure = self.sample("failure_scan")
        pred = _xy(failure, "prediction")
        gt = _xy(failure, "ground_truth")
        details = VGroup(
            text(f"error  {_float(failure['error_px'])} px", size=34, font=THEME.display_font, weight="BOLD"),
            text(f"pred   ({_float(pred[0])}, {_float(pred[1])})", size=18, font=THEME.mono_font, color=THEME.prediction),
            text(f"truth  ({_float(gt[0])}, {_float(gt[1])})", size=18, font=THEME.mono_font, color=THEME.ground_truth),
            label(
                f"AMBIGUOUS · {str(bool(failure['diagnostics']['ambiguity_flag'])).upper()}",
                size=15,
                color=THEME.primary_text,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        details.move_to(GRID.center(0, 4, -2.78)).align_to([GRID.left, 0, 0], LEFT)
        note = scope_note(
            "Observed: low site-specific support and thousands of plausible positions. The report does not claim a more specific physical cause.",
            width=6.4,
        )
        note.move_to(GRID.center(5, 12, -3.27))

        self.cue("100_failure_a", [FadeIn(reference), FadeIn(overlay), FadeIn(details)])
        self.cue("100_failure_b", [FadeIn(note)])
        self.pad_to_resolved_duration()


class ReproducibilityScene(FilmScene):
    scene_id = "reproducibility"

    def construct(self) -> None:
        project = self.resolved["project"]
        repository = project["repository_url"]
        website = project["website_url"]
        self.add(section_header("11", self.scene_record["title"]))
        command_lines = VGroup(
            text(f"git clone {repository}.git", size=21, font=THEME.mono_font),
            text("python -m pip install -c constraints.txt .", size=21, font=THEME.mono_font),
            text("metralign --reference reference.png --search search.png", size=21, font=THEME.mono_font),
            text("python evaluate.py --data-dir data/smoke --method full", size=21, font=THEME.mono_font),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34)
        if command_lines.width > GRID.width(0, 7):
            command_lines.scale_to_fit_width(GRID.width(0, 7))
        command_lines.move_to(GRID.center(0, 8, 0.3)).align_to([GRID.left, 0, 0], LEFT)
        rule = Line(
            [GRID.left, -1.45, 0], [GRID.col(8), -1.45, 0], color=THEME.rule, stroke_width=1
        )
        manifest_note = VGroup(
            label("BOUND RELEASE", size=15),
            text("SEALED MANIFESTS · COMPLETE REPORTS · PER-IMAGE HASHES", size=23),
            text("FULL TEST SUITE · HISTORY-AWARE RELEASE SCAN", size=23),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        if manifest_note.width > GRID.width(0, 7):
            manifest_note.scale_to_fit_width(GRID.width(0, 7))
        manifest_note.move_to(GRID.center(0, 8, -2.25)).align_to(command_lines, LEFT)

        divider = Line(
            [GRID.col(8) - 0.18, GRID.top - 0.72, 0],
            [GRID.col(8) - 0.18, GRID.bottom, 0],
            color=THEME.rule,
            stroke_width=1,
        )

        mark = fit_image(self.asset("metralign_mark"), 1.0, 1.0)
        mark.move_to(GRID.center(9, 11, 1.62))
        version = self.resolved["project"]["version"]
        name = text(
            f"{project['name'].upper()} {version}",
            size=42,
            font=THEME.display_font,
            weight="BOLD",
        )
        name.move_to(GRID.center(8, 12, 0.48))
        team = label(project["team"]["display_name"], size=15)
        team.next_to(name, DOWN, buff=0.25)
        repo = text(repository.removeprefix("https://"), size=23, font=THEME.mono_font)
        repo.next_to(team, DOWN, buff=0.42)
        site = text(
            website.removeprefix("https://").rstrip("/"),
            size=19,
            font=THEME.mono_font,
            color=THEME.muted_text,
        )
        site.next_to(repo, DOWN, buff=0.15)

        semi = fit_image(self.asset("semi_organiser_mark"), 1.55, 0.45)
        applied = fit_image(self.asset("applied_materials_partner_mark"), 1.15, 0.45)
        semi_label = label("ORGANISER", size=11)
        applied_label = label("DRIFT-SENSE INDUSTRY PARTNER", size=11)
        semi.next_to(semi_label, DOWN, buff=0.1)
        applied.next_to(applied_label, DOWN, buff=0.1)
        brand_row = VGroup(semi_label, applied_label).arrange(RIGHT, buff=1.15)
        brand_row.move_to(GRID.center(8, 12, -2.62))
        semi.next_to(semi_label, DOWN, buff=0.1)
        applied.next_to(applied_label, DOWN, buff=0.1)

        self.cue("110_end_a", [FadeIn(command_lines), Create(rule), FadeIn(manifest_note)])
        self.cue(
            "110_end_b",
            [Create(divider), FadeIn(mark), FadeIn(name), FadeIn(team), FadeIn(repo), FadeIn(site), FadeIn(semi_label), FadeIn(semi), FadeIn(applied_label), FadeIn(applied)],
        )
        self.pad_to_resolved_duration()
