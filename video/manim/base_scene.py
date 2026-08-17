"""Base scene that consumes only the resolved timeline and bound evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from manim import MovingCameraScene, config
from manim.utils.rate_functions import ease_in_out_cubic, ease_out_quart

from video.manim.evidence import REPO_ROOT, load_json, metric_value, resolve_repo_path, sample_value
from video.manim.theme import THEME


class FilmScene(MovingCameraScene):
    scene_id = ""

    def setup(self) -> None:
        super().setup()
        config.background_color = THEME.background
        self.camera.background_color = THEME.background
        resolved_path = Path(
            os.environ.get(
                "METRALIGN_RESOLVED_FILM",
                str(REPO_ROOT / "video" / "build" / "resolved_film.json"),
            )
        )
        self.resolved_path = resolved_path
        self.resolved = load_json(resolved_path)
        self.scene_record = next(
            scene
            for scene in self.resolved["timeline"]["scenes"]
            if scene["id"] == self.scene_id
        )
        self.segment_records = {
            segment["id"]: segment
            for segment in self.resolved["timeline"]["segments"]
            if segment["scene"] == self.scene_id
        }
        self.evidence_index = load_json(
            REPO_ROOT / self.resolved["resolved_assets"]["exported_evidence_index"]["path"]
        )

    def asset(self, name: str) -> Path:
        return resolve_repo_path(self.resolved["resolved_assets"][name]["path"])

    def exported(self, name: str) -> Path:
        from video.manim.evidence import exported_asset

        return exported_asset(self.evidence_index, name)

    def metric(self, name: str):
        return metric_value(self.resolved, name)

    def sample(self, name: str):
        return sample_value(self.resolved, name)

    def segment(self, segment_id: str) -> dict:
        return self.segment_records[segment_id]

    def segment_animation_seconds(self, segment_id: str, *, fraction: float = 0.55) -> float:
        record = self.segment(segment_id)
        duration = float(record["audio_duration_seconds"])
        return max(0.28, duration * fraction)

    def cue(
        self,
        segment_id: str,
        animations: Iterable = (),
        *,
        run_time: float | None = None,
    ) -> None:
        record = self.segment(segment_id)
        pre = float(record["pre_hold_seconds"])
        post = float(record["post_hold_seconds"])
        audio = float(record["audio_duration_seconds"])
        if pre:
            self.wait(pre)
        animation_list = list(animations)
        if animation_list:
            animation_time = min(run_time if run_time is not None else 0.48, audio)
            self.play(
                *animation_list,
                run_time=animation_time,
                rate_func=ease_out_quart,
            )
            if audio > animation_time:
                self.wait(audio - animation_time)
        elif audio:
            self.wait(audio)
        if post:
            self.wait(post)
        if segment_id != self.resolved["timeline"]["segments"][-1]["id"]:
            self.wait(float(self.resolved["timeline"].get("inter_segment_gap_seconds", 0.0)))

    def cue_sequence(
        self,
        segment_id: str,
        steps: Sequence[tuple[Iterable, float]],
    ) -> None:
        """Play a short semantic sequence while preserving the bound cue time."""

        record = self.segment(segment_id)
        pre = float(record["pre_hold_seconds"])
        post = float(record["post_hold_seconds"])
        audio = float(record["audio_duration_seconds"])
        if pre:
            self.wait(pre)
        consumed = 0.0
        for animations, seconds in steps:
            duration = min(float(seconds), max(0.0, audio - consumed))
            if duration <= 0:
                break
            animation_list = list(animations)
            if animation_list:
                self.play(
                    *animation_list,
                    run_time=duration,
                    rate_func=ease_in_out_cubic,
                )
            else:
                self.wait(duration)
            consumed += duration
        if audio > consumed:
            self.wait(audio - consumed)
        if post:
            self.wait(post)
        if segment_id != self.resolved["timeline"]["segments"][-1]["id"]:
            self.wait(float(self.resolved["timeline"].get("inter_segment_gap_seconds", 0.0)))

    def pad_to_resolved_duration(self) -> None:
        target = float(self.scene_record["end_seconds"]) - float(self.scene_record["start_seconds"])
        current = float(self.time)
        if target - current > 1e-4:
            self.wait(target - current)
        elif current - target > 1.0 / 30.0 + 1e-3:
            raise RuntimeError(
                f"scene {self.scene_id} exceeds resolved duration: {current:.4f} > {target:.4f}"
            )
