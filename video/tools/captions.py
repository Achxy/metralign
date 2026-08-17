#!/usr/bin/env python3
"""Generate SubRip captions from a resolved Metralign narration timeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


class CaptionError(ValueError):
    """Raised when a resolved timeline cannot produce valid captions."""


def _seconds_to_srt(value: float) -> str:
    if value < 0 or value >= float("inf"):
        raise CaptionError(f"invalid caption timestamp: {value!r}")
    milliseconds = int(round(value * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _srt_to_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise CaptionError(f"invalid SubRip timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise CaptionError(f"invalid SubRip timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0


def _segments(resolved: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    timeline = resolved.get("timeline")
    if not isinstance(timeline, dict):
        raise CaptionError("resolved film has no timeline mapping")
    segments = timeline.get("segments")
    if not isinstance(segments, list):
        raise CaptionError("resolved film has no timeline segment list")
    if not segments:
        raise CaptionError("resolved film contains no narration segments")
    if not all(isinstance(segment, dict) for segment in segments):
        raise CaptionError("every resolved narration segment must be a mapping")
    return segments


def generate_srt(resolved: Mapping[str, Any]) -> str:
    """Return one caption cue per narration segment."""

    cues: list[str] = []
    previous_end = 0.0
    for index, segment in enumerate(_segments(resolved), start=1):
        text = segment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CaptionError(f"narration segment {segment.get('id')!r} has no text")
        try:
            start = float(segment["caption_start_seconds"])
            end = float(segment["caption_end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CaptionError(
                f"narration segment {segment.get('id')!r} has invalid caption times"
            ) from exc
        if end <= start:
            raise CaptionError(
                f"narration segment {segment.get('id')!r} has non-positive audio duration"
            )
        if start + 0.0005 < previous_end:
            raise CaptionError(f"caption overlap at narration segment {segment.get('id')!r}")
        previous_end = end
        normalized_text = " ".join(text.split())
        cues.append(
            f"{index}\n{_seconds_to_srt(start)} --> {_seconds_to_srt(end)}\n{normalized_text}"
        )
    return "\n\n".join(cues) + "\n"


def parse_srt(text: str) -> list[dict[str, Any]]:
    """Parse the intentionally simple one-cue-per-segment SRT representation."""

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    cues: list[dict[str, Any]] = []
    for expected_index, block in enumerate(re.split(r"\n{2,}", normalized), start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            raise CaptionError(f"incomplete caption block {expected_index}")
        try:
            actual_index = int(lines[0])
        except ValueError as exc:
            raise CaptionError(f"invalid caption index {lines[0]!r}") from exc
        if actual_index != expected_index:
            raise CaptionError(
                f"caption index sequence mismatch: expected {expected_index}, got {actual_index}"
            )
        if " --> " not in lines[1]:
            raise CaptionError(f"caption {expected_index} has no SubRip time separator")
        start_text, end_text = lines[1].split(" --> ", 1)
        cues.append(
            {
                "index": actual_index,
                "start_seconds": _srt_to_seconds(start_text),
                "end_seconds": _srt_to_seconds(end_text),
                "text": " ".join(" ".join(lines[2:]).split()),
            }
        )
    return cues


def captions_match_timeline(
    resolved: Mapping[str, Any], caption_text: str, *, tolerance_seconds: float = 0.0011
) -> list[str]:
    """Return human-readable timeline mismatches; an empty list means a match."""

    try:
        cues = parse_srt(caption_text)
        segments = _segments(resolved)
    except CaptionError as exc:
        return [str(exc)]
    errors: list[str] = []
    if len(cues) != len(segments):
        return [f"caption count {len(cues)} does not match narration count {len(segments)}"]
    for cue, segment in zip(cues, segments):
        segment_id = segment.get("id")
        expected_text = " ".join(str(segment.get("text", "")).split())
        if cue["text"] != expected_text:
            errors.append(f"caption text mismatch for {segment_id}")
        for field, cue_field in (
            ("caption_start_seconds", "start_seconds"),
            ("caption_end_seconds", "end_seconds"),
        ):
            try:
                expected = float(segment[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid {field} for {segment_id}")
                continue
            if abs(cue[cue_field] - expected) > tolerance_seconds:
                errors.append(f"caption timestamp mismatch for {segment_id}: {field}")
    return errors


def _load_resolved(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptionError(f"cannot read resolved film {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptionError("resolved film must be a JSON object")
    if value.get("artifact_type") != "metralign.resolved_film":
        raise CaptionError("input is not a Metralign resolved film artifact")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved", type=Path, default=Path("video/build/resolved_film.json")
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        resolved = _load_resolved(args.resolved)
        content = generate_srt(resolved)
        output = args.output
        if output is None:
            outputs = resolved["outputs"]
            output = Path("video") / outputs["directory"] / outputs["captions"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    except CaptionError as exc:
        print(f"captions: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(parse_srt(content))} cues -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
