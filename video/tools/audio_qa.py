#!/usr/bin/env python3
"""Verify the resolved Metralign voice stem with FFprobe and EBU R128 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"

DEFAULT_DURATION_TOLERANCE_SECONDS = 0.05
DEFAULT_LOUDNESS_TOLERANCE_LU = 1.0
DEFAULT_TRUE_PEAK_TOLERANCE_DB = 0.2
NON_SILENCE_LOUDNESS_FLOOR_LUFS = -70.0
NON_SILENCE_PEAK_FLOOR_DBTP = -50.0

Probe = Callable[[Path], Mapping[str, Any]]
Analyzer = Callable[[Path, float, float], Mapping[str, Any]]


class AudioQAError(RuntimeError):
    """Raised when an audio QA input or measurement cannot be read."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_resolved(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioQAError(f"cannot read resolved film {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AudioQAError("resolved film must be a JSON object")
    if value.get("artifact_type") != "metralign.resolved_film":
        raise AudioQAError("input is not a Metralign resolved film artifact")
    return value


def resolved_voice_path(resolved: Mapping[str, Any], repo_root: Path = REPO_ROOT) -> Path:
    outputs = resolved.get("outputs")
    if not isinstance(outputs, Mapping):
        raise AudioQAError("resolved film has no outputs mapping")
    directory = outputs.get("directory")
    filename = outputs.get("voice_stem")
    if not isinstance(directory, str) or not directory.strip():
        raise AudioQAError("resolved outputs.directory must be non-empty text")
    if not isinstance(filename, str) or not filename.strip():
        raise AudioQAError("resolved outputs.voice_stem must be non-empty text")
    directory_path = Path(directory)
    if directory_path.is_absolute():
        return directory_path / filename
    if directory_path.parts and directory_path.parts[0] == "video":
        return repo_root / directory_path / filename
    return repo_root / "video" / directory_path / filename


def probe_audio(path: Path, *, executable: str = "ffprobe") -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration,format_name:"
            "stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise AudioQAError(f"cannot execute ffprobe: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise AudioQAError(f"ffprobe failed for {path}: {detail[:500]}")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioQAError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise AudioQAError(f"ffprobe returned a non-object for {path}")
    return value


def _measurement_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not an audio measurement")
    return float(value)


def parse_loudnorm_output(stderr: str) -> dict[str, Any]:
    """Extract the final loudnorm measurement object from FFmpeg stderr."""

    candidates = re.findall(r'\{\s*"input_i"\s*:.*?\}', stderr, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            raw = json.loads(candidate)
            return {
                "integrated_loudness_lufs": _measurement_number(raw["input_i"]),
                "true_peak_dbtp": _measurement_number(raw["input_tp"]),
                "loudness_range_lu": _measurement_number(raw["input_lra"]),
                "threshold_lufs": _measurement_number(raw["input_thresh"]),
                "normalization_type": raw.get("normalization_type"),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise AudioQAError("FFmpeg loudnorm output contains no complete measurement object")


def analyze_loudness(
    path: Path,
    target_loudness_lufs: float,
    target_true_peak_dbtp: float,
    *,
    executable: str = "ffmpeg",
) -> dict[str, Any]:
    filter_graph = (
        f"loudnorm=I={target_loudness_lufs:g}:LRA=7:"
        f"TP={target_true_peak_dbtp:g}:print_format=json"
    )
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        filter_graph,
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise AudioQAError(f"cannot execute ffmpeg: {exc}") from exc
    if completed.returncode:
        raise AudioQAError(f"FFmpeg loudness analysis failed: {completed.stderr[-500:]}")
    return parse_loudnorm_output(completed.stderr)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _duration_from_probe(probe: Mapping[str, Any], stream: Mapping[str, Any]) -> float | None:
    for value in (
        stream.get("duration"),
        probe.get("format", {}).get("duration")
        if isinstance(probe.get("format"), Mapping)
        else None,
    ):
        duration = _finite_number(value)
        if duration is not None and duration > 0:
            return duration
    return None


def _policy(resolved: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw = resolved.get("audio")
    if not isinstance(raw, Mapping):
        return {}, ["resolved film has no audio policy"]
    errors: list[str] = []
    parsed: dict[str, Any] = {}
    for key, conversion in (
        ("sample_rate_hz", int),
        ("channels", int),
        ("integrated_loudness_lufs", float),
        ("true_peak_dbtp", float),
    ):
        try:
            value = conversion(raw[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"audio policy {key} is missing or invalid")
            continue
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"audio policy {key} must be finite")
            continue
        if key in {"sample_rate_hz", "channels"} and value <= 0:
            errors.append(f"audio policy {key} must be positive")
            continue
        parsed[key] = value
    return parsed, errors


def _timeline_errors(resolved: Mapping[str, Any], tolerance_seconds: float) -> list[str]:
    errors: list[str] = []
    if resolved.get("missing_audio"):
        errors.append(f"resolved timeline has missing narration audio: {resolved['missing_audio']}")
    timeline = resolved.get("timeline")
    if not isinstance(timeline, Mapping):
        return errors + ["resolved film has no timeline"]
    total = _finite_number(timeline.get("duration_seconds"))
    segments = timeline.get("segments")
    if total is None or total <= 0:
        errors.append("timeline duration must be finite and positive")
    if not isinstance(segments, list) or not segments:
        return errors + ["timeline contains no narration segments"]
    previous_end = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            errors.append(f"timeline segment {index} is not a mapping")
            continue
        segment_id = str(segment.get("id", index))
        start = _finite_number(segment.get("caption_start_seconds"))
        end = _finite_number(segment.get("caption_end_seconds"))
        audio_duration = _finite_number(segment.get("audio_duration_seconds"))
        if start is None or end is None or audio_duration is None:
            errors.append(f"narration {segment_id} has invalid audio timing")
            continue
        if start < -tolerance_seconds or end <= start:
            errors.append(f"narration {segment_id} has an invalid caption interval")
        if start + tolerance_seconds < previous_end:
            errors.append(f"narration {segment_id} overlaps the preceding narration")
        if abs((end - start) - audio_duration) > tolerance_seconds:
            errors.append(f"narration {segment_id} duration differs from its caption interval")
        if total is not None and end > total + tolerance_seconds:
            errors.append(f"narration {segment_id} ends after the resolved timeline")
        previous_end = max(previous_end, end)
    return errors


def _json_number(value: Any) -> float | None:
    return _finite_number(value)


def run_audio_qa(
    resolved_path: Path,
    report_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    voice_override: Path | None = None,
    probe: Probe | None = None,
    analyzer: Analyzer | None = None,
    ffprobe_executable: str = "ffprobe",
    ffmpeg_executable: str = "ffmpeg",
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    loudness_tolerance_lu: float = DEFAULT_LOUDNESS_TOLERANCE_LU,
    true_peak_tolerance_db: float = DEFAULT_TRUE_PEAK_TOLERANCE_DB,
) -> dict[str, Any]:
    """Run all audio gates, write a JSON report, and return that report."""

    for name, value in (
        ("duration tolerance", duration_tolerance_seconds),
        ("loudness tolerance", loudness_tolerance_lu),
        ("true-peak tolerance", true_peak_tolerance_db),
    ):
        if not math.isfinite(value) or value < 0:
            raise AudioQAError(f"{name} must be finite and non-negative")
    resolved_path = resolved_path.resolve()
    report_path = report_path.resolve()
    resolved = load_resolved(resolved_path)
    policy, policy_errors = _policy(resolved)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, errors: list[str], details: Mapping[str, Any] | None = None) -> None:
        unique = list(dict.fromkeys(errors))
        checks[name] = {
            "passed": not unique,
            "errors": unique,
            **({"details": dict(details)} if details else {}),
        }

    record("policy", policy_errors)
    try:
        voice_path = voice_override.resolve() if voice_override else resolved_voice_path(resolved, repo_root)
        path_errors: list[str] = []
    except AudioQAError as exc:
        voice_path = None
        path_errors = [str(exc)]
    if voice_path is None or not voice_path.is_file():
        path_errors.append(f"voice stem does not exist: {voice_path}")
    record("voice_stem_exists", path_errors)

    timeline_errors = _timeline_errors(resolved, duration_tolerance_seconds)
    record("timeline_integrity", timeline_errors)
    expected_duration = _finite_number(
        resolved.get("timeline", {}).get("duration_seconds")
        if isinstance(resolved.get("timeline"), Mapping)
        else None
    )
    measurements: dict[str, Any] = {}
    stream_errors: list[str] = []
    duration_errors: list[str] = []
    measured_duration: float | None = None

    if voice_path is not None and voice_path.is_file():
        actual_probe = probe or (lambda path: probe_audio(path, executable=ffprobe_executable))
        try:
            probe_value = actual_probe(voice_path)
            streams = probe_value.get("streams")
            audio_streams = (
                [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"]
                if isinstance(streams, list)
                else []
            )
            if len(audio_streams) != 1:
                stream_errors.append(f"voice stem has {len(audio_streams)} audio streams; expected exactly one")
            if audio_streams:
                stream = audio_streams[0]
                sample_rate = int(stream.get("sample_rate", 0) or 0)
                channels = int(stream.get("channels", 0) or 0)
                measurements.update(
                    {
                        "codec_name": stream.get("codec_name"),
                        "format_name": probe_value.get("format", {}).get("format_name")
                        if isinstance(probe_value.get("format"), Mapping)
                        else None,
                        "sample_rate_hz": sample_rate,
                        "channels": channels,
                        "channel_layout": stream.get("channel_layout"),
                    }
                )
                expected_rate = policy.get("sample_rate_hz")
                expected_channels = policy.get("channels")
                if expected_rate is not None and sample_rate != expected_rate:
                    stream_errors.append(
                        f"voice stem sample rate is {sample_rate} Hz; expected {expected_rate} Hz"
                    )
                if expected_channels is not None and channels != expected_channels:
                    stream_errors.append(
                        f"voice stem has {channels} channels; expected {expected_channels}"
                    )
                measured_duration = _duration_from_probe(probe_value, stream)
                measurements["duration_seconds"] = measured_duration
                if measured_duration is None:
                    duration_errors.append("ffprobe returned no positive voice-stem duration")
                elif expected_duration is None:
                    duration_errors.append("resolved timeline has no finite duration")
                elif abs(measured_duration - expected_duration) > duration_tolerance_seconds:
                    duration_errors.append(
                        "voice-stem duration differs from the resolved timeline by "
                        f"{abs(measured_duration - expected_duration):.6f} s"
                    )
        except (AudioQAError, OSError, TypeError, ValueError) as exc:
            stream_errors.append(f"cannot probe voice stem: {exc}")
            duration_errors.append("duration alignment was not measurable")
    else:
        stream_errors.append("audio stream was not probed because the voice stem is missing")
        duration_errors.append("duration alignment was not measured because the voice stem is missing")
    record("stream_format", stream_errors)
    record(
        "duration_alignment",
        duration_errors,
        {
            "timeline_seconds": expected_duration,
            "voice_stem_seconds": measured_duration,
            "tolerance_seconds": duration_tolerance_seconds,
        },
    )

    analysis_errors: list[str] = []
    loudness_errors: list[str] = []
    peak_errors: list[str] = []
    non_silence_errors: list[str] = []
    target_loudness = policy.get("integrated_loudness_lufs")
    target_peak = policy.get("true_peak_dbtp")
    integrated: float | None = None
    peak: float | None = None
    if target_loudness is None or target_peak is None:
        analysis_errors.append("resolved loudness or true-peak policy is unavailable")
    elif voice_path is not None and voice_path.is_file():
        actual_analyzer = analyzer or (
            lambda path, loudness, true_peak: analyze_loudness(
                path, loudness, true_peak, executable=ffmpeg_executable
            )
        )
        try:
            analysis = actual_analyzer(voice_path, float(target_loudness), float(target_peak))
            raw_integrated = analysis.get("integrated_loudness_lufs")
            raw_peak = analysis.get("true_peak_dbtp")
            integrated = _finite_number(raw_integrated)
            peak = _finite_number(raw_peak)
            measurements.update(
                {
                    "integrated_loudness_lufs": _json_number(raw_integrated),
                    "true_peak_dbtp": _json_number(raw_peak),
                    "loudness_range_lu": _json_number(analysis.get("loudness_range_lu")),
                    "threshold_lufs": _json_number(analysis.get("threshold_lufs")),
                    "normalization_type": analysis.get("normalization_type"),
                }
            )
        except (AudioQAError, OSError, TypeError, ValueError) as exc:
            analysis_errors.append(f"cannot measure voice stem loudness: {exc}")
    else:
        analysis_errors.append("loudness was not measured because the voice stem is missing")

    if integrated is None or peak is None:
        non_silence_errors.extend(analysis_errors or ["audio energy measurements are non-finite"])
    elif integrated <= NON_SILENCE_LOUDNESS_FLOOR_LUFS or peak <= NON_SILENCE_PEAK_FLOOR_DBTP:
        non_silence_errors.append(
            f"voice stem is effectively silent ({integrated:.2f} LUFS, {peak:.2f} dBTP)"
        )
    if integrated is None:
        loudness_errors.extend(analysis_errors or ["integrated loudness is non-finite"])
    elif abs(integrated - float(target_loudness)) > loudness_tolerance_lu:
        loudness_errors.append(
            f"integrated loudness is {integrated:.2f} LUFS; target is "
            f"{float(target_loudness):.2f} ± {loudness_tolerance_lu:.2f} LU"
        )
    if peak is None:
        peak_errors.extend(analysis_errors or ["true peak is non-finite"])
    elif peak > float(target_peak) + true_peak_tolerance_db:
        peak_errors.append(
            f"true peak is {peak:.2f} dBTP; ceiling is {float(target_peak):.2f} dBTP "
            f"with {true_peak_tolerance_db:.2f} dB measurement tolerance"
        )
    record(
        "non_silence",
        non_silence_errors,
        {
            "integrated_loudness_floor_lufs": NON_SILENCE_LOUDNESS_FLOOR_LUFS,
            "true_peak_floor_dbtp": NON_SILENCE_PEAK_FLOOR_DBTP,
        },
    )
    record(
        "integrated_loudness",
        loudness_errors,
        {"target_lufs": target_loudness, "tolerance_lu": loudness_tolerance_lu},
    )
    record(
        "true_peak",
        peak_errors,
        {"ceiling_dbtp": target_peak, "measurement_tolerance_db": true_peak_tolerance_db},
    )

    passed = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": 1,
        "artifact_type": "metralign.audio_qa_report",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "passed": passed,
        "inputs": {
            "resolved_film": str(resolved_path),
            "voice_stem": str(voice_path) if voice_path is not None else None,
            "voice_stem_sha256": sha256_file(voice_path)
            if voice_path is not None and voice_path.is_file()
            else None,
        },
        "policy": {
            **policy,
            "duration_tolerance_seconds": duration_tolerance_seconds,
            "loudness_tolerance_lu": loudness_tolerance_lu,
            "true_peak_tolerance_db": true_peak_tolerance_db,
        },
        "measurements": measurements,
        "checks": checks,
        "summary": {
            "checks_passed": sum(bool(check["passed"]) for check in checks.values()),
            "checks_total": len(checks),
            "error_count": sum(len(check["errors"]) for check in checks.values()),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved", type=Path, default=VIDEO_ROOT / "build" / "resolved_film.json"
    )
    parser.add_argument(
        "--report", type=Path, default=VIDEO_ROOT / "build" / "reports" / "audio_qa.json"
    )
    parser.add_argument("--voice", type=Path, help="override the resolved voice-stem output")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--loudness-tolerance-lu", type=float, default=DEFAULT_LOUDNESS_TOLERANCE_LU
    )
    parser.add_argument(
        "--true-peak-tolerance-db", type=float, default=DEFAULT_TRUE_PEAK_TOLERANCE_DB
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_audio_qa(
            args.resolved,
            args.report,
            voice_override=args.voice,
            ffprobe_executable=args.ffprobe,
            ffmpeg_executable=args.ffmpeg,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            loudness_tolerance_lu=args.loudness_tolerance_lu,
            true_peak_tolerance_db=args.true_peak_tolerance_db,
        )
    except (AudioQAError, OSError, ValueError) as exc:
        print(f"audio_qa: {exc}", file=sys.stderr)
        return 2
    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"audio QA {status}: {report['summary']['checks_passed']}/"
        f"{report['summary']['checks_total']} checks -> {args.report}"
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
