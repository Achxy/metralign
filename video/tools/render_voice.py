#!/usr/bin/env python3
"""Render sentence-sized narration through Mimika with drop-in replacement support."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    import requests
except ImportError:  # pragma: no cover - exercised only outside the video environment
    requests = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"
MIN_SPEECH_SPEED = 0.5
MAX_SPEECH_SPEED = 2.0


class VoiceError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_speech_speed(value: Any, label: str = "speech_speed") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VoiceError(f"{label} must be finite")
    if not MIN_SPEECH_SPEED <= result <= MAX_SPEECH_SPEED:
        raise VoiceError(
            f"{label} must be between {MIN_SPEECH_SPEED} and {MAX_SPEECH_SPEED}"
        )
    return result


def voice_configuration_sha256(voice: dict[str, Any], speech_speed: float) -> str:
    """Hash the effective provider configuration for one segment.

    Replacing ``speed`` in a copy preserves the exact legacy hash whenever a
    segment uses the global speed, while making a per-segment override part of
    the reuse key.
    """

    effective_voice = dict(voice)
    effective_speed = validated_speech_speed(speech_speed)
    configured_speed = voice.get("speed")
    if (
        configured_speed is None
        or validated_speech_speed(configured_speed, "voice.speed") != effective_speed
    ):
        effective_voice["speed"] = effective_speed
    return sha256_bytes(json.dumps(effective_voice, sort_keys=True).encode("utf-8"))


def load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VoiceError(f"expected a mapping in {path}")
    return value


def stable_seed(base_seed: int, segment_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{segment_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VoiceError(f"command failed: {' '.join(command)}") from exc


def normalize_generated(source: Path, destination: Path, voice: dict[str, Any], audio: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = int(audio["sample_rate_hz"])
    channels = int(audio["channels"])
    loudness = float(audio["integrated_loudness_lufs"])
    peak = float(audio["true_peak_dbtp"])
    leading = float(audio["trim_leading_silence_seconds"])
    trailing = float(audio["trim_trailing_silence_seconds"])
    filter_graph = (
        f"silenceremove=start_periods=1:start_duration=0.02:start_silence={leading}:"
        "start_threshold=-48dB:stop_periods=-1:stop_duration=0.04:"
        f"stop_silence={trailing}:stop_threshold=-48dB,"
        f"aresample={sample_rate},loudnorm=I={loudness}:LRA=7:TP={peak}"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-af",
            filter_graph,
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s24le",
            str(destination),
        ]
    )


class MimikaProvider:
    def __init__(self, voice: dict[str, Any]):
        self.voice = voice
        self.base_url = str(voice["base_url"]).rstrip("/")

    def verify(self) -> dict[str, Any]:
        if requests is None:
            raise VoiceError(
                "requests is required; install the pinned video/requirements.txt"
            )
        try:
            response = requests.get(f"{self.base_url}/api/qwen3/voices", timeout=8)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise VoiceError(
                "Mimika is not reachable. Open Mimika Studio or start its local backend on port 7693."
            ) from exc
        payload = response.json()
        voices = payload.get("voices", [])
        profile = str(self.voice["profile_label"])
        match = next((item for item in voices if item.get("name") == profile), None)
        if match is None:
            raise VoiceError(f"Mimika voice profile {profile!r} is not available")
        return match

    def generate(
        self,
        text: str,
        segment_id: str,
        output: Path,
        *,
        speech_speed: float | None = None,
    ) -> dict[str, Any]:
        if requests is None:
            raise VoiceError(
                "requests is required; install the pinned video/requirements.txt"
            )
        seed = stable_seed(int(self.voice["seed"]), segment_id)
        speed = validated_speech_speed(
            self.voice.get("speed", 1.0) if speech_speed is None else speech_speed,
            f"narration {segment_id}.speech_speed",
        )
        request_payload = {
            "text": text,
            "mode": "clone",
            "voice_name": self.voice["profile_label"],
            "language": self.voice["language"],
            "speed": speed,
            "model_size": self.voice["model_size"],
            "model_quantization": self.voice["model_quantization"],
            "temperature": 0.72,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "seed": seed,
            "unload_after": False,
            "enqueue": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}{self.voice['endpoint']}",
                json=request_payload,
                timeout=240,
            )
            response.raise_for_status()
            result = response.json()
            audio_url = result["audio_url"]
            audio_response = requests.get(f"{self.base_url}{audio_url}", timeout=60)
            audio_response.raise_for_status()
        except (requests.RequestException, KeyError, ValueError) as exc:
            detail = getattr(response, "text", "") if "response" in locals() else ""
            raise VoiceError(f"Mimika generation failed for {segment_id}: {detail[:500]}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio_response.content)
        return {
            "provider": "mimika",
            "adapter": "qwen3_clone",
            "profile": self.voice["profile_label"],
            "model_size": self.voice["model_size"],
            "model_quantization": self.voice["model_quantization"],
            "seed": seed,
            "request": request_payload,
            "mimika_filename": result.get("filename"),
            "mimika_audio_url": audio_url,
            "download_sha256": sha256_bytes(audio_response.content),
        }


def resolved_segments(resolved: dict[str, Any]) -> list[dict[str, Any]]:
    segments = resolved.get("timeline", {}).get("segments")
    if not isinstance(segments, list):
        raise VoiceError("resolved film does not contain a narration timeline")
    return segments


def public_profile_record(profile: dict[str, Any]) -> dict[str, Any]:
    """Retain useful provider metadata without publishing workstation paths."""

    allowed = (
        "name",
        "source",
        "origin_engine",
        "duration_sec",
        "gender",
        "language",
        "source_label",
        "engines_supported",
    )
    return {key: profile[key] for key in allowed if key in profile}


def render_voice(
    manifest_path: Path,
    resolved_path: Path,
    *,
    force: bool = False,
    only: set[str] | None = None,
) -> dict[str, Any]:
    manifest = load_mapping(manifest_path)
    resolved = load_mapping(resolved_path)
    voice = manifest["voice"]
    audio = manifest["audio"]
    provider: MimikaProvider | None = None
    generated_dir = VIDEO_ROOT / voice["generated_directory"]
    raw_dir = VIDEO_ROOT / "build" / "audio" / "mimika_raw"
    replacement_dir = VIDEO_ROOT / voice["replacement_directory"]
    records: list[dict[str, Any]] = []
    prior_manifest_path = VIDEO_ROOT / voice["manifest"]
    prior: dict[str, Any] = {}
    if prior_manifest_path.is_file():
        prior = load_mapping(prior_manifest_path)
    prior_profile = prior.get("mimika_profile")
    profile_record = (
        public_profile_record(prior_profile)
        if isinstance(prior_profile, dict)
        else {
            "name": voice["profile_label"],
            "source": "canonical replacement profile",
        }
    )
    prior_records = {
        record.get("id"): record
        for record in prior.get("segments", [])
        if isinstance(record, dict)
    }

    for segment in resolved_segments(resolved):
        segment_id = segment["id"]
        if only and segment_id not in only:
            continue
        text_value = segment["text"]
        if not isinstance(text_value, str) or not text_value.strip():
            raise VoiceError(f"narration {segment_id} has invalid canonical text")
        speech_text = segment.get("speech_text", text_value)
        if not isinstance(speech_text, str) or not speech_text.strip():
            raise VoiceError(f"narration {segment_id} has invalid speech_text")
        speech_speed = validated_speech_speed(
            segment.get("speech_speed", voice.get("speed", 1.0)),
            f"narration {segment_id}.speech_speed",
        )
        text_sha = sha256_bytes(text_value.encode("utf-8"))
        speech_text_sha = sha256_bytes(speech_text.encode("utf-8"))
        configuration_sha = voice_configuration_sha256(voice, speech_speed)
        replacement = replacement_dir / f"{segment_id}.wav"
        generated = generated_dir / f"{segment_id}.wav"
        prior_record = prior_records.get(segment_id, {})
        prior_speech_text_sha = prior_record.get(
            "speech_text_sha256", prior_record.get("text_sha256")
        )
        reusable = (
            generated.is_file()
            and prior_speech_text_sha == speech_text_sha
            and prior_record.get("voice_configuration_sha256") == configuration_sha
        )
        generation: dict[str, Any] | None = None
        if not replacement.is_file() and (not reusable or force):
            if provider is None:
                provider = MimikaProvider(voice)
                profile_record = public_profile_record(provider.verify())
            raw_path = raw_dir / f"{segment_id}.wav"
            generation = provider.generate(
                speech_text,
                segment_id,
                raw_path,
                speech_speed=speech_speed,
            )
            normalize_generated(raw_path, generated, voice, audio)
        selected = replacement if replacement.is_file() else generated
        selected_kind = "replacement" if replacement.is_file() else "generated"
        if not selected.is_file():
            raise VoiceError(f"no narration file produced for {segment_id}")
        records.append(
            {
                "id": segment_id,
                "text": text_value,
                "text_sha256": text_sha,
                "speech_text": speech_text,
                "speech_text_sha256": speech_text_sha,
                "speech_speed": speech_speed,
                "selected_kind": selected_kind,
                "selected_path": selected.relative_to(REPO_ROOT).as_posix(),
                "selected_sha256": sha256_file(selected),
                "generated_path": generated.relative_to(REPO_ROOT).as_posix(),
                "generated_sha256": sha256_file(generated) if generated.is_file() else None,
                "replacement_path": replacement.relative_to(REPO_ROOT).as_posix(),
                "replacement_sha256": sha256_file(replacement) if replacement.is_file() else None,
                "voice_configuration_sha256": configuration_sha,
                "generation": generation or prior_record.get("generation"),
            }
        )

    if only:
        untouched = [
            record
            for record in prior.get("segments", [])
            if isinstance(record, dict) and record.get("id") not in only
        ]
        records = untouched + records
        records.sort(key=lambda record: record["id"])
    output = {
        "schema_version": 1,
        "provider": "Mimika Studio",
        "mimika_profile": profile_record,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "resolved_text_sha256": sha256_bytes(
            json.dumps(
                [{"id": record["id"], "text": record["text"]} for record in records],
                sort_keys=True,
            ).encode("utf-8")
        ),
        "resolved_speech_sha256": sha256_bytes(
            json.dumps(
                [
                    {
                        "id": record["id"],
                        "speech_text": record.get("speech_text", record["text"]),
                        "speech_speed": record.get(
                            "speech_speed", voice.get("speed", 1.0)
                        ),
                    }
                    for record in records
                ],
                sort_keys=True,
            ).encode("utf-8")
        ),
        "segments": records,
    }
    prior_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prior_manifest_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=VIDEO_ROOT / "film.yaml")
    parser.add_argument(
        "--resolved", type=Path, default=VIDEO_ROOT / "build" / "resolved_film.json"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = render_voice(
            args.manifest,
            args.resolved,
            force=args.force,
            only=set(args.only) or None,
        )
    except VoiceError as exc:
        print(f"render_voice: {exc}", file=sys.stderr)
        return 2
    print(f"voice manifest: {len(result['segments'])} segments -> video/voice/voice_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
