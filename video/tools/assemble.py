#!/usr/bin/env python3
"""Assemble rendered scenes and the resolved Mimika narration with FFmpeg."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"


class AssemblyError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssemblyError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else REPO_ROOT / path


def run(command: list[str]) -> None:
    completed = subprocess.run(command)
    if completed.returncode:
        raise AssemblyError(f"command failed: {' '.join(command)}")


def voice_stem(resolved: dict[str, Any], output: Path) -> None:
    segments = resolved["timeline"]["segments"]
    duration = float(resolved["timeline"]["duration_seconds"])
    audio = resolved["audio"]
    sample_rate = int(audio["sample_rate_hz"])
    channels = int(audio["channels"])
    if channels != 1:
        raise AssemblyError("the current narration mixer requires the canonical mono policy")
    inputs: list[str] = []
    filters: list[str] = []
    labels: list[str] = []
    for index, segment in enumerate(segments):
        source = repo_path(segment["selected_audio"])
        if not source.is_file():
            raise AssemblyError(f"missing narration audio: {source}")
        inputs.extend(["-i", str(source)])
        delay = int(round(float(segment["caption_start_seconds"]) * 1000))
        label_name = f"a{index}"
        filters.append(
            f"[{index}:a]aresample={sample_rate},aformat=sample_fmts=fltp:channel_layouts=mono,"
            f"adelay={delay}|{delay}[{label_name}]"
        )
        labels.append(f"[{label_name}]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,"
        f"apad=pad_dur={duration},atrim=duration={duration},"
        f"loudnorm=I={audio['integrated_loudness_lufs']}:LRA=7:TP={audio['true_peak_dbtp']}[outa]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outa]",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_s24le",
            str(output),
        ]
    )


def assemble(resolved_path: Path, *, profile: str) -> dict[str, Path]:
    resolved = load_json(resolved_path)
    index_path = VIDEO_ROOT / "build" / "scenes" / profile / "scene_manifest.json"
    scene_index = load_json(index_path)
    scenes = scene_index["scenes"]
    expected_ids = [scene["id"] for scene in resolved["timeline"]["scenes"]]
    if [scene["id"] for scene in scenes] != expected_ids:
        raise AssemblyError("scene manifest does not match resolved scene order")
    profiles = resolved.get("render", {}).get("profiles", {})
    profile_record = profiles.get(profile) if isinstance(profiles, dict) else None
    if not isinstance(profile_record, dict):
        raise AssemblyError(f"resolved render profile is missing: {profile}")
    width = int(profile_record["width"])
    height = int(profile_record["height"])
    fps = int(profile_record["fps"])
    if profile == "preview":
        final_dir = VIDEO_ROOT / "build" / "preview"
        no_vo = final_dir / "Metralign_Preview_NoVO.mp4"
        vo = final_dir / "Metralign_Preview_VO.wav"
        demo = final_dir / "Metralign_Preview_Demo.mp4"
    else:
        output_spec = resolved["outputs"]
        final_dir = VIDEO_ROOT / output_spec["directory"]
        no_vo = final_dir / output_spec["no_voice_video"]
        vo = final_dir / output_spec["voice_stem"]
        demo = final_dir / output_spec["narrated_video"]
    final_dir.mkdir(parents=True, exist_ok=True)
    duration = float(resolved["timeline"]["duration_seconds"])
    frame_duration = round(duration * fps) / fps
    metadata_arguments: list[str] = []
    for key, value in resolved["film"].get("metadata", {}).items():
        metadata_arguments.extend(["-metadata", f"{key}={value}"])

    with tempfile.TemporaryDirectory(prefix="metralign-film-") as temporary:
        temporary_path = Path(temporary)
        concat_file = temporary_path / "scenes.txt"
        concat_file.write_text(
            "".join(f"file '{repo_path(scene['output']).as_posix()}'\n" for scene in scenes),
            encoding="utf-8",
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-vf",
                (
                    f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:#090909,"
                    f"tpad=stop_mode=clone:stop_duration=1,trim=duration={frame_duration}"
                ),
                "-an",
                "-c:v",
                str(profile_record["codec"]),
                "-preset",
                str(profile_record.get("preset", "medium")),
                "-crf",
                str(profile_record["crf"]),
                "-pix_fmt",
                str(profile_record.get("pixel_format", "yuv420p")),
                "-movflags",
                "+faststart",
                *metadata_arguments,
                str(no_vo),
            ]
        )
    voice_stem(resolved, vo)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(no_vo),
            "-i",
            str(vo),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            *metadata_arguments,
            str(demo),
        ]
    )
    assembly_manifest = {
        "schema_version": 1,
        "artifact_type": "metralign.film_assembly",
        "profile": profile,
        "resolved_film_sha256": sha256_file(resolved_path),
        "scene_manifest": {
            "path": index_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(index_path),
        },
        "scene_inputs": [
            {
                "id": scene["id"],
                "path": scene["output"],
                "sha256": sha256_file(repo_path(scene["output"])),
            }
            for scene in scenes
        ],
        "outputs": {
            "no_voice": {
                "path": no_vo.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(no_vo),
            },
            "voice": {
                "path": vo.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(vo),
            },
            "demo": {
                "path": demo.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(demo),
            },
        },
    }
    manifest_path = VIDEO_ROOT / "build" / "reports" / f"assembly_{profile}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(assembly_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"no_voice": no_vo, "voice": vo, "demo": demo}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved", type=Path, default=VIDEO_ROOT / "build" / "resolved_film.json"
    )
    parser.add_argument("--profile", choices=("preview", "final"), default="preview")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = assemble(args.resolved, profile=args.profile)
    except (AssemblyError, OSError, json.JSONDecodeError) as exc:
        print(f"assemble: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
