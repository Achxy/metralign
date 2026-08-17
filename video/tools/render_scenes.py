#!/usr/bin/env python3
"""Render hash-addressed Manim scenes from the resolved film timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"
SCENE_SOURCE = VIDEO_ROOT / "manim" / "scenes" / "film.py"
MANIM_BIN = VIDEO_ROOT / ".venv" / "bin" / "manim"
MANIM_CONFIG = VIDEO_ROOT / "manim" / "manim.cfg"
MANIM_SOURCES = sorted((VIDEO_ROOT / "manim").rglob("*.py"))
RENDER_TOOL = Path(__file__).resolve()


class RenderError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RenderError(f"expected JSON object: {path}")
    return value


def resolved_path(locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else REPO_ROOT / path


def hash_scene(
    resolved: dict[str, Any],
    scene: dict[str, Any],
    *,
    profile_name: str,
    profile: dict[str, Any],
    manim_version: str,
) -> str:
    digest = hashlib.sha256(b"metralign-manim-scene-v2\0")
    digest.update(bytes.fromhex(sha256_file(MANIM_CONFIG)))
    digest.update(bytes.fromhex(sha256_file(RENDER_TOOL)))
    digest.update(profile_name.encode())
    digest.update(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode())
    digest.update(manim_version.encode())
    for source in MANIM_SOURCES:
        digest.update(source.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    digest.update(json.dumps(scene, sort_keys=True, separators=(",", ":")).encode())
    segments = [
        segment
        for segment in resolved["timeline"]["segments"]
        if segment["scene"] == scene["id"]
    ]
    digest.update(json.dumps(segments, sort_keys=True, separators=(",", ":")).encode())
    for asset_name in sorted(scene.get("assets", [])):
        record = resolved["resolved_assets"][asset_name]
        digest.update(asset_name.encode())
        digest.update(str(record.get("sha256")).encode())
    for input_record in resolved.get("input_files", []):
        if input_record.get("role") in {"evidence_asset", "evidence_source"}:
            digest.update(str(input_record.get("path")).encode())
            digest.update(str(input_record.get("sha256")).encode())
    return digest.hexdigest()


def find_rendered(media_dir: Path, output_name: str) -> Path:
    candidates = [
        path
        for path in media_dir.rglob(output_name)
        if "partial_movie_files" not in path.parts and path.is_file()
    ]
    if not candidates:
        raise RenderError(f"Manim did not produce {output_name}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def normalize_scene_clip(
    source: Path,
    destination: Path,
    *,
    frame_count: int,
    profile: dict[str, Any],
) -> None:
    """Encode exactly ``frame_count`` frames, padding only with the last real frame."""

    if frame_count <= 0:
        raise RenderError(f"scene frame count must be positive: {frame_count}")
    fps = int(profile["fps"])
    temporary = destination.with_name(f".{destination.stem}.normalizing.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps},tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frame_count},setpts=PTS-STARTPTS",
        "-an",
        "-c:v",
        str(profile["codec"]),
        "-preset",
        str(profile.get("preset", "medium")),
        "-crf",
        str(profile["crf"]),
        "-pix_fmt",
        str(profile.get("pixel_format", "yuv420p")),
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(command)
    if completed.returncode:
        raise RenderError(f"FFmpeg could not normalize scene clip: {destination.name}")
    temporary.replace(destination)


def render_scenes(
    resolved_file: Path,
    *,
    profile: str,
    only: set[str] | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    resolved = load_json(resolved_file)
    if resolved.get("missing_audio"):
        raise RenderError("resolved film still has missing narration audio")
    profiles = resolved.get("render", {}).get("profiles", {})
    profile_record = profiles.get(profile) if isinstance(profiles, dict) else None
    if not isinstance(profile_record, dict):
        raise RenderError(f"resolved render profile is missing: {profile}")
    width = int(profile_record["width"])
    height = int(profile_record["height"])
    fps = int(profile_record["fps"])
    quality_names = {
        "low_quality": "l",
        "medium_quality": "m",
        "high_quality": "h",
        "production_quality": "p",
        "fourk_quality": "k",
    }
    quality = quality_names.get(str(profile_record.get("quality")))
    if quality is None:
        raise RenderError(f"unknown Manim quality in profile {profile}: {profile_record.get('quality')}")
    manim_version = subprocess.check_output([str(MANIM_BIN), "--version"], text=True).strip()
    output_dir = VIDEO_ROOT / "build" / "scenes" / profile
    media_dir = VIDEO_ROOT / "build" / "manim" / profile
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "scene_manifest.json"
    previous = load_json(index_path) if index_path.is_file() else {"scenes": []}
    previous_by_id = {
        record["id"]: record
        for record in previous.get("scenes", [])
        if isinstance(record, dict) and "id" in record
    }
    rendered: list[dict[str, Any]] = []
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    homebrew_texlive_share = Path("/opt/homebrew/opt/texlive/share")
    if homebrew_texlive_share.is_dir():
        homebrew_bin = Path("/opt/homebrew/bin")
        if homebrew_bin.is_dir():
            env["PATH"] = f"{homebrew_bin}:{env.get('PATH', '')}"
        env.setdefault(
            "TEXMFCNF",
            str(homebrew_texlive_share / "texmf-dist" / "web2c"),
        )
        env.setdefault("TEXMFROOT", str(homebrew_texlive_share))
        env.setdefault("TEXMFDIST", str(homebrew_texlive_share / "texmf-dist"))
        env.setdefault("TEXMFLOCAL", str(homebrew_texlive_share / "texmf-local"))
    env["METRALIGN_RESOLVED_FILM"] = str(resolved_file.resolve())

    for scene in resolved["timeline"]["scenes"]:
        scene_id = scene["id"]
        if only and scene_id not in only:
            if scene_id in previous_by_id:
                rendered.append(previous_by_id[scene_id])
            continue
        scene_hash = hash_scene(
            resolved,
            scene,
            profile_name=profile,
            profile=profile_record,
            manim_version=manim_version,
        )
        canonical = output_dir / f"{int(scene['order']):03d}_{scene_id}.mp4"
        prior = previous_by_id.get(scene_id)
        if (
            not force
            and prior
            and prior.get("scene_hash") == scene_hash
            and canonical.is_file()
            and prior.get("output_sha256") == sha256_file(canonical)
        ):
            print(f"reuse {scene_id}: {scene_hash[:12]}")
            rendered.append(prior)
            continue
        output_name = f"{scene_id}.mp4"
        command = [
            str(MANIM_BIN),
            "render",
            "--config_file",
            str(MANIM_CONFIG),
            "--renderer",
            str(resolved.get("render", {}).get("renderer", "cairo")),
            "--media_dir",
            str(media_dir),
            "--progress_bar",
            "none",
            "--verbosity",
            "warning",
            "-q",
            quality,
            "-r",
            f"{width},{height}",
            "--fps",
            str(fps),
            "-o",
            output_name,
        ]
        if profile == "final" or force:
            command.append("--disable_caching")
        command.extend([str(SCENE_SOURCE), str(scene["class"])])
        print(f"render {scene_id} [{profile}] {scene_hash[:12]}")
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        if completed.returncode:
            raise RenderError(f"Manim failed for scene {scene_id}")
        source = find_rendered(media_dir, output_name)
        start_frame = round(float(scene["start_seconds"]) * fps)
        end_frame = round(float(scene["end_seconds"]) * fps)
        normalize_scene_clip(
            source,
            canonical,
            frame_count=end_frame - start_frame,
            profile=profile_record,
        )
        record = {
            "id": scene_id,
            "order": scene["order"],
            "class": scene["class"],
            "profile": profile,
            "scene_hash": scene_hash,
            "output": canonical.relative_to(REPO_ROOT).as_posix(),
            "output_sha256": sha256_file(canonical),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": end_frame - start_frame,
            "resolved_duration_seconds": round(
                float(scene["end_seconds"]) - float(scene["start_seconds"]), 6
            ),
        }
        rendered.append(record)
    rendered.sort(key=lambda record: record["order"])
    index = {
        "schema_version": 1,
        "profile": profile,
        "resolved_film_sha256": sha256_file(resolved_file),
        "manim_version": manim_version,
        "render_profile": profile_record,
        "scenes": rendered,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolved", type=Path, default=VIDEO_ROOT / "build" / "resolved_film.json"
    )
    parser.add_argument("--profile", choices=("preview", "final"), default="preview")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = render_scenes(
            args.resolved,
            profile=args.profile,
            only=set(args.only) or None,
            force=args.force,
        )
    except (RenderError, OSError, json.JSONDecodeError) as exc:
        print(f"render_scenes: {exc}", file=sys.stderr)
        return 2
    print(f"scene manifest: {len(records)} scenes [{args.profile}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
