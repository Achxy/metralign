#!/usr/bin/env python3
"""Generate the final build report from resolved manifests and ffprobe output."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:format_tags:stream=codec_name,codec_type,width,height,avg_frame_rate,sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def build_report(resolved_path: Path, output: Path) -> str:
    resolved = load_json(resolved_path)
    output_spec = resolved["outputs"]
    final_dir = VIDEO_ROOT / output_spec["directory"]
    demo = final_dir / output_spec["narrated_video"]
    no_voice = final_dir / output_spec["no_voice_video"]
    voice = final_dir / output_spec["voice_stem"]
    captions = final_dir / output_spec["captions"]
    scene_manifest = load_json(VIDEO_ROOT / "build" / "scenes" / "final" / "scene_manifest.json")
    qa_report = load_json(VIDEO_ROOT / "build" / "reports" / "film_qa.json")
    audio_qa_report = load_json(VIDEO_ROOT / "build" / "reports" / "audio_qa.json")
    if not qa_report.get("passed"):
        raise ValueError("final film QA has not passed; refusing to write a success report")
    if not audio_qa_report.get("passed"):
        raise ValueError("final film audio QA has not passed; refusing to write a success report")
    video_probe = probe(demo)
    video_stream = next(stream for stream in video_probe["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in video_probe["streams"] if stream["codec_type"] == "audio")
    inputs = resolved.get("input_files", [])
    evidence_files = [record for record in inputs if str(record.get("role", "")).startswith("evidence")]
    provenance_files = [record for record in inputs if "provenance" in str(record.get("role", ""))]
    commit = resolved.get("derivation", {}).get("git", {}).get("commit")
    dirty = resolved.get("derivation", {}).get("git", {}).get("dirty")
    duration = float(video_probe["format"]["duration"])
    manim_version = scene_manifest.get("manim_version", "unknown")
    voice_policy = resolved["voice"]
    frozen_pair_count = resolved["resolved_evidence"]["metrics"]["frozen_pair_count"]["value"]
    lines = [
        "# Metralign technical film — build report",
        "",
        "## Delivery",
        "",
        f"- Film duration: `{duration:.3f} s`",
        f"- Video: `{video_stream['width']} × {video_stream['height']}` at `{video_stream['avg_frame_rate']}`",
        f"- Video codec: `{video_stream['codec_name']}`",
        f"- Audio codec: `{audio_stream['codec_name']}` at `{audio_stream['sample_rate']} Hz`, `{audio_stream['channels']}` channel",
        f"- Repository commit: `{commit}` (`dirty={str(dirty).lower()}` at resolution time)",
        f"- Manim: `{manim_version}`",
        f"- Build timestamp: `{datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}`",
        f"- Voice profile: `Mimika Studio / {voice_policy['profile_label']} / Qwen3-TTS {voice_policy['model_size']} clone`",
        f"- Scenes: `{len(scene_manifest['scenes'])}`",
        f"- Narration segments: `{len(resolved['timeline']['segments'])}`",
        "",
        "## Files",
        "",
        f"- `{demo.name}` — SHA-256 `{sha256_file(demo)}`",
        f"- `{no_voice.name}` — SHA-256 `{sha256_file(no_voice)}`",
        f"- `{voice.name}` — SHA-256 `{sha256_file(voice)}`",
        f"- `{captions.name}` — SHA-256 `{sha256_file(captions)}`",
        "",
        "## Evidence and provenance",
        "",
        f"The resolver bound `{len(evidence_files)}` evidence inputs and `{len(provenance_files)}` provenance records. Every bound input path, byte size, role, and SHA-256 is stored in `video/build/resolved_film.json`.",
        "",
        "The film uses project-generated scientific evidence, source-bound licensed microscopy, and official event marks only. No generated scientific image, decorative icon pack, stock people, pseudo-HUD asset, or fabricated terminal frame appears in the film.",
        "",
        "## Claim boundaries",
        "",
        f"- The headline {frozen_pair_count}-pair result is a sealed synthetic stress test.",
        "- Acquired SEM/TEM checks retain their declared digital-crop, publisher-registration, proxy, and fallback limits.",
        "- The separately implemented renderer remains synthetic and shares task-level geometry assumptions.",
        "- The demonstrated failure is an unchanged sealed case; its explanation is limited to recorded diagnostics.",
        "",
        "## QA status",
        "",
        f"Automated numerical, duration, codec, frame geometry, narration, captions, asset, evidence, provenance, metadata, and residue checks passed (`{qa_report['summary']['checks_passed']}/{qa_report['summary']['checks_total']}`). Independent audio gates also passed (`{audio_qa_report['summary']['checks_passed']}/{audio_qa_report['summary']['checks_total']}`). Contact sheets contain the first, middle, and last frame of every scene plus one frame per second for visual review.",
        "",
    ]
    content = "\n".join(lines)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved", type=Path, default=VIDEO_ROOT / "build" / "resolved_film.json")
    parser.add_argument("--output", type=Path, default=VIDEO_ROOT / "final" / "build_report.md")
    args = parser.parse_args(argv)
    build_report(args.resolved, args.output)
    print(f"build report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
