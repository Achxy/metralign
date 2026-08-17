#!/usr/bin/env python3
"""Extract authoritative QA frames and compose legible contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"


class ContactSheetError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContactSheetError(f"expected JSON object: {path}")
    return value


def repo_path(locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_frame(video: Path, seconds: float, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, seconds):.6f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        "scale=480:-2",
        str(output),
    ]
    if subprocess.run(command).returncode:
        raise ContactSheetError(f"failed to extract {video}@{seconds}")


def font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def compose(entries: list[tuple[Path, str]], output: Path, *, columns: int = 3) -> None:
    thumb_width, thumb_height = 480, 270
    label_height = 42
    rows = math.ceil(len(entries) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#090909")
    draw = ImageDraw.Draw(sheet)
    typeface = font(18)
    for index, (path, caption) in enumerate(entries):
        row, col = divmod(index, columns)
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = col * thumb_width + (thumb_width - image.width) // 2
        y = row * (thumb_height + label_height) + (thumb_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.line(
            (col * thumb_width, y + image.height, (col + 1) * thumb_width, y + image.height),
            fill="#343434",
            width=1,
        )
        draw.text(
            (col * thumb_width + 12, row * (thumb_height + label_height) + thumb_height + 9),
            caption,
            fill="#F1F0EC",
            font=typeface,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def build_contact_sheets(video: Path, scene_manifest: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(scene_manifest)
    frames_dir = output_dir / "frames"
    scene_entries: list[tuple[Path, str]] = []
    for scene in manifest["scenes"]:
        source = repo_path(scene["output"])
        duration = float(scene["resolved_duration_seconds"])
        for label_name, timestamp in (
            ("first", min(0.05, duration / 4)),
            ("middle", duration / 2),
            ("last", max(0.0, duration - 0.06)),
        ):
            output = frames_dir / "scenes" / f"{int(scene['order']):03d}_{scene['id']}_{label_name}.jpg"
            extract_frame(source, timestamp, output)
            scene_entries.append((output, f"{scene['id']} · {label_name} · {timestamp:.2f}s"))
    scene_sheet = output_dir / "contact_sheet_scenes.png"
    compose(scene_entries, scene_sheet, columns=3)

    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        text=True,
    )
    duration = float(probe.strip())
    every_second: list[tuple[Path, str]] = []
    for second in range(int(math.floor(duration)) + 1):
        output = frames_dir / "seconds" / f"second_{second:04d}.jpg"
        extract_frame(video, min(second + 0.01, max(0.0, duration - 0.02)), output)
        every_second.append((output, f"film · {second:03d}s"))
    sheets: list[Path] = []
    page_size = 20
    for page_index in range(math.ceil(len(every_second) / page_size)):
        chunk = every_second[page_index * page_size : (page_index + 1) * page_size]
        page = output_dir / f"contact_sheet_seconds_{page_index + 1:02d}.png"
        compose(chunk, page, columns=4)
        sheets.append(page)
    report = {
        "schema_version": 1,
        "video": video.relative_to(REPO_ROOT).as_posix(),
        "duration_seconds": duration,
        "scene_frame_count": len(scene_entries),
        "one_second_frame_count": len(every_second),
        "scene_contact_sheet": {
            "path": scene_sheet.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(scene_sheet),
        },
        "one_second_contact_sheets": [
            {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": sha256_file(path)}
            for path in sheets
        ],
    }
    report_path = output_dir / "contact_sheet_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=VIDEO_ROOT / "build" / "reports" / "frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_contact_sheets(args.video, args.scene_manifest, args.output_dir)
    except (ContactSheetError, OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"contact_sheet: {exc}", file=sys.stderr)
        return 2
    print(
        f"contact sheets: {report['scene_frame_count']} scene frames / "
        f"{report['one_second_frame_count']} one-second frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

