#!/usr/bin/env python3
"""One-command evidence, narration, Manim, assembly, and QA pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_ROOT = REPO_ROOT / "video"
# Keep the venv launcher path intact.  Resolving it follows the symlink back to
# Homebrew's base interpreter and drops the pinned film dependencies.
PYTHON = Path(sys.executable)
RESOLVED = VIDEO_ROOT / "build" / "resolved_film.json"


class BuildError(RuntimeError):
    """Raised when a production stage exits unsuccessfully."""


def final_output(key: str) -> Path:
    try:
        resolved = json.loads(RESOLVED.read_text(encoding="utf-8"))
        outputs = resolved["outputs"]
        return VIDEO_ROOT / outputs["directory"] / outputs[key]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot resolve final output {key!r} from {RESOLVED}") from exc


def run(*arguments: str | Path) -> None:
    command = [str(argument) for argument in arguments]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(REPO_ROOT / "src")))
    env.setdefault("MPLCONFIGDIR", str(VIDEO_ROOT / "build" / "matplotlib"))
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if completed.returncode:
        raise BuildError(f"stage failed with exit code {completed.returncode}: {' '.join(command)}")


def evidence() -> None:
    exporter = VIDEO_ROOT / "tools" / "export_evidence.py"
    if not exporter.is_file():
        raise BuildError("evidence exporter is missing: video/tools/export_evidence.py")
    # Release builds consume the checked-in, hash-bound evidence bundle.  The
    # exporter deliberately refuses to overwrite a non-empty destination, so
    # the production pipeline verifies the sealed bundle instead of mutating
    # it.  New evidence is exported explicitly before the source checkpoint.
    run(PYTHON, exporter, "--verify")


def resolve(*, allow_missing: bool) -> None:
    command: list[str | Path] = [
        PYTHON,
        VIDEO_ROOT / "tools" / "resolve_film.py",
        "--manifest",
        VIDEO_ROOT / "film.yaml",
        "--output",
        RESOLVED,
    ]
    if allow_missing:
        command.append("--allow-missing")
    run(*command)


def voice() -> None:
    # The first resolution supplies metric-bound text to the voice provider.
    # The second measures the resulting WAVs and becomes the authoritative timeline.
    resolve(allow_missing=True)
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "render_voice.py",
        "--manifest",
        VIDEO_ROOT / "film.yaml",
        "--resolved",
        RESOLVED,
    )
    resolve(allow_missing=False)
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "captions.py",
        "--resolved",
        RESOLVED,
        "--output",
        final_output("captions"),
    )


def scenes(profile: str, *, force: bool = False) -> None:
    command: list[str | Path] = [
        PYTHON,
        VIDEO_ROOT / "tools" / "render_scenes.py",
        "--resolved",
        RESOLVED,
        "--profile",
        profile,
    ]
    if force:
        command.append("--force")
    run(*command)


def assemble(profile: str) -> None:
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "assemble.py",
        "--resolved",
        RESOLVED,
        "--profile",
        profile,
    )


def final_audio_qa() -> None:
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "audio_qa.py",
        "--resolved",
        RESOLVED,
        "--report",
        VIDEO_ROOT / "build" / "reports" / "audio_qa.json",
    )


def contact_sheet(profile: str) -> None:
    if profile == "preview":
        video = VIDEO_ROOT / "build" / "preview" / "Metralign_Preview_Demo.mp4"
    else:
        video = final_output("narrated_video")
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "contact_sheet.py",
        "--video",
        video,
        "--scene-manifest",
        VIDEO_ROOT / "build" / "scenes" / profile / "scene_manifest.json",
        "--output-dir",
        VIDEO_ROOT / "build" / "reports" / f"frames_{profile}",
    )


def final_qa() -> None:
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "qa.py",
        "--manifest",
        VIDEO_ROOT / "film.yaml",
        "--resolved",
        RESOLVED,
        "--video",
        final_output("narrated_video"),
        "--captions",
        final_output("captions"),
        "--report",
        VIDEO_ROOT / "build" / "reports" / "film_qa.json",
    )


def build_report() -> None:
    run(
        PYTHON,
        VIDEO_ROOT / "tools" / "build_report.py",
        "--resolved",
        RESOLVED,
        "--output",
        final_output("build_report"),
    )


def full_pipeline(profile: str, *, force: bool = False) -> None:
    evidence()
    voice()
    scenes(profile, force=force)
    assemble(profile)
    if profile == "final":
        final_audio_qa()
    contact_sheet(profile)
    if profile == "final":
        final_qa()
        build_report()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "stage",
        nargs="?",
        default="final",
        choices=("evidence", "voice", "scenes", "preview", "final", "qa"),
    )
    result.add_argument("--profile", choices=("preview", "final"), default="preview")
    result.add_argument("--force", action="store_true", help="rerender scene files")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.stage == "evidence":
            evidence()
        elif args.stage == "voice":
            evidence()
            voice()
        elif args.stage == "scenes":
            resolve(allow_missing=False)
            scenes(args.profile, force=args.force)
        elif args.stage == "preview":
            full_pipeline("preview", force=args.force)
        elif args.stage == "final":
            full_pipeline("final", force=args.force)
        elif args.stage == "qa":
            final_audio_qa()
            contact_sheet("final")
            final_qa()
            build_report()
    except (BuildError, OSError) as exc:
        print(f"video_build: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
