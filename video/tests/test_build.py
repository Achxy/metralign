from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import build as video_build  # noqa: E402


def test_release_evidence_stage_verifies_sealed_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    video_root = tmp_path / "video"
    exporter = video_root / "tools" / "export_evidence.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# fixture\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(video_build, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(
        video_build,
        "run",
        lambda *arguments: calls.append(tuple(str(value) for value in arguments)),
    )

    video_build.evidence()

    assert calls == [(str(video_build.PYTHON), str(exporter), "--verify")]
