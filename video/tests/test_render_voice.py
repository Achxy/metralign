from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import render_voice  # noqa: E402


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _voice_policy() -> dict:
    return {
        "provider": "mimika",
        "adapter": "qwen3_clone",
        "profile_label": "Fixture",
        "base_url": "http://127.0.0.1:7693",
        "endpoint": "/api/qwen3/generate",
        "language": "English",
        "model_size": "0.6B",
        "model_quantization": "bf16",
        "seed": 1234,
        "speed": 1.0,
        "generated_directory": "voice/generated",
        "replacement_directory": "voice/replacements/fixture",
        "manifest": "voice/voice_manifest.json",
    }


@pytest.fixture
def voice_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    video_root = tmp_path / "video"
    voice = _voice_policy()
    manifest = {"voice": voice, "audio": {}}
    resolved = {
        "timeline": {
            "segments": [
                {
                    "id": "000_test",
                    "text": "Canonical caption text.",
                    "speech_text": "Spoken pronunciation text.",
                    "speech_speed": 0.86,
                }
            ]
        }
    }
    manifest_path = _write_json(video_root / "film.yaml", manifest)
    resolved_path = _write_json(video_root / "build/resolved_film.json", resolved)
    calls: list[dict] = []

    class FakeProvider:
        def __init__(self, provider_voice: dict):
            self.voice = provider_voice

        def verify(self) -> dict:
            return {"name": "Fixture", "source": "unit test"}

        def generate(
            self,
            text: str,
            segment_id: str,
            output: Path,
            *,
            speech_speed: float | None = None,
        ) -> dict:
            calls.append(
                {"text": text, "id": segment_id, "speech_speed": speech_speed}
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"raw:{text}:{speech_speed}".encode())
            return {
                "provider": "fake",
                "request": {"text": text, "speed": speech_speed},
            }

    def fake_normalize(
        source: Path,
        destination: Path,
        _voice: dict,
        _audio: dict,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    monkeypatch.setattr(render_voice, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(render_voice, "VIDEO_ROOT", video_root)
    monkeypatch.setattr(render_voice, "MimikaProvider", FakeProvider)
    monkeypatch.setattr(render_voice, "normalize_generated", fake_normalize)
    return {
        "video_root": video_root,
        "voice": voice,
        "manifest_path": manifest_path,
        "resolved": resolved,
        "resolved_path": resolved_path,
        "calls": calls,
    }


def test_renderer_uses_speech_text_and_records_separate_hashes_and_speed(
    voice_fixture: dict,
):
    result = render_voice.render_voice(
        voice_fixture["manifest_path"], voice_fixture["resolved_path"]
    )

    assert voice_fixture["calls"] == [
        {"text": "Spoken pronunciation text.", "id": "000_test", "speech_speed": 0.86}
    ]
    record = result["segments"][0]
    assert record["text"] == "Canonical caption text."
    assert record["speech_text"] == "Spoken pronunciation text."
    assert record["speech_speed"] == pytest.approx(0.86)
    assert record["text_sha256"] == hashlib.sha256(b"Canonical caption text.").hexdigest()
    assert record["speech_text_sha256"] == hashlib.sha256(
        b"Spoken pronunciation text."
    ).hexdigest()
    assert record["text_sha256"] != record["speech_text_sha256"]
    assert record["voice_configuration_sha256"] == (
        render_voice.voice_configuration_sha256(voice_fixture["voice"], 0.86)
    )
    assert result["resolved_text_sha256"] != result["resolved_speech_sha256"]


def test_speech_text_and_speed_changes_invalidate_generated_audio_reuse(
    voice_fixture: dict,
):
    render_voice.render_voice(voice_fixture["manifest_path"], voice_fixture["resolved_path"])
    render_voice.render_voice(voice_fixture["manifest_path"], voice_fixture["resolved_path"])
    assert len(voice_fixture["calls"]) == 1

    segment = voice_fixture["resolved"]["timeline"]["segments"][0]
    segment["speech_text"] = "A revised spoken form."
    _write_json(voice_fixture["resolved_path"], voice_fixture["resolved"])
    render_voice.render_voice(voice_fixture["manifest_path"], voice_fixture["resolved_path"])
    assert len(voice_fixture["calls"]) == 2

    segment["speech_speed"] = 0.93
    _write_json(voice_fixture["resolved_path"], voice_fixture["resolved"])
    result = render_voice.render_voice(
        voice_fixture["manifest_path"], voice_fixture["resolved_path"]
    )
    assert len(voice_fixture["calls"]) == 3
    assert voice_fixture["calls"][-1]["speech_speed"] == pytest.approx(0.93)
    assert result["segments"][0]["voice_configuration_sha256"] == (
        render_voice.voice_configuration_sha256(voice_fixture["voice"], 0.93)
    )


def test_legacy_resolved_segment_and_prior_manifest_remain_reusable(
    voice_fixture: dict,
):
    voice_fixture["resolved"]["timeline"]["segments"][0].pop("speech_text")
    voice_fixture["resolved"]["timeline"]["segments"][0].pop("speech_speed")
    _write_json(voice_fixture["resolved_path"], voice_fixture["resolved"])
    generated = voice_fixture["video_root"] / "voice/generated/000_test.wav"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"legacy generated audio")
    canonical_text = "Canonical caption text."
    legacy_config_sha = hashlib.sha256(
        json.dumps(voice_fixture["voice"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(
        voice_fixture["video_root"] / "voice/voice_manifest.json",
        {
            "mimika_profile": {"name": "Fixture"},
            "segments": [
                {
                    "id": "000_test",
                    "text": canonical_text,
                    "text_sha256": hashlib.sha256(canonical_text.encode()).hexdigest(),
                    "voice_configuration_sha256": legacy_config_sha,
                    "generation": {"provider": "legacy"},
                }
            ],
        },
    )

    result = render_voice.render_voice(
        voice_fixture["manifest_path"], voice_fixture["resolved_path"]
    )
    assert voice_fixture["calls"] == []
    record = result["segments"][0]
    assert record["speech_text"] == canonical_text
    assert record["speech_speed"] == pytest.approx(1.0)
    assert record["generation"] == {"provider": "legacy"}
    assert record["voice_configuration_sha256"] == legacy_config_sha


def test_mimika_request_uses_per_segment_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    posted: list[dict] = []

    class FakeResponse:
        def __init__(self, payload: dict | None = None, content: bytes = b""):
            self._payload = payload or {}
            self.content = content
            self.text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    def fake_post(url: str, *, json: dict, timeout: int):
        posted.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse({"audio_url": "/audio/test.wav", "filename": "test.wav"})

    monkeypatch.setattr(
        render_voice,
        "requests",
        SimpleNamespace(
            post=fake_post,
            get=lambda *_args, **_kwargs: FakeResponse(content=b"wave bytes"),
            RequestException=RuntimeError,
        ),
    )
    output = tmp_path / "raw.wav"
    result = render_voice.MimikaProvider(_voice_policy()).generate(
        "Spoken text.", "segment", output, speech_speed=0.79
    )
    assert posted[0]["json"]["speed"] == pytest.approx(0.79)
    assert result["request"]["speed"] == pytest.approx(0.79)
    assert output.read_bytes() == b"wave bytes"


@pytest.mark.parametrize("value", [0, 0.49, 2.01, True, "1.0", float("inf")])
def test_renderer_rejects_invalid_speech_speed(value):
    with pytest.raises(render_voice.VoiceError, match="speech_speed"):
        render_voice.validated_speech_speed(value)
