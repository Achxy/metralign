from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import audio_qa  # noqa: E402


def _resolved(duration: float = 10.0) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "metralign.resolved_film",
        "audio": {
            "sample_rate_hz": 48000,
            "channels": 1,
            "integrated_loudness_lufs": -16,
            "true_peak_dbtp": -1.5,
        },
        "outputs": {"directory": "final", "voice_stem": "Metralign_VO.wav"},
        "missing_audio": [],
        "timeline": {
            "duration_seconds": duration,
            "segments": [
                {
                    "id": "000_a",
                    "caption_start_seconds": 0.5,
                    "caption_end_seconds": 4.0,
                    "audio_duration_seconds": 3.5,
                },
                {
                    "id": "000_b",
                    "caption_start_seconds": 4.5,
                    "caption_end_seconds": 9.25,
                    "audio_duration_seconds": 4.75,
                },
            ],
        },
    }


def _write_fixture(tmp_path: Path, resolved: dict | None = None) -> tuple[Path, Path, Path]:
    value = resolved or _resolved()
    resolved_path = tmp_path / "video/build/resolved_film.json"
    resolved_path.parent.mkdir(parents=True)
    resolved_path.write_text(json.dumps(value), encoding="utf-8")
    voice = tmp_path / "video/final/Metralign_VO.wav"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"fixture voice")
    report = tmp_path / "video/build/reports/audio_qa.json"
    return resolved_path, voice, report


def _probe(
    *, sample_rate: int = 48000, channels: int = 1, duration: float = 10.0
):
    def probe(_path: Path) -> dict:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "pcm_s24le",
                    "sample_rate": str(sample_rate),
                    "channels": channels,
                    "channel_layout": "mono" if channels == 1 else "stereo",
                    "duration": f"{duration:.6f}",
                }
            ],
            "format": {"duration": f"{duration:.6f}", "format_name": "wav"},
        }

    return probe


def _analysis(*, loudness: float = -16.1, peak: float = -1.55):
    def analyze(_path: Path, target_loudness: float, target_peak: float) -> dict:
        assert target_loudness == -16
        assert target_peak == -1.5
        return {
            "integrated_loudness_lufs": loudness,
            "true_peak_dbtp": peak,
            "loudness_range_lu": 3.2,
            "threshold_lufs": -26.4,
            "normalization_type": "dynamic",
        }

    return analyze


def test_audio_qa_passes_and_writes_source_resolved_report(tmp_path: Path):
    resolved_path, voice, report_path = _write_fixture(tmp_path)
    report = audio_qa.run_audio_qa(
        resolved_path,
        report_path,
        repo_root=tmp_path,
        probe=_probe(),
        analyzer=_analysis(),
    )

    assert report["passed"] is True
    assert report["summary"] == {"checks_passed": 8, "checks_total": 8, "error_count": 0}
    assert report["inputs"]["voice_stem"] == str(voice)
    assert report["policy"]["sample_rate_hz"] == 48000
    assert report["measurements"]["channels"] == 1
    assert report["measurements"]["integrated_loudness_lufs"] == pytest.approx(-16.1)
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True


def test_audio_qa_reports_format_duration_loudness_and_peak_failures(tmp_path: Path):
    resolved_path, _voice, report_path = _write_fixture(tmp_path)
    report = audio_qa.run_audio_qa(
        resolved_path,
        report_path,
        repo_root=tmp_path,
        probe=_probe(sample_rate=44100, channels=2, duration=9.7),
        analyzer=_analysis(loudness=-19.0, peak=-0.8),
    )

    assert report["passed"] is False
    assert report["checks"]["stream_format"]["passed"] is False
    assert report["checks"]["duration_alignment"]["passed"] is False
    assert report["checks"]["integrated_loudness"]["passed"] is False
    assert report["checks"]["true_peak"]["passed"] is False
    assert report["checks"]["non_silence"]["passed"] is True
    assert report_path.is_file()


def test_audio_qa_rejects_silence_without_writing_nonfinite_json(tmp_path: Path):
    resolved_path, _voice, report_path = _write_fixture(tmp_path)
    report = audio_qa.run_audio_qa(
        resolved_path,
        report_path,
        repo_root=tmp_path,
        probe=_probe(),
        analyzer=_analysis(loudness=float("-inf"), peak=float("-inf")),
    )

    assert report["passed"] is False
    assert report["checks"]["non_silence"]["passed"] is False
    assert report["checks"]["integrated_loudness"]["passed"] is False
    assert report["checks"]["true_peak"]["passed"] is False
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["measurements"]["integrated_loudness_lufs"] is None
    assert persisted["measurements"]["true_peak_dbtp"] is None


def test_parse_loudnorm_output_uses_final_measurement_object():
    stderr = """
    irrelevant prefix
    {"input_i":"-20.00","input_tp":"-4.00","input_lra":"2.00","input_thresh":"-30.00"}
    more output
    {
      "input_i" : "-16.03",
      "input_tp" : "-1.52",
      "input_lra" : "3.10",
      "input_thresh" : "-26.20",
      "normalization_type" : "dynamic"
    }
    """
    measured = audio_qa.parse_loudnorm_output(stderr)
    assert measured["integrated_loudness_lufs"] == pytest.approx(-16.03)
    assert measured["true_peak_dbtp"] == pytest.approx(-1.52)
    assert measured["normalization_type"] == "dynamic"


def test_resolved_voice_path_accepts_video_prefixed_directory(tmp_path: Path):
    resolved = _resolved()
    resolved["outputs"]["directory"] = "video/final"
    assert audio_qa.resolved_voice_path(resolved, tmp_path) == (
        tmp_path / "video/final/Metralign_VO.wav"
    )
