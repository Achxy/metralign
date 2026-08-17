#!/usr/bin/env python3
"""Automated release QA for the hash-bound Metralign technical film."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:  # Supports both ``python video/tools/qa.py`` and package-style imports.
    from captions import CaptionError, captions_match_timeline
    from resolve_film import (
        ResolutionError,
        _load_yaml,
        media_duration,
        probe_media,
        resolve_json_pointer,
        sha256_file,
    )
except ImportError:  # pragma: no cover - exercised by package-style callers
    from .captions import CaptionError, captions_match_timeline
    from .resolve_film import (
        ResolutionError,
        _load_yaml,
        media_duration,
        probe_media,
        resolve_json_pointer,
        sha256_file,
    )


Probe = Callable[[Path], Mapping[str, Any]]
UNRESOLVED_PATTERNS = (
    re.compile(r"\{\{[^{}]+\}\}"),
    re.compile(r"\$\{[^{}]+\}"),
    re.compile(r"<<[^<>]+>>"),
)
PROMPT_RESIDUE_PATTERNS = (
    re.compile(r"\bignore (?:all |any )?(?:previous|prior) instructions\b", re.I),
    re.compile(r"\b(?:system|developer|assistant) (?:prompt|message)\b", re.I),
    re.compile(r"\bchain[- ]of[- ]thought\b", re.I),
    re.compile(r"\binternal reasoning\b", re.I),
    re.compile(r"\b(?:mission|objective)\s*:\s*build\b", re.I),
    re.compile(r"<(?:system|developer|assistant)(?:\s|>)", re.I),
)
NUMBER_LITERAL = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:%|\b)")


class QAError(ValueError):
    """Raised when the QA inputs themselves cannot be parsed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QAError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QAError(f"{label} must be a JSON object")
    return value


def _repo_root(manifest_path: Path) -> Path:
    current = manifest_path.resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return manifest_path.resolve().parent


def _path_for_locator(locator: Any, repo_root: Path) -> Path | None:
    if not isinstance(locator, str) or not locator:
        return None
    path = Path(locator)
    return path if path.is_absolute() else repo_root / path


def _walk_strings(value: Any, prefix: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{prefix}[{index}]")


def _check_tokens(
    documents: Iterable[tuple[str, Any]], forbidden_tokens: Iterable[str]
) -> list[str]:
    errors: list[str] = []
    forbidden = [(token, token.casefold()) for token in forbidden_tokens if isinstance(token, str)]
    for document_name, document in documents:
        for pointer, value in _walk_strings(document):
            for token, folded in forbidden:
                if folded in value.casefold():
                    errors.append(f"{document_name}{pointer}: forbidden token {token!r}")
            for pattern in UNRESOLVED_PATTERNS:
                if pattern.search(value):
                    errors.append(
                        f"{document_name}{pointer}: unresolved template token {pattern.search(value).group(0)!r}"
                    )
            for pattern in PROMPT_RESIDUE_PATTERNS:
                if pattern.search(value):
                    errors.append(f"{document_name}{pointer}: possible prompt residue")
    return errors


def _rate(value: Any) -> float | None:
    if not isinstance(value, str) or value in {"", "0/0", "N/A"}:
        return None
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def _video_stream(probe: Mapping[str, Any]) -> Mapping[str, Any] | None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        return None
    return next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )


def _audio_stream(probe: Mapping[str, Any]) -> Mapping[str, Any] | None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        return None
    return next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )


def _source_value(record: Mapping[str, Any], repo_root: Path) -> tuple[Any, list[str]]:
    errors: list[str] = []
    source = _path_for_locator(record.get("source"), repo_root)
    pointer = record.get("pointer")
    digest = record.get("source_sha256")
    if source is None or not source.is_file():
        return None, [f"missing evidence source {record.get('source')!r}"]
    actual_digest = sha256_file(source)
    if actual_digest != digest:
        errors.append(f"evidence source hash mismatch: {record.get('source')}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        value = resolve_json_pointer(document, pointer)
    except (OSError, json.JSONDecodeError, ResolutionError) as exc:
        errors.append(f"cannot re-resolve {record.get('source')}#{pointer}: {exc}")
        return None, errors
    return value, errors


def _equal_json_values(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) == float(right)
    return left == right


def _provenance_errors(resolved: Mapping[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    if resolved.get("artifact_type") != "metralign.resolved_film":
        errors.append("resolved artifact_type is not metralign.resolved_film")
    generated = resolved.get("generated_at_utc")
    if not isinstance(generated, str):
        errors.append("generated_at_utc is missing")
    else:
        try:
            datetime.fromisoformat(generated.replace("Z", "+00:00"))
        except ValueError:
            errors.append("generated_at_utc is not ISO-8601")
    derivation = resolved.get("derivation")
    git = derivation.get("git") if isinstance(derivation, dict) else None
    commit = git.get("commit") if isinstance(git, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        errors.append("derivation.git.commit is missing or invalid")
    resolver = derivation.get("resolver") if isinstance(derivation, dict) else None
    if not isinstance(resolver, dict) or not re.fullmatch(
        r"[0-9a-f]{64}", str(resolver.get("sha256", ""))
    ):
        errors.append("resolver source provenance is missing")
    else:
        resolver_path = _path_for_locator(resolver.get("path"), repo_root)
        if resolver_path is None or not resolver_path.is_file():
            errors.append("bound resolver source is missing")
        elif sha256_file(resolver_path) != resolver.get("sha256"):
            errors.append("bound resolver source hash mismatch")
    input_files = resolved.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        errors.append("input file hash ledger is missing")
        return errors
    for index, raw_record in enumerate(input_files):
        if not isinstance(raw_record, dict):
            errors.append(f"input_files[{index}] is not a mapping")
            continue
        path = _path_for_locator(raw_record.get("path"), repo_root)
        if path is None or not path.is_file():
            errors.append(f"missing bound input {raw_record.get('path')!r}")
            continue
        if sha256_file(path) != raw_record.get("sha256"):
            errors.append(f"bound input hash mismatch: {raw_record.get('path')}")
    return errors


def _manifest_binding_errors(
    manifest_path: Path, manifest: Mapping[str, Any], resolved: Mapping[str, Any], repo_root: Path
) -> list[str]:
    errors: list[str] = []
    binding = resolved.get("canonical_manifest")
    if not isinstance(binding, dict):
        return ["canonical manifest binding is missing"]
    if binding.get("sha256") != sha256_file(manifest_path):
        errors.append("canonical film.yaml hash mismatch")
    bound_path = _path_for_locator(binding.get("path"), repo_root)
    if bound_path is None or bound_path.resolve() != manifest_path.resolve():
        errors.append("resolved artifact is bound to a different canonical manifest path")
    if manifest.get("schema_version") != resolved.get("schema_version"):
        errors.append("canonical/resolved schema version mismatch")
    for section in ("theme", "audio", "voice", "render", "outputs", "qa"):
        if resolved.get(section) != manifest.get(section):
            errors.append(f"resolved {section} configuration differs from film.yaml")
    return errors


def _asset_errors(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], repo_root: Path
) -> list[str]:
    errors: list[str] = []
    declared = manifest.get("assets")
    assets = resolved.get("resolved_assets")
    if not isinstance(declared, dict) or not isinstance(assets, dict):
        return ["canonical or resolved assets mapping is missing"]
    for name, raw_declaration in declared.items():
        declaration = raw_declaration if isinstance(raw_declaration, dict) else {}
        record = assets.get(name)
        optional = bool(declaration.get("optional", False))
        if not isinstance(record, dict):
            errors.append(f"asset {name} is absent from the resolved artifact")
            continue
        path = _path_for_locator(record.get("path"), repo_root)
        if path is None or not path.is_file():
            if not optional:
                errors.append(f"required asset {name} is missing: {record.get('path')}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"asset hash mismatch: {name}")
        asset_type = str(declaration.get("type", ""))
        source = declaration.get("source")
        requires_provenance = (
            asset_type.startswith(("licensed_", "official_"))
            or isinstance(source, str)
            and source.startswith(("http://", "https://"))
        )
        provenance = record.get("provenance")
        if requires_provenance and not isinstance(provenance, dict):
            errors.append(f"licensed asset {name} has no provenance binding")
        if isinstance(provenance, dict):
            provenance_path = _path_for_locator(provenance.get("path"), repo_root)
            if provenance_path is None or not provenance_path.is_file():
                errors.append(f"asset provenance is missing: {name}")
            elif sha256_file(provenance_path) != provenance.get("sha256"):
                errors.append(f"asset provenance hash mismatch: {name}")
            elif provenance_path.suffix.lower() == ".json" and requires_provenance:
                try:
                    registry = json.loads(provenance_path.read_text(encoding="utf-8"))
                    entries = registry.get("assets") if isinstance(registry, dict) else None
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"asset provenance registry is invalid for {name}: {exc}")
                    entries = None
                if isinstance(entries, list):
                    registry_entry = next(
                        (
                            entry
                            for entry in entries
                            if isinstance(entry, dict) and entry.get("path") == record.get("path")
                        ),
                        None,
                    )
                    if registry_entry is None:
                        errors.append(f"asset {name} is absent from its provenance registry")
                    else:
                        if registry_entry.get("sha256") != record.get("sha256"):
                            errors.append(f"asset registry hash mismatch: {name}")
                        if isinstance(source, str) and source.startswith(("http://", "https://")):
                            if registry_entry.get("source_url") != source:
                                errors.append(f"asset registry source URL mismatch: {name}")
                elif entries is not None:
                    errors.append(f"asset provenance registry has no assets list for {name}")
    for extra in sorted(set(assets) - set(declared)):
        errors.append(f"resolved artifact contains undeclared asset {extra}")
    return errors


def _audio_errors(resolved: Mapping[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    missing = resolved.get("missing_audio")
    if missing:
        errors.append(f"missing narration audio: {missing}")
    timeline = resolved.get("timeline")
    segments = timeline.get("segments") if isinstance(timeline, dict) else None
    if not isinstance(segments, list):
        return errors + ["resolved narration timeline is missing"]
    for record in segments:
        if not isinstance(record, dict):
            errors.append("invalid narration timeline record")
            continue
        segment_id = record.get("id")
        path = _path_for_locator(record.get("selected_audio"), repo_root)
        if path is None or not path.is_file():
            errors.append(f"narration audio missing for {segment_id}")
            continue
        if sha256_file(path) != record.get("selected_audio_sha256"):
            errors.append(f"narration audio hash mismatch for {segment_id}")
        try:
            duration = float(record.get("audio_duration_seconds"))
        except (TypeError, ValueError):
            duration = 0.0
        if not math.isfinite(duration) or duration <= 0:
            errors.append(f"invalid narration duration for {segment_id}")
        if record.get("selected_audio_kind") not in {"replacement", "generated"}:
            errors.append(f"invalid narration audio source for {segment_id}")
    return errors


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _semantic_ledger_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _voice_manifest_errors(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    repo_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Bind rendered narration to the exact semantics used to create it.

    A WAV path and hash alone cannot reveal that the file predates an edit to
    narration text, a pronunciation override, or speech speed. ``render_voice``
    records those inputs in a sidecar manifest; this gate joins that ledger
    one-to-one with the resolved timeline and re-hashes the selected bytes.
    """

    errors: list[str] = []
    details: dict[str, Any] = {}
    canonical_voice = manifest.get("voice")
    resolved_voice = resolved.get("voice")
    if not isinstance(canonical_voice, dict):
        return ["canonical voice configuration is missing"], details
    if not isinstance(resolved_voice, dict):
        return ["resolved voice configuration is missing"], details

    canonical_locator = canonical_voice.get("manifest")
    resolved_locator = resolved_voice.get("manifest")
    if not isinstance(canonical_locator, str) or not canonical_locator:
        return ["voice.manifest is not configured in film.yaml"], details
    if resolved_locator != canonical_locator:
        errors.append("resolved voice-manifest locator differs from film.yaml")

    # Voice paths are configured relative to video/film.yaml, matching the
    # renderer. Absolute paths remain supported for isolated test/build use.
    locator_path = Path(canonical_locator)
    voice_manifest_path = (
        locator_path
        if locator_path.is_absolute()
        else manifest_path.resolve().parent / locator_path
    )
    details["path"] = str(voice_manifest_path)
    if not voice_manifest_path.is_file():
        return errors + [f"voice manifest is missing: {voice_manifest_path}"], details
    try:
        voice_manifest = _load_json(voice_manifest_path, "voice manifest")
    except QAError as exc:
        return errors + [str(exc)], details
    details["sha256"] = sha256_file(voice_manifest_path)

    if voice_manifest.get("manifest_sha256") != sha256_file(manifest_path):
        errors.append("voice manifest is bound to a different film.yaml revision")

    timeline = resolved.get("timeline")
    resolved_segments = timeline.get("segments") if isinstance(timeline, dict) else None
    raw_records = voice_manifest.get("segments")
    if not isinstance(resolved_segments, list):
        return errors + ["resolved narration timeline is missing"], details
    if not isinstance(raw_records, list):
        return errors + ["voice manifest segment list is missing"], details

    details["resolved_segment_count"] = len(resolved_segments)
    details["voice_record_count"] = len(raw_records)
    timeline_by_id: dict[str, Mapping[str, Any]] = {}
    timeline_order: list[str] = []
    for index, segment in enumerate(resolved_segments):
        if not isinstance(segment, dict):
            errors.append(f"resolved timeline segment {index} is not a mapping")
            continue
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            errors.append(f"resolved timeline segment {index} has no valid id")
            continue
        if segment_id in timeline_by_id:
            errors.append(f"resolved timeline contains duplicate segment id {segment_id}")
            continue
        timeline_by_id[segment_id] = segment
        timeline_order.append(segment_id)

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            errors.append(f"voice manifest segment {index} is not a mapping")
            continue
        segment_id = record.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            errors.append(f"voice manifest segment {index} has no valid id")
            continue
        if segment_id in records_by_id:
            errors.append(f"voice manifest contains duplicate record for {segment_id}")
            continue
        records_by_id[segment_id] = record

    for segment_id in timeline_order:
        if segment_id not in records_by_id:
            errors.append(f"voice manifest is missing record for {segment_id}")
    for segment_id in sorted(set(records_by_id) - set(timeline_by_id)):
        errors.append(f"voice manifest contains extra record {segment_id}")

    expected_text_ledger: list[dict[str, Any]] = []
    expected_speech_ledger: list[dict[str, Any]] = []
    for segment_id in timeline_order:
        segment = timeline_by_id[segment_id]
        text = segment.get("text")
        if not isinstance(text, str):
            errors.append(f"resolved canonical text is invalid for {segment_id}")
            continue
        speech_text = segment.get("speech_text", text)
        if not isinstance(speech_text, str):
            errors.append(f"resolved speech_text is invalid for {segment_id}")
            continue
        speech_speed = segment.get("speech_speed")
        if (
            isinstance(speech_speed, bool)
            or not isinstance(speech_speed, (int, float))
            or not math.isfinite(float(speech_speed))
        ):
            errors.append(f"resolved speech_speed is invalid for {segment_id}")
            continue

        expected_text_ledger.append({"id": segment_id, "text": text})
        expected_speech_ledger.append(
            {
                "id": segment_id,
                "speech_text": speech_text,
                "speech_speed": float(speech_speed),
            }
        )
        record = records_by_id.get(segment_id)
        if record is None:
            continue

        expected_text_sha = _text_sha256(text)
        expected_speech_text_sha = _text_sha256(speech_text)
        if record.get("text") != text:
            errors.append(f"voice manifest canonical text mismatch for {segment_id}")
        if record.get("text_sha256") != expected_text_sha:
            errors.append(f"voice manifest canonical text hash mismatch for {segment_id}")
        if record.get("speech_text") != speech_text:
            errors.append(f"voice manifest speech_text mismatch for {segment_id}")
        if record.get("speech_text_sha256") != expected_speech_text_sha:
            errors.append(f"voice manifest speech_text hash mismatch for {segment_id}")

        record_speed = record.get("speech_speed")
        if (
            isinstance(record_speed, bool)
            or not isinstance(record_speed, (int, float))
            or not math.isfinite(float(record_speed))
            or float(record_speed) != float(speech_speed)
        ):
            errors.append(f"voice manifest speech_speed mismatch for {segment_id}")

        expected_kind = segment.get("selected_audio_kind")
        expected_path = segment.get("selected_audio")
        expected_sha = segment.get("selected_audio_sha256")
        if record.get("selected_kind") != expected_kind:
            errors.append(f"voice manifest selected audio kind mismatch for {segment_id}")
        if record.get("selected_path") != expected_path:
            errors.append(f"voice manifest selected audio path mismatch for {segment_id}")
        if record.get("selected_sha256") != expected_sha:
            errors.append(f"voice manifest selected audio hash mismatch for {segment_id}")

        selected_path = _path_for_locator(expected_path, repo_root)
        if selected_path is None or not selected_path.is_file():
            errors.append(f"voice manifest selected audio is missing for {segment_id}")
        else:
            current_sha = sha256_file(selected_path)
            if current_sha != expected_sha:
                errors.append(f"resolved selected audio bytes changed for {segment_id}")
            if current_sha != record.get("selected_sha256"):
                errors.append(f"voice manifest selected audio bytes changed for {segment_id}")

    if voice_manifest.get("resolved_text_sha256") != _semantic_ledger_sha256(
        expected_text_ledger
    ):
        errors.append("voice manifest canonical text ledger hash mismatch")
    if voice_manifest.get("resolved_speech_sha256") != _semantic_ledger_sha256(
        expected_speech_ledger
    ):
        errors.append("voice manifest speech ledger hash mismatch")
    return errors, details


def _scene_narration_errors(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    raw_scenes = manifest.get("scenes")
    raw_segments = (
        manifest.get("narration", {}).get("segments")
        if isinstance(manifest.get("narration"), dict)
        else None
    )
    timeline = resolved.get("timeline")
    resolved_scenes = timeline.get("scenes") if isinstance(timeline, dict) else None
    resolved_segments = timeline.get("segments") if isinstance(timeline, dict) else None
    if not all(isinstance(value, list) for value in (raw_scenes, raw_segments, resolved_scenes, resolved_segments)):
        return ["canonical or resolved scene/narration list is missing"]
    expected_scene_ids = [scene.get("id") for scene in sorted(raw_scenes, key=lambda s: s.get("order", 0))]
    actual_scene_ids = [scene.get("id") for scene in resolved_scenes]
    if actual_scene_ids != expected_scene_ids:
        errors.append("resolved scene order or coverage does not match film.yaml")
    expected_segments = {segment.get("id"): segment for segment in raw_segments}
    if len(expected_segments) != len(raw_segments):
        errors.append("canonical narration segment IDs are not unique")
    actual_ids = [segment.get("id") for segment in resolved_segments]
    expected_order = [segment_id for scene in sorted(raw_scenes, key=lambda s: s.get("order", 0)) for segment_id in scene.get("narration", [])]
    if actual_ids != expected_order:
        errors.append("resolved narration order or coverage does not match film.yaml")
    for segment in resolved_segments:
        canonical = expected_segments.get(segment.get("id"))
        if not isinstance(canonical, dict):
            continue
        if "text" in canonical and segment.get("text") != canonical.get("text"):
            errors.append(f"resolved narration text differs for {segment.get('id')}")
        if "text_template" in canonical:
            if segment.get("text_template") != canonical.get("text_template"):
                errors.append(f"resolved narration template differs for {segment.get('id')}")
            if segment.get("text_source") != "metric_template":
                errors.append(f"resolved narration source is not metric-bound for {segment.get('id')}")
    return errors


def _contact_sheet_errors(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    repo_root: Path,
    video_path: Path,
    video_duration: float | None,
) -> list[str]:
    policy = manifest.get("qa")
    if not isinstance(policy, dict) or not policy.get("require_contact_sheet"):
        return []
    report_path = _path_for_locator(policy.get("contact_sheet_report"), repo_root)
    if report_path is None or not report_path.is_file():
        return [f"contact-sheet report is missing: {policy.get('contact_sheet_report')!r}"]
    try:
        report = _load_json(report_path, "contact-sheet report")
    except QAError as exc:
        return [str(exc)]
    errors: list[str] = []
    bound_video = _path_for_locator(report.get("video"), repo_root)
    if bound_video is None or bound_video.resolve() != video_path.resolve():
        errors.append("contact-sheet report is bound to a different video")
    scenes = resolved.get("timeline", {}).get("scenes", [])
    expected_scene_frames = len(scenes) * 3 if isinstance(scenes, list) else 0
    if report.get("scene_frame_count") != expected_scene_frames:
        errors.append("contact-sheet report does not contain first/middle/last for every scene")
    if video_duration is not None:
        expected_seconds = int(math.floor(video_duration)) + 1
        if report.get("one_second_frame_count") != expected_seconds:
            errors.append("contact-sheet report does not contain one frame per film second")
    records: list[Mapping[str, Any]] = []
    scene_sheet = report.get("scene_contact_sheet")
    if isinstance(scene_sheet, dict):
        records.append(scene_sheet)
    else:
        errors.append("scene contact sheet binding is missing")
    second_sheets = report.get("one_second_contact_sheets")
    if isinstance(second_sheets, list):
        records.extend(record for record in second_sheets if isinstance(record, dict))
    else:
        errors.append("one-second contact sheet bindings are missing")
    for record in records:
        path = _path_for_locator(record.get("path"), repo_root)
        if path is None or not path.is_file():
            errors.append(f"contact sheet is missing: {record.get('path')!r}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"contact-sheet hash mismatch: {record.get('path')}")
    return errors


def _assembly_errors(
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    resolved_path: Path,
    repo_root: Path,
    video_path: Path,
) -> list[str]:
    policy = manifest.get("qa")
    if not isinstance(policy, dict) or not policy.get("require_assembly_binding"):
        return []
    assembly_path = _path_for_locator(policy.get("assembly_manifest"), repo_root)
    if assembly_path is None or not assembly_path.is_file():
        return [f"assembly manifest is missing: {policy.get('assembly_manifest')!r}"]
    try:
        assembly = _load_json(assembly_path, "assembly manifest")
    except QAError as exc:
        return [str(exc)]
    errors: list[str] = []
    if assembly.get("artifact_type") != "metralign.film_assembly":
        errors.append("assembly artifact type is invalid")
    if assembly.get("profile") != "final":
        errors.append("assembly manifest is not the final profile")
    if assembly.get("resolved_film_sha256") != sha256_file(resolved_path):
        errors.append("assembly is bound to a different resolved film")
    scene_binding = assembly.get("scene_manifest")
    scene_manifest: dict[str, Any] | None = None
    if not isinstance(scene_binding, dict):
        errors.append("assembly scene-manifest binding is missing")
    else:
        scene_path = _path_for_locator(scene_binding.get("path"), repo_root)
        if scene_path is None or not scene_path.is_file():
            errors.append("bound scene manifest is missing")
        elif sha256_file(scene_path) != scene_binding.get("sha256"):
            errors.append("bound scene-manifest hash differs")
        else:
            try:
                scene_manifest = _load_json(scene_path, "scene manifest")
            except QAError as exc:
                errors.append(str(exc))
    expected_scenes = resolved.get("timeline", {}).get("scenes", [])
    if scene_manifest is not None and isinstance(expected_scenes, list):
        records = scene_manifest.get("scenes")
        if not isinstance(records, list):
            errors.append("scene manifest has no scene list")
            records = []
        if [record.get("id") for record in records if isinstance(record, dict)] != [
            scene.get("id") for scene in expected_scenes if isinstance(scene, dict)
        ]:
            errors.append("rendered scene order or coverage differs from the resolved timeline")
        if scene_manifest.get("resolved_film_sha256") != sha256_file(resolved_path):
            errors.append("scene manifest is bound to a different resolved film")
        if scene_manifest.get("render_profile") != resolved.get("render", {}).get("profiles", {}).get("final"):
            errors.append("scene manifest render profile differs from film.yaml")
        fps = int(resolved.get("render", {}).get("profiles", {}).get("final", {}).get("fps", 0))
        for scene, record in zip(expected_scenes, records):
            if not isinstance(scene, dict) or not isinstance(record, dict):
                continue
            path = _path_for_locator(record.get("output"), repo_root)
            if path is None or not path.is_file():
                errors.append(f"rendered scene is missing: {record.get('id')}")
                continue
            if sha256_file(path) != record.get("output_sha256"):
                errors.append(f"rendered scene hash mismatch: {record.get('id')}")
            expected_frames = round(float(scene["end_seconds"]) * fps) - round(
                float(scene["start_seconds"]) * fps
            )
            if record.get("frame_count") != expected_frames:
                errors.append(f"rendered scene frame count mismatch: {record.get('id')}")
    scene_inputs = assembly.get("scene_inputs")
    if scene_manifest is not None and isinstance(scene_inputs, list):
        rendered = scene_manifest.get("scenes", [])
        expected_inputs = [
            {"id": record.get("id"), "path": record.get("output"), "sha256": record.get("output_sha256")}
            for record in rendered
            if isinstance(record, dict)
        ]
        if scene_inputs != expected_inputs:
            errors.append("assembly scene input ledger differs from the scene manifest")
    else:
        errors.append("assembly scene input ledger is missing")
    outputs = assembly.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("assembly output ledger is missing")
        return errors
    output_spec = resolved.get("outputs", {})
    expected_paths = {
        "demo": video_path,
        "no_voice": repo_root / "video" / output_spec.get("directory", "") / output_spec.get("no_voice_video", ""),
        "voice": repo_root / "video" / output_spec.get("directory", "") / output_spec.get("voice_stem", ""),
    }
    for name, expected_path in expected_paths.items():
        record = outputs.get(name)
        if not isinstance(record, dict):
            errors.append(f"assembly output binding is missing: {name}")
            continue
        path = _path_for_locator(record.get("path"), repo_root)
        if path is None or path.resolve() != expected_path.resolve() or not path.is_file():
            errors.append(f"assembly output path is invalid: {name}")
        elif sha256_file(path) != record.get("sha256"):
            errors.append(f"assembly output hash mismatch: {name}")
    return errors


def _evidence_errors(
    manifest: Mapping[str, Any], resolved: Mapping[str, Any], repo_root: Path
) -> tuple[list[str], list[str]]:
    reference_errors: list[str] = []
    numerical_errors: list[str] = []
    evidence = resolved.get("resolved_evidence")
    if not isinstance(evidence, dict):
        return ["resolved evidence mapping is missing"], ["numerical bindings are missing"]
    for section_name in ("metrics", "samples"):
        section = evidence.get(section_name)
        if not isinstance(section, dict):
            reference_errors.append(f"resolved evidence section {section_name} is missing")
            continue
        for name, record in section.items():
            if not isinstance(record, dict):
                reference_errors.append(f"invalid evidence record {section_name}.{name}")
                continue
            value, errors = _source_value(record, repo_root)
            reference_errors.extend(f"{section_name}.{name}: {error}" for error in errors)
            if not _equal_json_values(value, record.get("value")):
                reference_errors.append(
                    f"{section_name}.{name}: resolved value no longer matches source pointer"
                )
            if section_name == "metrics" and isinstance(record.get("value"), (int, float)):
                if isinstance(record.get("value"), bool) or not math.isfinite(float(record["value"])):
                    numerical_errors.append(f"metric {name} is not a finite source-bound number")
                for required in ("source", "source_sha256", "pointer"):
                    if not record.get(required):
                        numerical_errors.append(f"metric {name} lacks {required}")

    claims = evidence.get("claims")
    canonical_claims = manifest.get("claims")
    metrics = evidence.get("metrics")
    if not isinstance(claims, dict) or not isinstance(canonical_claims, dict) or not isinstance(metrics, dict):
        reference_errors.append("claim bindings are incomplete")
    else:
        metric_values = {name: record.get("value") for name, record in metrics.items() if isinstance(record, dict)}
        for name, canonical in canonical_claims.items():
            resolved_claim = claims.get(name)
            if not isinstance(canonical, dict) or not isinstance(resolved_claim, dict):
                reference_errors.append(f"claim {name} is absent or malformed")
                continue
            if "text_template" in canonical:
                try:
                    expected = canonical["text_template"].format_map(metric_values)
                except (KeyError, ValueError) as exc:
                    numerical_errors.append(f"claim {name} cannot resolve its metric tokens: {exc}")
                    continue
                if resolved_claim.get("text") != expected:
                    numerical_errors.append(f"claim {name} text does not match its source metrics")
            elif resolved_claim.get("text") != canonical.get("text"):
                reference_errors.append(f"claim {name} text differs from film.yaml")

    # Authored prose may name values only through templates. Timing/configuration
    # numbers live in typed fields and are intentionally outside this check.
    narration = manifest.get("narration", {})
    segments = narration.get("segments", []) if isinstance(narration, dict) else []
    timeline = resolved.get("timeline", {})
    resolved_segments = timeline.get("segments", []) if isinstance(timeline, dict) else []
    resolved_by_id = {
        segment.get("id"): segment
        for segment in resolved_segments
        if isinstance(segment, dict)
    }
    prose: list[tuple[str, Any]] = []
    for segment in segments if isinstance(segments, list) else []:
        if isinstance(segment, dict):
            segment_id = segment.get("id")
            if "text" in segment:
                prose.append((f"narration.{segment_id}", segment.get("text")))
            if "text_template" in segment:
                template = segment.get("text_template")
                resolved_segment = resolved_by_id.get(segment_id)
                if not isinstance(template, str) or not isinstance(resolved_segment, dict):
                    numerical_errors.append(f"narration {segment_id} has an invalid metric template")
                    continue
                fields: list[str] = []
                try:
                    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
                        if field_name is None:
                            continue
                        if not field_name or any(
                            character in field_name for character in (".", "[", "]")
                        ):
                            raise ValueError(f"invalid field {field_name!r}")
                        if format_spec or conversion:
                            raise ValueError(f"transformed field {field_name!r}")
                        fields.append(field_name)
                    expected = template.format_map(metric_values)
                except (KeyError, ValueError) as exc:
                    numerical_errors.append(
                        f"narration {segment_id} cannot resolve source metrics: {exc}"
                    )
                    continue
                unique_fields = list(dict.fromkeys(fields))
                if not unique_fields:
                    numerical_errors.append(
                        f"narration {segment_id} text_template contains no metric reference"
                    )
                if resolved_segment.get("metric_references") != unique_fields:
                    numerical_errors.append(
                        f"narration {segment_id} metric reference ledger differs from its template"
                    )
                if resolved_segment.get("text") != expected:
                    numerical_errors.append(
                        f"narration {segment_id} text does not match its source metrics"
                    )
    for name, claim in canonical_claims.items() if isinstance(canonical_claims, dict) else []:
        if isinstance(claim, dict) and "text" in claim:
            prose.append((f"claims.{name}.text", claim.get("text")))
    for label, value in prose:
        if isinstance(value, str) and NUMBER_LITERAL.search(value):
            numerical_errors.append(
                f"{label} contains a copied numerical literal; use a source-bound text_template"
            )
    return reference_errors, numerical_errors


def run_qa(
    manifest_path: Path,
    resolved_path: Path,
    video_path: Path,
    captions_path: Path,
    *,
    probe: Probe = probe_media,
) -> dict[str, Any]:
    """Run all machine-checkable final-film gates and return a JSON report."""

    manifest_path = manifest_path.resolve()
    resolved_path = resolved_path.resolve()
    video_path = video_path.resolve()
    captions_path = captions_path.resolve()
    manifest = _load_yaml(manifest_path)
    resolved = _load_json(resolved_path, "resolved film")
    repo_root = _repo_root(manifest_path)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, errors: Iterable[str], details: Mapping[str, Any] | None = None) -> None:
        error_list = list(dict.fromkeys(errors))
        checks[name] = {
            "passed": not error_list,
            "errors": error_list,
            **({"details": dict(details)} if details else {}),
        }

    record("manifest_binding", _manifest_binding_errors(manifest_path, manifest, resolved, repo_root))
    record("provenance", _provenance_errors(resolved, repo_root))
    record("assets", _asset_errors(manifest, resolved, repo_root))
    record("audio_inputs", _audio_errors(resolved, repo_root))
    voice_manifest_errors, voice_manifest_details = _voice_manifest_errors(
        manifest_path, manifest, resolved, repo_root
    )
    record(
        "voice_manifest_binding", voice_manifest_errors, voice_manifest_details
    )
    record("scene_and_narration_coverage", _scene_narration_errors(manifest, resolved))
    evidence_errors, numerical_errors = _evidence_errors(manifest, resolved, repo_root)
    record("evidence", evidence_errors)
    record("numerical_source_binding", numerical_errors)

    caption_text = ""
    caption_errors: list[str] = []
    if not captions_path.is_file():
        caption_errors.append(f"captions do not exist: {captions_path}")
    else:
        try:
            caption_text = captions_path.read_text(encoding="utf-8")
            caption_errors.extend(captions_match_timeline(resolved, caption_text))
        except (OSError, UnicodeError, CaptionError) as exc:
            caption_errors.append(f"cannot validate captions: {exc}")
    record("captions", caption_errors)

    video_errors: list[str] = []
    video_details: dict[str, Any] = {}
    video_probe: Mapping[str, Any] = {}
    if not video_path.is_file():
        video_errors.append(f"final video does not exist: {video_path}")
    else:
        try:
            video_probe = probe(video_path)
            stream = _video_stream(video_probe)
            frame = manifest.get("film", {}).get("frame", {})
            expected_width = int(frame.get("width"))
            expected_height = int(frame.get("height"))
            expected_fps = float(frame.get("fps"))
            if stream is None:
                video_errors.append("ffprobe found no video stream")
            else:
                width, height = stream.get("width"), stream.get("height")
                fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
                video_details.update({"width": width, "height": height, "fps": fps})
                if (width, height) != (expected_width, expected_height):
                    video_errors.append(
                        f"video dimensions are {width}x{height}, expected {expected_width}x{expected_height}"
                    )
                if fps is None or abs(fps - expected_fps) > 1e-6:
                    video_errors.append(f"video fps is {fps}, expected {expected_fps:g}")
            if _audio_stream(video_probe) is None:
                video_errors.append("ffprobe found no audio stream")
            duration = media_duration(video_probe, video_path)
            maximum = float(manifest.get("film", {}).get("maximum_duration_seconds"))
            video_details["duration_seconds"] = duration
            if duration <= 0:
                video_errors.append("video duration must be positive")
            if duration >= maximum:
                video_errors.append(
                    f"video duration {duration:.3f}s is not below {maximum:.3f}s"
                )
        except (ResolutionError, TypeError, ValueError) as exc:
            video_errors.append(f"cannot validate final video: {exc}")
    record("video", video_errors, video_details)
    record(
        "assembly_binding",
        _assembly_errors(manifest, resolved, resolved_path, repo_root, video_path),
    )
    duration_value = video_details.get("duration_seconds")
    record(
        "contact_sheets",
        _contact_sheet_errors(
            manifest,
            resolved,
            repo_root,
            video_path,
            float(duration_value) if isinstance(duration_value, (int, float)) else None,
        ),
    )

    forbidden_tokens = manifest.get("qa", {}).get("forbidden_tokens", [])
    metadata = video_probe.get("format", {}).get("tags", {}) if isinstance(video_probe.get("format"), dict) else {}
    authored_manifest = {key: value for key, value in manifest.items() if key != "qa"}
    rendered_artifact = {key: value for key, value in resolved.items() if key != "qa"}
    token_errors = _check_tokens(
        (
            ("manifest", authored_manifest),
            ("resolved", rendered_artifact),
            ("captions", caption_text),
            ("video_metadata", metadata),
        ),
        forbidden_tokens if isinstance(forbidden_tokens, list) else [],
    )
    record("no_unresolved_tokens_or_prompt_residue", token_errors)

    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": 1,
        "artifact_type": "metralign.film_qa_report",
        "passed": passed,
        "inputs": {
            "manifest": str(manifest_path),
            "resolved_film": str(resolved_path),
            "video": str(video_path),
            "captions": str(captions_path),
        },
        "summary": {
            "checks_passed": sum(check["passed"] for check in checks.values()),
            "checks_total": len(checks),
            "error_count": sum(len(check["errors"]) for check in checks.values()),
        },
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("video/film.yaml"))
    parser.add_argument("--resolved", type=Path, default=Path("video/build/resolved_film.json"))
    parser.add_argument("--video", type=Path, default=Path("video/final/Metralign_Demo.mp4"))
    parser.add_argument("--captions", type=Path, default=Path("video/final/Metralign_Demo.srt"))
    parser.add_argument("--report", type=Path, help="optional machine-readable QA report")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_qa(
            args.manifest,
            args.resolved,
            args.video,
            args.captions,
            probe=lambda path: probe_media(path, ffprobe=args.ffprobe),
        )
    except (QAError, ResolutionError) as exc:
        print(f"film_qa: {exc}", file=sys.stderr)
        return 2
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if report["passed"]:
        print(
            f"PASS: {report['summary']['checks_passed']}/{report['summary']['checks_total']} film QA checks"
        )
        return 0
    print(
        f"FAIL: {report['summary']['error_count']} errors across "
        f"{report['summary']['checks_total'] - report['summary']['checks_passed']} checks",
        file=sys.stderr,
    )
    for name, check in report["checks"].items():
        for error in check["errors"]:
            print(f"- {name}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
