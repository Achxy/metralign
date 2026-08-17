#!/usr/bin/env python3
"""Resolve the canonical film manifest into a hash-bound build description.

``film.yaml`` remains the only authored film specification.  This tool reads
its evidence pointers, asset locators, and narration declarations and emits a
derived JSON file.  Evidence values in that file always travel with the JSON
Pointer and SHA-256 of the source from which they were read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import string
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


class ResolutionError(ValueError):
    """Raised when the canonical film cannot be resolved safely."""


Probe = Callable[[Path], Mapping[str, Any]]


# Mimika's speed control is useful for pronunciation and cadence corrections,
# but extreme values make narration unintelligible.  Keep the authored
# per-segment contract deliberately narrow and deterministic.
MIN_SPEECH_SPEED = 0.5
MAX_SPEECH_SPEED = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ResolutionError(
            "PyYAML is required; install the pinned video/requirements.txt"
        ) from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ResolutionError(f"cannot read canonical manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ResolutionError("film.yaml must contain a top-level mapping")
    return value


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"cannot read JSON evidence {path}: {exc}") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResolutionError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ResolutionError(f"{label} must be a list")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolutionError(f"{label} must be numeric")
    result = float(value)
    if result < minimum:
        raise ResolutionError(f"{label} must be at least {minimum}")
    return result


def _speech_speed(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResolutionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ResolutionError(f"{label} must be finite")
    if not MIN_SPEECH_SPEED <= result <= MAX_SPEECH_SPEED:
        raise ResolutionError(
            f"{label} must be between {MIN_SPEECH_SPEED} and {MAX_SPEECH_SPEED}"
        )
    return result


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON Pointer without accepting lookalike syntax."""

    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ResolutionError(f"invalid JSON Pointer: {pointer!r}")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            if token == "-" or not token.isdigit():
                raise ResolutionError(f"invalid array index {token!r} in {pointer!r}")
            index = int(token)
            if index >= len(current):
                raise ResolutionError(f"array index out of range in {pointer!r}")
            current = current[index]
        elif isinstance(current, dict):
            if token not in current:
                raise ResolutionError(f"missing key {token!r} in {pointer!r}")
            current = current[token]
        else:
            raise ResolutionError(f"cannot traverse {pointer!r} through a scalar")
    return current


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ResolutionError(f"command failed ({' '.join(command)}): {detail.strip()}") from exc
    return completed.stdout.strip()


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    """Return ffprobe's machine-readable stream and format metadata."""

    executable = shutil.which(ffprobe)
    if executable is None:
        raise ResolutionError(f"ffprobe executable not found: {ffprobe}")
    output = _run(
        [
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:format_tags:stream=index,codec_type,width,height,avg_frame_rate,r_frame_rate,sample_rate,channels,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ResolutionError(f"ffprobe returned invalid JSON for {path}") from exc
    if not isinstance(value, dict):
        raise ResolutionError(f"ffprobe returned an invalid document for {path}")
    return value


def media_duration(probe: Mapping[str, Any], path: Path) -> float:
    candidates: list[Any] = []
    format_value = probe.get("format")
    if isinstance(format_value, dict):
        candidates.append(format_value.get("duration"))
    streams = probe.get("streams")
    if isinstance(streams, list):
        candidates.extend(
            stream.get("duration") for stream in streams if isinstance(stream, dict)
        )
    for candidate in candidates:
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration >= 0 and duration < float("inf"):
            return duration
    raise ResolutionError(f"ffprobe did not report a finite duration for {path}")


def _repo_root(manifest_path: Path) -> Path:
    try:
        return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=manifest_path.parent))
    except ResolutionError:
        return manifest_path.parent


def _git_binding(repo_root: Path) -> dict[str, Any]:
    try:
        commit = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        status = _run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo_root)
    except ResolutionError:
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def _locator(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _path_from_manifest(base_dir: Path, locator: str, label: str) -> Path:
    if not isinstance(locator, str) or not locator.strip():
        raise ResolutionError(f"{label} must be a non-empty path string")
    return (base_dir / locator).resolve()


def _hash_record(path: Path, repo_root: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _locator(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _utc_timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        try:
            instant = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        except (ValueError, OverflowError) as exc:
            raise ResolutionError("SOURCE_DATE_EPOCH must be an integer timestamp") from exc
    else:
        instant = datetime.now(timezone.utc)
    return instant.isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_project(
    manifest: Mapping[str, Any], base_dir: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    project = _mapping(manifest.get("project"), "project")
    resolved = {
        key: project.get(key)
        for key in (
            "name",
            "technical_title",
            "submission_context",
            "event",
            "repository_url",
            "website_url",
            "license",
            "team",
        )
    }
    source = project.get("version_source")
    if source is None:
        resolved["version"] = None
        return resolved, None
    declaration = _mapping(source, "project.version_source")
    path = _path_from_manifest(base_dir, declaration.get("file"), "project.version_source.file")
    key = declaration.get("key")
    if not path.is_file() or not isinstance(key, str) or not key:
        raise ResolutionError("project.version_source requires an existing file and dotted key")
    try:
        current: Any = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ResolutionError(f"cannot read project version source {path}: {exc}") from exc
    for component in key.split("."):
        if not isinstance(current, dict) or component not in current:
            raise ResolutionError(f"project version key {key!r} is absent from {path}")
        current = current[component]
    if not isinstance(current, str) or not current:
        raise ResolutionError(f"project version key {key!r} is not text")
    resolved["version"] = current
    resolved["version_source"] = {
        "path": _locator(path, repo_root),
        "key": key,
        "sha256": sha256_file(path),
    }
    return resolved, _hash_record(path, repo_root, role="project_version_source")


def _resolve_declared_evidence(
    manifest: Mapping[str, Any], base_dir: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    evidence = _mapping(manifest.get("evidence"), "evidence")
    cache: dict[Path, tuple[Any, str]] = {}
    input_records: dict[str, dict[str, Any]] = {}

    def load_source(locator: str, role: str) -> tuple[Path, Any, str]:
        path = _path_from_manifest(base_dir, locator, role)
        if not path.is_file():
            raise ResolutionError(f"missing {role}: {_locator(path, repo_root)}")
        if path not in cache:
            cache[path] = (_load_json(path), sha256_file(path))
        document, digest = cache[path]
        input_records[_locator(path, repo_root)] = _hash_record(path, repo_root, role=role)
        return path, document, digest

    resolved: dict[str, Any] = {"metrics": {}, "samples": {}, "citations": {}, "claims": {}}
    for section_name in ("metrics", "samples"):
        declarations = _mapping(evidence.get(section_name, {}), f"evidence.{section_name}")
        for name, raw_declaration in declarations.items():
            declaration = _mapping(raw_declaration, f"evidence.{section_name}.{name}")
            source = declaration.get("source")
            pointer = declaration.get("pointer")
            if not isinstance(source, str) or not isinstance(pointer, str):
                raise ResolutionError(
                    f"evidence.{section_name}.{name} requires source and pointer"
                )
            path, document, digest = load_source(source, f"evidence_{section_name}")
            value = resolve_json_pointer(document, pointer)
            resolved[section_name][name] = {
                "source": _locator(path, repo_root),
                "source_sha256": digest,
                "pointer": pointer,
                "value": value,
            }

    citations = _mapping(evidence.get("citations", {}), "evidence.citations")
    for name, raw_citation in citations.items():
        citation = _mapping(raw_citation, f"evidence.citations.{name}")
        source = citation.get("source")
        if not isinstance(source, str):
            raise ResolutionError(f"evidence.citations.{name}.source must be a path")
        path = _path_from_manifest(base_dir, source, f"citation {name}")
        if not path.is_file():
            raise ResolutionError(f"missing citation source: {_locator(path, repo_root)}")
        input_records[_locator(path, repo_root)] = _hash_record(
            path, repo_root, role="citation_source"
        )
        resolved["citations"][name] = {
            "label": citation.get("label"),
            "locator": citation.get("locator"),
            "source": _locator(path, repo_root),
            "source_sha256": sha256_file(path),
        }

    metric_values = {
        name: record["value"] for name, record in resolved["metrics"].items()
    }
    claims = _mapping(manifest.get("claims", {}), "claims")
    for name, raw_claim in claims.items():
        claim = _mapping(raw_claim, f"claims.{name}")
        text = claim.get("text")
        template = claim.get("text_template")
        if (text is None) == (template is None):
            raise ResolutionError(f"claim {name} must define exactly one of text or text_template")
        if template is not None:
            if not isinstance(template, str):
                raise ResolutionError(f"claim {name}.text_template must be text")
            try:
                rendered = template.format_map(metric_values)
            except (KeyError, ValueError) as exc:
                raise ResolutionError(f"claim {name} has an unresolved metric token: {exc}") from exc
        elif isinstance(text, str):
            rendered = text
        else:
            raise ResolutionError(f"claim {name}.text must be text")
        references = _list(claim.get("evidence", []), f"claims.{name}.evidence")
        resolved["claims"][name] = {"text": rendered, "references": references}

    return resolved, input_records


def _validate_reference(
    token: str,
    *,
    manifest: Mapping[str, Any],
    resolved_evidence: Mapping[str, Any],
    base_dir: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    if not isinstance(token, str) or ":" not in token:
        raise ResolutionError(f"invalid evidence reference: {token!r}")
    kind, target = token.split(":", 1)
    section_for_kind = {
        "metric": "metrics",
        "sample": "samples",
        "citation": "citations",
        "claim": "claims",
    }
    if kind in section_for_kind:
        section = _mapping(
            resolved_evidence.get(section_for_kind[kind]),
            f"resolved evidence {section_for_kind[kind]}",
        )
        if target not in section:
            raise ResolutionError(f"unknown evidence reference: {token}")
        return None
    if kind in {"asset", "source"}:
        path = _path_from_manifest(base_dir, target, token)
        if not path.is_file():
            raise ResolutionError(f"missing referenced {kind}: {_locator(path, repo_root)}")
        return _hash_record(path, repo_root, role=f"evidence_{kind}")
    raise ResolutionError(f"unknown evidence reference kind: {kind}")


def _resolve_assets(
    manifest: Mapping[str, Any], base_dir: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    declarations = _mapping(manifest.get("assets", {}), "assets")
    resolved: dict[str, Any] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for name, raw_asset in declarations.items():
        asset = _mapping(raw_asset, f"assets.{name}")
        path = _path_from_manifest(base_dir, asset.get("file"), f"asset {name}")
        optional = bool(asset.get("optional", False))
        record: dict[str, Any] = {
            "type": asset.get("type"),
            "source": asset.get("source"),
            "modified": asset.get("modified"),
            "optional": optional,
            "path": _locator(path, repo_root),
            "exists": path.is_file(),
        }
        if path.is_file():
            hashed = _hash_record(path, repo_root, role="asset")
            record.update({"sha256": hashed["sha256"], "bytes": hashed["bytes"]})
            inputs[record["path"]] = hashed
        elif not optional:
            raise ResolutionError(f"missing required asset {name}: {record['path']}")
        provenance = asset.get("provenance")
        if provenance is not None:
            provenance_path = _path_from_manifest(
                base_dir, provenance, f"asset {name} provenance"
            )
            if not provenance_path.is_file():
                raise ResolutionError(
                    f"missing provenance for asset {name}: {_locator(provenance_path, repo_root)}"
                )
            provenance_record = _hash_record(
                provenance_path, repo_root, role="asset_provenance"
            )
            record["provenance"] = provenance_record
            inputs[provenance_record["path"]] = provenance_record
        resolved[name] = record
    return resolved, inputs


def _resolve_audio_and_timeline(
    manifest: Mapping[str, Any],
    base_dir: Path,
    repo_root: Path,
    *,
    metric_values: Mapping[str, Any],
    probe: Probe,
    allow_missing: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    narration = _mapping(manifest.get("narration"), "narration")
    raw_segments = _list(narration.get("segments"), "narration.segments")
    audio = _mapping(manifest.get("audio"), "audio")
    voice = _mapping(manifest.get("voice"), "voice")
    gap = _number(audio.get("inter_segment_gap_seconds", 0), "audio.inter_segment_gap_seconds")
    generated_dir = _path_from_manifest(
        base_dir, voice.get("generated_directory"), "voice.generated_directory"
    )
    replacement_dir = _path_from_manifest(
        base_dir, voice.get("replacement_directory"), "voice.replacement_directory"
    )
    precedence = bool(voice.get("replacement_precedence", False))
    default_speech_speed = _speech_speed(voice.get("speed", 1.0), "voice.speed")

    segments_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_segment in enumerate(raw_segments):
        segment = _mapping(raw_segment, f"narration.segments[{index}]")
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            raise ResolutionError(f"narration.segments[{index}].id must be text")
        if segment_id in segments_by_id:
            raise ResolutionError(f"duplicate narration segment id: {segment_id}")
        text = segment.get("text")
        template = segment.get("text_template")
        if (text is None) == (template is None):
            raise ResolutionError(
                f"narration segment {segment_id} must define exactly one of text or text_template"
            )
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ResolutionError(f"narration segment {segment_id} has invalid text")
        if template is not None and (not isinstance(template, str) or not template.strip()):
            raise ResolutionError(f"narration segment {segment_id} has invalid text_template")
        if "speech_text" in segment:
            speech_text = segment["speech_text"]
            if not isinstance(speech_text, str) or not speech_text.strip():
                raise ResolutionError(f"narration segment {segment_id} has invalid speech_text")
        if "speech_speed" in segment:
            _speech_speed(
                segment["speech_speed"], f"narration segment {segment_id}.speech_speed"
            )
        segments_by_id[segment_id] = segment

    raw_scenes = _list(manifest.get("scenes"), "scenes")
    scene_ids: set[str] = set()
    orders: set[int] = set()
    ordered_scenes: list[dict[str, Any]] = []
    used_segments: list[str] = []
    for index, raw_scene in enumerate(raw_scenes):
        scene = _mapping(raw_scene, f"scenes[{index}]")
        scene_id = scene.get("id")
        order = scene.get("order")
        if not isinstance(scene_id, str) or not scene_id:
            raise ResolutionError(f"scenes[{index}].id must be text")
        if not isinstance(order, int) or isinstance(order, bool):
            raise ResolutionError(f"scene {scene_id}.order must be an integer")
        if scene_id in scene_ids or order in orders:
            raise ResolutionError(f"duplicate scene id or order: {scene_id}/{order}")
        scene_ids.add(scene_id)
        orders.add(order)
        narration_ids = _list(scene.get("narration"), f"scene {scene_id}.narration")
        for segment_id in narration_ids:
            if segment_id not in segments_by_id:
                raise ResolutionError(f"scene {scene_id} references unknown narration {segment_id}")
            if segments_by_id[segment_id].get("scene") != scene_id:
                raise ResolutionError(
                    f"narration {segment_id} declares scene {segments_by_id[segment_id].get('scene')!r}, not {scene_id!r}"
                )
            used_segments.append(segment_id)
        ordered_scenes.append(scene)
    if len(used_segments) != len(set(used_segments)):
        raise ResolutionError("a narration segment is assigned to more than one scene")
    missing_assignments = sorted(set(segments_by_id) - set(used_segments))
    if missing_assignments:
        raise ResolutionError(f"narration segments absent from scenes: {missing_assignments}")

    ordered_scenes.sort(key=lambda value: value["order"])
    cursor = 0.0
    timeline_segments: list[dict[str, Any]] = []
    timeline_scenes: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    missing_audio: list[str] = []
    for scene_index, scene in enumerate(ordered_scenes):
        scene_start = cursor
        scene_segment_ids: list[str] = []
        for narration_index, segment_id in enumerate(scene["narration"]):
            segment = segments_by_id[segment_id]
            template = segment.get("text_template")
            metric_references: list[str] = []
            if template is not None:
                try:
                    for _, field_name, format_spec, conversion in string.Formatter().parse(template):
                        if field_name is None:
                            continue
                        if not field_name or any(
                            character in field_name for character in (".", "[", "]")
                        ):
                            raise ResolutionError(
                                f"narration {segment_id} uses a non-metric template field {field_name!r}"
                            )
                        if format_spec or conversion:
                            raise ResolutionError(
                                f"narration {segment_id} may not transform source metric {field_name!r}"
                            )
                        if field_name not in metric_values:
                            raise ResolutionError(
                                f"narration {segment_id} references unknown metric {field_name!r}"
                            )
                        metric_references.append(field_name)
                    rendered_text = template.format_map(metric_values)
                except (KeyError, ValueError) as exc:
                    raise ResolutionError(
                        f"narration {segment_id} has an unresolved metric token: {exc}"
                    ) from exc
            else:
                rendered_text = segment["text"]
            # Canonical text remains the source for captions and claims.  These
            # two resolved fields affect synthesis only and default exactly to
            # the legacy behaviour when no override is authored.
            speech_text = segment.get("speech_text", rendered_text)
            speech_speed = _speech_speed(
                segment.get("speech_speed", default_speech_speed),
                f"narration segment {segment_id}.speech_speed",
            )
            replacement = replacement_dir / f"{segment_id}.wav"
            generated = generated_dir / f"{segment_id}.wav"
            if precedence and replacement.is_file():
                selected, selected_kind = replacement, "replacement"
            elif generated.is_file():
                selected, selected_kind = generated, "generated"
            elif replacement.is_file():
                selected, selected_kind = replacement, "replacement"
            else:
                selected, selected_kind = generated, "missing"
                missing_audio.append(segment_id)

            pre_hold = _number(
                segment.get("pre_hold_seconds", 0), f"narration {segment_id}.pre_hold_seconds"
            )
            post_hold = _number(
                segment.get("post_hold_seconds", 0), f"narration {segment_id}.post_hold_seconds"
            )
            audio_duration = 0.0
            audio_record: dict[str, Any] | None = None
            if selected.is_file():
                audio_duration = media_duration(probe(selected), selected)
                audio_record = _hash_record(selected, repo_root, role="narration_audio")
                inputs[audio_record["path"]] = audio_record
            elif not allow_missing:
                raise ResolutionError(
                    f"missing narration audio {segment_id}; checked {_locator(replacement, repo_root)} and {_locator(generated, repo_root)}"
                )

            audio_start = cursor + pre_hold
            audio_end = audio_start + audio_duration
            segment_end = audio_end + post_hold
            timeline_record = {
                "id": segment_id,
                "scene": scene["id"],
                "text": rendered_text,
                "speech_text": speech_text,
                "speech_speed": speech_speed,
                "text_source": "metric_template" if template is not None else "literal",
                "text_template": template,
                "metric_references": list(dict.fromkeys(metric_references)),
                "visual_action": segment.get("visual_action"),
                "selected_audio_kind": selected_kind,
                "selected_audio": _locator(selected, repo_root),
                "selected_audio_sha256": audio_record["sha256"] if audio_record else None,
                "audio_duration_seconds": round(audio_duration, 6),
                "pre_hold_seconds": pre_hold,
                "post_hold_seconds": post_hold,
                "segment_start_seconds": round(cursor, 6),
                "caption_start_seconds": round(audio_start, 6),
                "caption_end_seconds": round(audio_end, 6),
                "segment_end_seconds": round(segment_end, 6),
            }
            timeline_segments.append(timeline_record)
            scene_segment_ids.append(segment_id)
            is_last = (
                scene_index == len(ordered_scenes) - 1
                and narration_index == len(scene["narration"]) - 1
            )
            cursor = segment_end + (0.0 if is_last else gap)
        timeline_scenes.append(
            {
                "id": scene["id"],
                "order": scene["order"],
                "class": scene.get("class"),
                "title": scene.get("title"),
                "transition_out": scene.get("transition_out"),
                "assets": scene.get("assets", []),
                "evidence": scene.get("evidence", []),
                "narration": scene_segment_ids,
                "start_seconds": round(scene_start, 6),
                "end_seconds": round(cursor, 6),
            }
        )

    return (
        {
            "duration_seconds": round(cursor, 6),
            "inter_segment_gap_seconds": gap,
            "scenes": timeline_scenes,
            "segments": timeline_segments,
        },
        inputs,
        missing_audio,
    )


def resolve_film(
    manifest_path: Path,
    *,
    probe: Probe = probe_media,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Resolve and validate ``film.yaml`` without modifying the canonical file."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ResolutionError(f"canonical manifest does not exist: {manifest_path}")
    manifest = _load_yaml(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ResolutionError(f"unsupported film schema_version: {manifest.get('schema_version')!r}")
    base_dir = manifest_path.parent
    repo_root = _repo_root(manifest_path)
    resolved_project, project_input = _resolve_project(manifest, base_dir, repo_root)

    resolved_evidence, evidence_inputs = _resolve_declared_evidence(
        manifest, base_dir, repo_root
    )
    resolved_assets, asset_inputs = _resolve_assets(manifest, base_dir, repo_root)
    timeline, audio_inputs, missing_audio = _resolve_audio_and_timeline(
        manifest,
        base_dir,
        repo_root,
        metric_values={
            name: record["value"]
            for name, record in resolved_evidence["metrics"].items()
        },
        probe=probe,
        allow_missing=allow_missing,
    )

    referenced_inputs: dict[str, dict[str, Any]] = {}
    for claim in resolved_evidence["claims"].values():
        for token in claim["references"]:
            record = _validate_reference(
                token,
                manifest=manifest,
                resolved_evidence=resolved_evidence,
                base_dir=base_dir,
                repo_root=repo_root,
            )
            if record:
                referenced_inputs[record["path"]] = record
    scenes_by_id = {scene["id"]: scene for scene in manifest["scenes"]}
    for scene in timeline["scenes"]:
        canonical_scene = scenes_by_id[scene["id"]]
        for asset_name in _list(canonical_scene.get("assets", []), f"scene {scene['id']}.assets"):
            if asset_name not in resolved_assets:
                raise ResolutionError(f"scene {scene['id']} references unknown asset {asset_name}")
        for token in _list(canonical_scene.get("evidence", []), f"scene {scene['id']}.evidence"):
            record = _validate_reference(
                token,
                manifest=manifest,
                resolved_evidence=resolved_evidence,
                base_dir=base_dir,
                repo_root=repo_root,
            )
            if record:
                referenced_inputs[record["path"]] = record

    manifest_record = _hash_record(manifest_path, repo_root, role="canonical_manifest")
    tool_path = Path(__file__).resolve()
    inputs = {
        record["path"]: record
        for record in (
            [manifest_record]
            + ([project_input] if project_input else [])
            + list(evidence_inputs.values())
            + list(asset_inputs.values())
            + list(audio_inputs.values())
            + list(referenced_inputs.values())
        )
    }
    film = _mapping(manifest.get("film"), "film")
    maximum_duration = _number(
        film.get("maximum_duration_seconds"), "film.maximum_duration_seconds", minimum=0.001
    )

    return {
        "schema_version": 1,
        "artifact_type": "metralign.resolved_film",
        "generated_at_utc": _utc_timestamp(),
        "canonical_manifest": manifest_record,
        "derivation": {
            "git": _git_binding(repo_root),
            "resolver": _hash_record(tool_path, repo_root, role="resolver_source"),
            "source_date_epoch": os.environ.get("SOURCE_DATE_EPOCH"),
        },
        "project": resolved_project,
        "film": {
            "title": film.get("title"),
            "maximum_duration_seconds": maximum_duration,
            "target_duration_seconds": film.get("target_duration_seconds"),
            "frame": film.get("frame"),
            "metadata": film.get("metadata"),
        },
        "theme": manifest.get("theme"),
        "audio": manifest.get("audio"),
        "voice": manifest.get("voice"),
        "render": manifest.get("render"),
        "outputs": manifest.get("outputs"),
        "qa": manifest.get("qa"),
        "resolved_assets": resolved_assets,
        "resolved_evidence": resolved_evidence,
        "timeline": timeline,
        "missing_audio": missing_audio,
        "input_files": sorted(inputs.values(), key=lambda record: record["path"]),
    }


def write_resolved_film(
    manifest_path: Path,
    output_path: Path,
    *,
    probe: Probe = probe_media,
    allow_missing: bool = False,
) -> dict[str, Any]:
    result = resolve_film(manifest_path, probe=probe, allow_missing=allow_missing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("video/film.yaml"), help="canonical film YAML"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video/build/resolved_film.json"),
        help="derived JSON output",
    )
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="emit an explicitly incomplete development artifact (final QA rejects it)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = write_resolved_film(
            args.manifest,
            args.output,
            probe=lambda path: probe_media(path, ffprobe=args.ffprobe),
            allow_missing=args.allow_missing,
        )
    except ResolutionError as exc:
        print(f"resolve_film: {exc}", file=sys.stderr)
        return 2
    print(
        f"resolved {len(result['timeline']['scenes'])} scenes / "
        f"{len(result['timeline']['segments'])} narration segments -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
