"""Automatic, explicitly heuristic failure taxonomy."""

from __future__ import annotations


def classify_failure(record: dict, prediction: dict, error: float) -> str | None:
    if error <= 5.0:
        return None
    if abs(prediction["selected_scale"] - record["actual_scale"]) > 0.003:
        return "scale estimation"
    expected_rotation = record.get("template_rotation_deg", -record["rotation_deg"])
    if abs(prediction["selected_rotation_deg"] - expected_rotation) > 0.6:
        return "rotation estimation"
    if prediction["ambiguity_flag"]:
        return "periodic ambiguity"
    if record["suite"] == "scan_distortion":
        return "scan distortion"
    residual = prediction.get("channel_scores", {}).get("residual")
    if residual is not None and residual < 0.08:
        return "residual SNR insufficient"
    x, y = record["center_x"], record["center_y"]
    size = record["search_geometry"]["width"]
    if min(x, y, size - 1 - x, size - 1 - y) < 60:
        return "edge-of-search truncation"
    if prediction["confidence"] < 0.35:
        return "false high-correlation candidate"
    if error < 12.0:
        return "subpixel refinement error"
    return "wrong lattice family"
