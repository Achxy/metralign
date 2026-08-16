#!/usr/bin/env python3
"""Locate the reference FOV center in a search image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from drift_sense.localizer import LocalizationConfig, localize


def _load_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"image does not exist: {path}")
    with Image.open(path) as image:
        if image.mode not in {"L", "I", "I;16", "F"}:
            image = image.convert("L")
        array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid grayscale image: {path}")
    if float(np.ptp(array)) <= 0:
        raise ValueError(f"image has no intensity variation: {path}")
    return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["baseline0", "multiscale", "structure_gradient", "structure_residual", "full"],
        default="full",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--scale-range", type=float, default=0.006)
    parser.add_argument("--rotation-range", type=float, default=3.0)
    parser.add_argument(
        "--phase-calibration", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evidence-channel", choices=["structural", "gradient", "raw"], default="structural"
    )
    parser.add_argument(
        "--spatial-residual", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--lattice-grouping", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ambiguity-rule", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--subpixel-refinement", choices=["parabolic", "dft", "none"], default="parabolic"
    )
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.scale_range < 0 or args.rotation_range < 0:
        parser.error("search ranges must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        reference = _load_grayscale(args.reference)
        search = _load_grayscale(args.search)
        result = localize(
            reference,
            search,
            LocalizationConfig(
                method=args.method,
                top_k=args.top_k,
                scale_range=args.scale_range,
                rotation_range=args.rotation_range,
                enable_phase_calibration=args.phase_calibration,
                periodic_evidence_channel=args.evidence_channel,
                enable_spatial_residual=args.spatial_residual,
                enable_lattice_grouping=args.lattice_grouping,
                enable_ambiguity_rule=args.ambiguity_rule,
                subpixel_refinement=args.subpixel_refinement,
            ),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.diagnostics:
        print(json.dumps(result.to_dict(), sort_keys=True), file=sys.stderr)
    print(f"{result.x:.6f} {result.y:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
