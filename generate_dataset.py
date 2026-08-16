#!/usr/bin/env python3
"""Generate deterministic Metralign image pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from drift_sense.dataset import AUGMENTATIONS, GeneratorConfig, generate_pair, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["dram", "finfet", "both"], required=True)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument(
        "--suite",
        choices=["iid", "high_noise", "geometry_ood", "transform_ood", "periodic_ambiguity", "scan_distortion", "cross_generator"],
        default="iid",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--image-size", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--resampler", choices=["area", "lanczos"], default="area")
    parser.add_argument(
        "--disable-augmentation",
        action="append",
        choices=AUGMENTATIONS,
        default=[],
        help="Disable one stress component; repeat for a component ablation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = {
        "image_size": args.image_size,
        "supersample": args.supersample,
        "suite": args.suite,
        "difficulty": args.difficulty,
        "resampler": args.resampler,
        "disabled_augmentations": tuple(args.disable_augmentation),
    }
    if args.config:
        values.update(json.loads(args.config.read_text()))
    try:
        cfg = GeneratorConfig(**values)
    except TypeError as exc:
        raise SystemExit(f"invalid generator configuration: {exc}") from exc
    cfg.disabled_augmentations = tuple(cfg.disabled_augmentations)
    unknown = sorted(set(cfg.disabled_augmentations) - set(AUGMENTATIONS))
    if (
        args.num_pairs < 1
        or cfg.image_size < 64
        or cfg.supersample < 1
        or cfg.suite not in {"iid", "high_noise", "geometry_ood", "transform_ood", "periodic_ambiguity", "scan_distortion", "cross_generator"}
        or cfg.difficulty not in {"easy", "medium", "hard"}
        or cfg.resampler not in {"area", "lanczos"}
        or not (0 < cfg.nominal_scale < 1)
        or unknown
    ):
        raise SystemExit(
            "invalid generator configuration: require num-pairs >=1, image-size >=64, "
            "supersample >=1, 0<nominal-scale<1, known suite/difficulty/resampler, "
            f"and known disabled augmentations (unknown={unknown})"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architectures = ["dram", "finfet"] if args.architecture == "both" else [args.architecture]
    records = []
    for index in range(args.num_pairs):
        architecture = architectures[index % len(architectures)]
        records.append(generate_pair(args.output_dir, index, architecture, args.seed, cfg))
        print(f"generated {index + 1}/{args.num_pairs}", file=sys.stderr)
    manifest = write_manifest(args.output_dir, records)
    print(str(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
