#!/usr/bin/env python3
"""Generate the standalone Metralign generator-transfer suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from drift_sense.independent_renderer import (
    IndependentRendererConfig,
    generate_independent_pair,
    verify_independent_suite,
    write_independent_suite,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("dram", "finfet", "both"), default="both")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2_026_081_701)
    parser.add_argument("--image-size", type=int, default=1000)
    parser.add_argument("--nominal-scale", type=float, default=0.1)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="medium")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_pairs < 1:
        raise SystemExit("num-pairs must be at least 1")
    config = IndependentRendererConfig(
        image_size=args.image_size,
        nominal_scale=args.nominal_scale,
        supersample=args.supersample,
        difficulty=args.difficulty,
    )
    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    architectures = ("dram", "finfet") if args.architecture == "both" else (args.architecture,)
    records = []
    for index in range(args.num_pairs):
        records.append(
            generate_independent_pair(
                args.output_dir,
                index,
                architectures[index % len(architectures)],
                args.seed,
                config,
            )
        )
        print(f"generated {index + 1}/{args.num_pairs}", file=sys.stderr)
    metadata = write_independent_suite(args.output_dir, records, config)
    verify_independent_suite(args.output_dir)
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.jsonl"),
        "metadata": str(args.output_dir / "suite-metadata.json"),
        "dataset_sha256": metadata["dataset_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
