#!/usr/bin/env python3
"""Compare Metralign with reproducible adapters for established OpenCV methods."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

try:
    import skimage
except ImportError:  # pragma: no cover - core-only installation
    skimage = None

from drift_sense.dataset import load_manifest
from drift_sense.external_baselines import BaselineConfig, METHODS, run_baseline
from drift_sense.localizer import LocalizationConfig, localize
from evaluate import build_artifact_binding, validate_records


METHOD_REFERENCES = {
    "opencv_template": "https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html",
    "opencv_grid_template": "https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html",
    "opencv_template_phase": "https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html",
    "opencv_ecc_affine": "https://docs.opencv.org/4.x/dc/d6b/group__video__track.html",
    "opencv_sift_ransac": "https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html",
    "skimage_template_phase": "https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation",
    "metralign": "local implementation under src/drift_sense",
}
METHOD_METADATA = {
    "opencv_template": {
        "category": "area-based template localization",
        "primitive": "cv2.matchTemplate / TM_CCOEFF_NORMED",
        "settings": {
            "template_sampling": "nominal scale with cv2.INTER_AREA",
            "transform_search": "none",
            "subpixel_refinement": "none",
        },
    },
    "opencv_grid_template": {
        "category": "area-based scale/rotation template localization",
        "primitive": "cv2.matchTemplate / TM_CCOEFF_NORMED",
        "settings": {
            "template_sampling": "cv2.INTER_AREA then cv2.INTER_LANCZOS4 rotation",
            "scale_grid": "linspace(nominal-scale-range, nominal+scale-range, scale-steps)",
            "rotation_grid": "linspace(-rotation-range, +rotation-range, rotation-steps)",
            "subpixel_refinement": "none",
        },
    },
    "opencv_template_phase": {
        "category": "area-based coarse match with Fourier subpixel refinement",
        "primitive": "cv2.matchTemplate then cv2.phaseCorrelate",
        "settings": {
            "coarse_match": "TM_CCOEFF_NORMED at nominal scale",
            "window": "cv2.createHanningWindow / CV_32F",
            "phase_normalization": "OpenCV default",
        },
    },
    "opencv_ecc_affine": {
        "category": "intensity-based affine registration",
        "primitive": "cv2.findTransformECC / MOTION_AFFINE",
        "settings": {
            "initialization": "OpenCV nominal-scale TM_CCOEFF_NORMED coordinate",
            "termination": "COUNT|EPS, 100 iterations, epsilon 1e-6",
            "gaussian_filter_size": 5,
            "plausible_scale_interval": [0.05, 0.20],
        },
    },
    "opencv_sift_ransac": {
        "category": "local-feature similarity registration",
        "primitive": "cv2.SIFT + BFMatcher + cv2.estimateAffinePartial2D / RANSAC",
        "settings": {
            "sift": "nfeatures=5000, contrastThreshold=0.01",
            "matcher": "L2 brute force, k=2, Lowe ratio 0.75",
            "ransac": "3px threshold, 5000 iterations, confidence 0.999, refineIters=10",
            "minimum_inliers": 4,
            "plausible_scale_interval": [0.05, 0.20],
        },
    },
    "skimage_template_phase": {
        "category": "independent normalized-correlation and Fourier subpixel registration",
        "primitive": "skimage.feature.match_template + skimage.registration.phase_cross_correlation",
        "settings": {
            "template_sampling": "skimage.transform.resize, anti_aliasing=True, order=1",
            "coarse_match": "skimage.feature.match_template at nominal scale",
            "phase_refinement": "upsample_factor=20, normalization=None",
            "claim_boundary": (
                "direct whole-image phase correlation is not applicable because the "
                "inputs have unequal physical fields of view and a nominal 10x sampling "
                "difference; phase correlation is therefore used only after an "
                "independent coarse template crop"
            ),
        },
    },
    "metralign": {
        "category": "periodic-structure-specific localization",
        "primitive": "local frozen implementation",
        "settings": "read from the archived schema-v2 report",
    },
}
THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 5.0)
ROOT = Path(__file__).resolve().parent


def _load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def _portable_locator(path: Path, root: Path = ROOT) -> str:
    """Return a stable locator without recording a machine-specific absolute path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"{resolved.parent.name}/{resolved.name}"


def _archived_metralign_reference(
    report_dir: Path | None,
    dataset_name: str,
    manifest_sha256: str,
    selected_ids: list[str],
) -> dict[str, object] | None:
    if report_dir is None:
        return None
    path = (report_dir / f"{dataset_name}.json").resolve()
    if not path.is_file():
        raise ValueError(f"missing Metralign reference report: {path}")
    report_bytes = path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Metralign report manifest mismatch: {path}")
    method = report.get("methods", {}).get("full")
    if not isinstance(method, dict):
        raise ValueError(f"Metralign report is missing full method: {path}")
    samples = method.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Metralign report is missing full-method samples: {path}")
    by_id: dict[str, dict] = {}
    for row in samples:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or sample_id in by_id:
            raise ValueError(f"Metralign report has invalid or duplicate sample IDs: {path}")
        by_id[sample_id] = row
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(
            f"Metralign report omits {len(missing)} selected sample IDs: {path}"
        )
    compact_samples = []
    for sample_id in selected_ids:
        row = by_id[sample_id]
        compact_samples.append(
            {
                "id": sample_id,
                "architecture": row.get("architecture"),
                "suite": row.get("suite"),
                "ground_truth": row.get("ground_truth"),
                "prediction": row.get("prediction"),
                "error": float(row["error"]),
                "runtime_ms": float(row["runtime_ms"]),
            }
        )
    return {
        "source_report": _portable_locator(path),
        "report_sha256": sha256(report_bytes).hexdigest(),
        "algorithm_git_commit": report.get("artifact_binding", {}).get("git_commit"),
        "evaluated_record_count": len(compact_samples),
        "metrics": summarize(compact_samples),
        "samples": compact_samples,
        "note": (
            "Archived coordinates were not recomputed by this comparison. This compact "
            "per-sample extraction is self-contained for aggregation and is bound to the "
            "untouched source report by SHA-256."
        ),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    resolved = [row for row in rows if row["error"] is not None]
    errors = np.asarray([row["error"] for row in resolved], dtype=np.float64)
    runtimes = np.asarray([row["runtime_ms"] for row in rows], dtype=np.float64)
    result: dict[str, object] = {
        "count": total,
        "resolved_count": len(resolved),
        "unresolved_count": total - len(resolved),
        "coverage": len(resolved) / total,
        "mean_runtime_ms": float(np.mean(runtimes)),
        "p95_runtime_ms": float(np.percentile(runtimes, 95, method="linear")),
    }
    for threshold in THRESHOLDS:
        successes = int(np.count_nonzero(errors <= threshold)) if errors.size else 0
        result[f"success_le_{threshold:g}px"] = successes / total
        result[f"resolved_success_le_{threshold:g}px"] = (
            successes / len(resolved) if resolved else 0.0
        )
    if errors.size:
        result.update(
            {
                "mean_error_px_resolved": float(np.mean(errors)),
                "median_error_px_resolved": float(np.median(errors)),
                "p95_error_px_resolved": float(
                    np.percentile(errors, 95, method="linear")
                ),
                "max_error_px_resolved": float(np.max(errors)),
            }
        )
    else:
        result.update(
            {
                "mean_error_px_resolved": None,
                "median_error_px_resolved": None,
                "p95_error_px_resolved": None,
                "max_error_px_resolved": None,
            }
        )
    return result


def _evaluate_method(
    method: str,
    root: Path,
    records: list[dict],
    cfg: BaselineConfig,
    quiet: bool,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        reference = _load_grayscale(root / record["reference"])
        search = _load_grayscale(root / record["search"])
        if method == "metralign":
            start = time.perf_counter_ns()
            prediction = localize(reference, search, LocalizationConfig(method="full"))
            result = {
                "x": float(prediction.x),
                "y": float(prediction.y),
                "status": "resolved",
                "score": float(prediction.selected_score),
                "runtime_ms": (time.perf_counter_ns() - start) / 1_000_000.0,
                "diagnostics": {
                    "ambiguity_flag": prediction.ambiguity_flag,
                    "decision_support": prediction.decision_support,
                },
            }
        else:
            result = run_baseline(method, reference, search, cfg).to_dict()
        x, y = result["x"], result["y"]
        error = (
            None
            if x is None or y is None
            else float(np.hypot(float(x) - record["center_x"], float(y) - record["center_y"]))
        )
        rows.append(
            {
                "id": record["id"],
                "architecture": record["architecture"],
                "suite": record["suite"],
                "ground_truth": [float(record["center_x"]), float(record["center_y"])],
                "prediction": None if x is None else [float(x), float(y)],
                "error": error,
                "status": result["status"],
                "score": result["score"],
                "runtime_ms": float(result["runtime_ms"]),
                "diagnostics": result["diagnostics"],
            }
        )
        if not quiet:
            print(f"{method}: {index + 1}/{len(records)}", file=sys.stderr)
    return {"metrics": summarize(rows), "samples": rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--method", choices=[*METHODS, "metralign", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--metralign-report-dir",
        type=Path,
        help="optional directory of archived per-suite Metralign JSON reports",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--nominal-scale", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, default=0.006)
    parser.add_argument("--scale-steps", type=int, default=5)
    parser.add_argument("--rotation-range", type=float, default=3.0)
    parser.add_argument("--rotation-steps", type=int, default=5)
    parser.add_argument("--ecc-downsample", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = BaselineConfig(
        nominal_scale=args.nominal_scale,
        scale_range=args.scale_range,
        scale_steps=args.scale_steps,
        rotation_range=args.rotation_range,
        rotation_steps=args.rotation_steps,
        ecc_downsample=args.ecc_downsample,
    )
    methods = list(METHODS) if args.method == "all" else [args.method]
    cv2.setNumThreads(1)
    datasets: dict[str, object] = {}
    try:
        for data_dir in args.data_dir:
            manifest = (data_dir / "manifest.jsonl" if data_dir.is_dir() else data_dir).resolve()
            records = load_manifest(manifest)
            validate_records(manifest.parent, records)
            selected = records[: args.limit] if args.limit is not None else records
            dataset_name = manifest.parent.name
            if dataset_name in datasets:
                raise ValueError(f"duplicate dataset name: {dataset_name}")
            binding = build_artifact_binding(
                manifest, records, Path(__file__).resolve().parent
            )
            datasets[dataset_name] = {
                "manifest": _portable_locator(manifest),
                "artifact_binding": binding,
                "archived_metralign": _archived_metralign_reference(
                    args.metralign_report_dir,
                    dataset_name,
                    binding["manifest_sha256"],
                    [str(record["id"]) for record in selected],
                ),
                "dataset_record_count": len(records),
                "evaluated_record_count": len(selected),
                "methods": {
                    method: _evaluate_method(
                        method, manifest.parent, selected, cfg, args.quiet
                    )
                    for method in methods
                },
            }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = {
        "schema_version": 2,
        "study": "external-registration-baseline-comparison",
        "scope": (
            "General-purpose OpenCV methods adapted to the known nominal sampling "
            "ratio and evaluated on identical manifests. Unresolved estimates count "
            "against all-sample success rates."
        ),
        "task_definition": {
            "input": (
                "one 1000x1000 finer-sampling grayscale reference and one 1000x1000 "
                "wider-field grayscale search image of the same latent region"
            ),
            "output": "reference field-of-view center (x, y) in search-image pixels",
            "known_prior": "nominal reference/search sampling ratio 0.1",
            "ground_truth": "manifest center_x and center_y",
            "error": "Euclidean center error in search-image pixels",
            "fairness": (
                "Every method receives identical image bytes and the same nominal ratio. "
                "No Metralign prediction initializes an external method, and settings do "
                "not vary by suite."
            ),
        },
        "configuration": asdict(cfg),
        "method_references": {method: METHOD_REFERENCES[method] for method in methods},
        "method_metadata": {method: METHOD_METADATA[method] for method in methods},
        "external_software": {
            "opencv": {
                "version": cv2.__version__,
                "license": "Apache License 2.0",
                "license_source": "https://github.com/opencv/opencv/blob/4.x/LICENSE",
            },
            "scikit_image": {
                "version": skimage.__version__ if skimage is not None else None,
                "license": "BSD 3-Clause",
                "license_source": "https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt",
            },
        },
        "metric_definition": {
            "coordinate_error": "Euclidean distance in search-image pixels",
            "success_denominator": "all evaluated records, including unresolved estimates",
            "resolved_metrics": "reported separately and explicitly suffixed _resolved",
            "runtime_scope": "method adapter wall clock; excludes image decoding",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "scikit_image": skimage.__version__ if skimage is not None else None,
            "pillow": pillow_version,
        },
        "datasets": datasets,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
