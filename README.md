# Drift-Sense

Drift-Sense estimates the center of a high-resolution reference field of view inside a lower-resolution wide-search image. The default method measures lattice scale and rotation from phase drift, cancels the periodic backbone with matched one-lattice-period spatial differences, chooses a separable or joint residual match according to measured line-constant energy, applies the required center-nearest rule when site-specific evidence is insufficient, and refines the selected peak to a floating-point coordinate.

## Install — one command

From the repository root, with CPython 3.10–3.14:

```bash
python -m pip install -c constraints.txt .
```

This constrained command selects the checked-in reference dependency versions. `python -m pip install .` remains supported when integration into an existing environment is more important than reproducing the reference environment.

## Infer — one command

```bash
python infer.py --reference path/to/reference.png --search path/to/search.png
```

Successful stdout is exactly one line containing `x y` in search-image pixels, for example:

```text
512.381204 477.992015
```

The origin is the top-left search pixel; `x` increases rightward and `y` increases downward. Both values may be fractional. `--diagnostics` writes a JSON object to stderr without changing stdout. Invalid input returns a nonzero status and writes the error to stderr.

## Method

The default `full` method follows this sequence:

1. Clip each image to its 1st–99th percentile range and normalize it using its median and median absolute deviation, with a standard-deviation fallback.
2. Estimate the two lattice fundamentals with zero-padded one-dimensional spectra. Phase drift across scan lines estimates their vectors, which provide relative scale, rotation, search-image pitch, confidence, and a test for separable axes. The estimated scale remains bounded by `--scale-range` around 0.1, and rotation remains bounded by `--rotation-range`, which defaults to 3.0°. Low-confidence or inconsistent phase estimates trigger a bounded coarse-to-fine residual transform search.
3. Warp the reference once to that transform. For both transformed reference and search, apply the same symmetric spatial transfer function: subtract the average of samples displaced by plus and minus one measured real-lattice period. Separate `period_x` and `period_y` channels cancel the repeated backbone without relying on image-specific Fourier-notch masks.
4. Measure how much residual energy is explained by a sequence constant along each line. Use separable one-dimensional sequence correlation only when the lattice axes are separable and both residual channels are sufficiently line-constant. Otherwise correlate both residual channels jointly in two dimensions and rank coordinates by their equal-weight mean. A candidate must also clear the residual-evidence floor in each channel, preventing a one-channel spike from winning the fused map.
5. If the transformed reference has too little site-specific residual, or joint residual correlation has insufficient evidence, correlate the structural lattice backbone. The ambiguity decision scans every non-maximum-suppressed local maximum inside the score threshold, including peaks outside the truncated top-K list. Raw tied maxima always remain authoritative for the required center-nearest selection. When the search lattice estimate yields a reliable real-space basis, the same peaks are grouped by nearest integer-lattice offset only to validate and report lattice-family evidence; grouping is trusted when its representatives cover at least 65% of the raw tied peaks. A close score alone is insufficient to invoke center-nearest selection: it must also be supported by local peak perturbation, an unstable transform estimate, or low residual evidence. The residual path uses the same multi-evidence rule with its tighter score threshold.
6. Fit a parabola around the selected peak. The separable path refines each axis and corrects for measured lattice slant; the joint and structural paths refine the two-dimensional peak directly.

No model weights, network service, GPU, or notebook is required. Inference is deterministic for fixed input images and arguments.

Available methods are `baseline0`, `multiscale`, `structure_gradient`, `structure_residual`, and `full`. `baseline0` performs robust normalization, 0.1× area downsampling, normalized template correlation, and peak refinement. `multiscale`, `structure_gradient`, and `structure_residual` retain the former transform-grid, gradient, and Fourier-notch formulations as legacy ablations. The default `full` path does not use their weighted structural/gradient/Fourier-notch grid.

Useful controls:

```bash
python infer.py \
  --reference path/to/reference.png \
  --search path/to/search.png \
  --method full \
  --top-k 32 \
  --scale-range 0.006 \
  --rotation-range 3.0 \
  --diagnostics
```

`--scale-range` is an absolute scale radius around the nominal 0.1 relationship. `--rotation-range` is in degrees.

## Generate deterministic synthetic pairs

The generator creates the reference and search captures independently from one continuous latent wafer field; it does not paste a finished reference into the search image. Ground-truth centers are transformed from shared world coordinates into search-image coordinates and then adjusted for the generated scan-line displacement. Process variation is part of the shared latent geometry. There are no synthetic brightness barcodes or other location labels embedded in the images.

Generate 100 alternating DRAM and FinFET pairs:

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 100 \
  --output-dir data/generated \
  --seed 2026 \
  --difficulty medium \
  --suite iid
```

Supported suites are `iid`, `high_noise`, `geometry_ood`, `transform_ood`, `periodic_ambiguity`, `scan_distortion`, and `cross_generator`. When supersampling is enabled, the primary renderer uses the requested resampler for the reference (`area` by default) and the other path for the search (`lanczos` by default), preventing a shared resampling signature. In `cross_generator`, the alternate renderer uses Kaiser-windowed polyphase sampling for the reference and Hann-windowed polyphase sampling for the search. A JSON sidecar is written for every pair, and `manifest.jsonl` is the input to evaluation.

The generator is a deterministic stress-test implementation, not a calibrated model of a particular microscope, process node, material stack, beam energy, or dose. Its geometry and acquisition ranges must not be described as physically typical without evidence. The `cross_generator` suite also changes edge response, probe blur, illumination, noise ordering, and acquisition distributions. It still shares the latent architecture and coordinate-geometry implementation with the primary renderer, so it tests capture-renderer shift rather than independence from every synthetic modeling assumption. See `references.md` for the evidence and status of individual mechanisms.

### Controlled generator-component studies

`--disable-augmentation` is repeatable. It neutralizes only the named generator mechanisms and records the disabled names in each sample's metadata, so the command is reproducible from its seed and arguments:

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 100 \
  --output-dir data/generated/dev-disabled-noise-jitter \
  --seed 71000019 \
  --suite iid \
  --disable-augmentation gaussian_noise \
  --disable-augmentation scan_jitter
```

For a paired, leave-one-component-out study, use the same seed and generator configuration through the wrapper:

```bash
python ablate_augmentations.py \
  --output-dir results/dev_augmentation_ablation \
  --num-pairs 100 \
  --seed 71000019 \
  --architecture both \
  --suite iid \
  --supersample 2 \
  --method full
```

The wrapper evaluates an all-enabled dataset and one same-seed dataset per disabled mechanism, verifies and hashes their manifests, and writes `augmentation_ablation.json`. This is a development robustness-sensitivity study for a training-free localizer: its deltas do not measure training benefit, prove physical realism, or replace the protected reporting split. Use a new output directory for each run.

## Evaluate

Evaluate the default method and save the same JSON report that is printed to stdout:

```bash
python evaluate.py \
  --data-dir data/generated \
  --method full \
  --output results/generated_full.json
```

Run all checked-in methods for an ablation-style comparison:

```bash
python evaluate.py \
  --data-dir data/generated \
  --method all \
  --output results/generated_all.json
```

Evaluation reports use schema version 2. They contain the manifest SHA-256, exact method configuration, environment versions, Euclidean coordinate error, success rates at 0.5, 1, 2, 3, and 5 pixels, mean/median/P90/P95/P99/maximum error, architecture/suite/difficulty breakdowns, per-sample diagnostics, and failure counts.

The primary `runtime_ms` field is evaluator wall time measured immediately around `localize()` with `time.perf_counter_ns`; it excludes image decoding and report construction. `localizer_runtime_ms` separately records the localizer's internal timer. Image I/O and full per-sample wall time are retained as `image_io_ms` and `sample_wall_ms`.

Confirm a report's schema, manifest binding, and runtime definition before using it:

```bash
REPORT=results/generated_full.json python - <<'PY'
import json
import os
from hashlib import sha256
from pathlib import Path

report = json.loads(Path(os.environ["REPORT"]).read_text())
assert report["schema_version"] == 2
manifest = Path(report["manifest"])
assert sha256(manifest.read_bytes()).hexdigest() == report["manifest_sha256"]
assert report["metric_definition"]["primary_runtime_field"] == "runtime_ms"
assert "wall clock around localize()" in report["metric_definition"]["runtime_scope"]
print("report confirmed", report["evaluated_record_count"])
PY
```

## Deterministic benchmark suites

`benchmark_suites.py` assigns non-overlapping seeds, generates every requested suite, verifies each manifest, runs evaluation, verifies report-to-manifest hashes and configuration, and writes an aggregate report. Inspect a development plan before running it:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-dev \
  --split dev \
  --dev-pairs 100 \
  --method all \
  --dry-run
```

Run the development suites by removing `--dry-run`. Access to the protected reporting split requires an explicit confirmation flag:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method full \
  --confirm-report
```

`--confirm-report` records the confirmation in the plan and prevents accidental reads of the frozen reporting split. Do not use it until the method and settings are frozen. The command above documents the sealed protocol; the official reporting split has already been run once and is archived under `results/frozen/`. The resulting per-suite reports are schema version 2; the aggregate benchmark wrapper is schema version 1 and preserves each suite's manifest hash, metrics, failure counts, environment, and metric definition.

### Shared-pipeline development study

Generate one fixed development manifest, then measure cumulative production stages and controlled alternatives against that same data:

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 100 \
  --output-dir data/generated/pipeline-dev \
  --seed 72000019 \
  --suite iid

python benchmark_pipeline.py \
  --data-dir data/generated/pipeline-dev \
  --output-dir results/dev_pipeline_study \
  --top-k-values 8 16 32 64 128
```

`benchmark_pipeline.py` writes `development_pipeline_study.json`. Its stage block starts at ZNCC and progressively enables phase calibration, raw then structural spatial residuals, reliable-basis lattice-family diagnostics, the multi-evidence ambiguity rule, and parabolic refinement. Its representation, refinement, and top-K blocks are controlled alternatives, not cumulative stages. Every value is development-only evidence for configuration selection; it must not be presented as frozen held-out performance. The input data directory must contain `manifest.jsonl`, and the output directory must not already exist.

## Experimental status

The frozen reporting split contains 1,400 previously protected pairs: 200 per suite, alternating DRAM and FinFET, at 1000×1000 search resolution and supersample 2. The method and configuration were fixed at clean algorithm commit `c9363bfce535a812eb541417f3297602e97f619a`; all reports declare implementation SHA-256 `7819d767b5ab3aeadd40bb99addefcf28948bca9c07bb7a84b5fb20345f39881` and `working_tree_dirty=false`.

| Frozen suite | Count | ≤0.5 px | ≤1 px | Median (px) | P95 (px) | Max (px) | P95 runtime (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID | 200 | 100.0% | 100.0% | 0.094 | 0.296 | 0.447 | 263.296 |
| High noise | 200 | 98.5% | 99.5% | 0.090 | 0.318 | 247.665 | 312.943 |
| Geometry OOD | 200 | 100.0% | 100.0% | 0.084 | 0.285 | 0.469 | 289.272 |
| Transform OOD | 200 | 100.0% | 100.0% | 0.090 | 0.313 | 0.451 | 276.439 |
| Periodic ambiguity | 200 | 100.0% | 100.0% | 0.025 | 0.086 | 0.165 | 381.369 |
| Scan distortion | 200 | 99.5% | 99.5% | 0.125 | 0.392 | 313.800 | 273.494 |
| Cross-generator | 200 | 99.0% | 100.0% | 0.110 | 0.366 | 0.543 | 259.503 |

Pooled performance is 1,394/1,400 (99.5714%) within 0.5 px and 1,398/1,400 (99.8571%) within 1, 3, and 5 px. Median error is 0.0830 px, P95 is 0.3193 px, P99 is 0.4361 px, and maximum error is 313.800 px. Mean error is 0.5152 px because the two genuine failures are large; mean error over the 1,398 cases within 5 px is 0.1143 px. Evaluator wall time around `localize()` averages 239.405 ms with P95 359.649 ms, excluding image I/O.

Both failures are FinFET cases flagged ambiguous. In `high_noise/000081_finfet`, weak site-specific residual evidence leaves 4,606 periodic alternatives and center-nearest selection is 247.665 px from an off-center ground truth. In `scan_distortion/000185_finfet`, weak residual evidence, 4,428 alternatives, only 50.9% reliable-basis coverage, and strong row shift produce a 313.800 px error. The estimated scale and rotation remain close in both cases; the identifiable limitation is absolute-site disambiguation when FinFET evidence is nearly one-dimensional. No tuning followed inspection of the reporting split.

The machine-readable aggregate, seven schema-v2 reports, exact manifests, ledger, development studies, representative cases, and failure montages are in `results/frozen/`; see `results/frozen/ARTIFACTS.md`. The aggregate file SHA-256 is `a169bffa170707da166206640150702c87202670a1a22745ff6128ad46ff5b69`. An independent audit recomputed all 1,400 records, 2,800 image bindings, ground-truth errors, group metrics, and aggregate values.

The shared-pipeline and generator-component artifacts under `results/frozen/development/` remain development-only. On the fixed 100-pair IID development manifest, the K sweep at 8, 16, 32, 64, and 128 produced identical coordinate metrics; mean runtimes ranged from 207.1 to 226.3 ms. K=32 is retained as a stable middle setting, not as a held-out optimization claim. The ambiguity scan and its optional lattice diagnostics are not truncated to K, but K still controls the retained list used for ordinary ranking and runner-up diagnostics.

The final presentation is available as `DriftSense_LatticeLock_Frozen.pptx` and `DriftSense_LatticeLock_Frozen.pdf`. `PRESENTATION.md` records the evidence mapping behind every slide.

## Fresh-machine reproduction

The commands below are the complete POSIX setup and smoke path for the frozen default method.

```bash
# Run from the root of a fresh release checkout.
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python generate_dataset.py --architecture dram --num-pairs 1 --output-dir data/smoke --seed 2026 --supersample 1
python infer.py --reference data/smoke/000000_dram_reference.png --search data/smoke/000000_dram_search.png --method full
python evaluate.py --data-dir data/smoke --method full --output results/smoke.json
python -m pytest
python tools/scan_release.py . --history
```

Dataset generation at the default 1000×1000 size is CPU- and storage-intensive; use a directory on a volume with adequate free space for larger suites. The CI workflow repeats installation, tests, generation, inference-output validation, evaluation, and repository scanning on clean Linux runners.

### Dependency policy

- `pyproject.toml` is the canonical package metadata and declares compatible dependency ranges plus exact isolated-build tools.
- `requirements.txt` installs only runtime dependencies under `constraints.txt`.
- `requirements-dev.txt` installs the local package and test extra under the same constraints.
- `constraints.txt` is a universal resolution for CPython 3.10–3.14. Environment markers select compatible NumPy and SciPy versions for Python 3.10, 3.11, and 3.12–3.14; all other runtime and test dependencies are exact pins.
- CI installs from `requirements-dev.txt`, runs `pip check`, and tests Python 3.10, 3.12, and 3.14. Regenerate constraints only as an intentional dependency update and rerun the full matrix.

The constraints were resolved with `uv 0.11.25` against packages published by 2026-08-16:

```bash
uv pip compile pyproject.toml \
  --all-extras \
  --universal \
  --python-version 3.10 \
  --exclude-newer 2026-08-16 \
  --no-annotate \
  --no-strip-markers \
  --output-file constraints.txt
```

Exact version constraints make resolver output repeatable; they do not authenticate distribution files. CI separately scans Git history with TruffleHog OSS and runs the project-specific residue scan. GitHub Actions are referenced by full commit SHA, with release tags retained only as comments.

## Repository map

```text
infer.py                  coordinate-only inference CLI
generate_dataset.py       deterministic dataset CLI
evaluate.py               metrics and diagnostics CLI
benchmark_suites.py       deterministic split and suite orchestrator
benchmark_pipeline.py     shared-pipeline development study
ablate_augmentations.py   paired generator-component sensitivity study
src/drift_sense/          localization and rendering modules
configs/default.json      default generator settings
constraints.txt           exact cross-version dependency resolution
requirements-dev.txt      constrained local package and test install
tests/                    unit and CLI tests
examples/                 runnable command recipes
results/                  machine-readable evaluation reports
results/frozen/           sealed reports, manifests, cases, and audit index
tools/scan_release.py     release-residue scanner
PRESENTATION.md           evidence-linked slide outline
DriftSense_LatticeLock_Frozen.pptx  editable final presentation
DriftSense_LatticeLock_Frozen.pdf   rendered final presentation
references.md             source and mechanism audit
```

## License

The source is available under the MIT License; see `LICENSE`.
