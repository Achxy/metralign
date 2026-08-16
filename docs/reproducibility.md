# Reproducibility protocol

This document separates three activities that should not be conflated: a one-pair smoke test, development studies, and the frozen reporting protocol.

## Reference environment

Supported CPython versions are 3.10–3.14. Install the constrained contributor environment with:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

`constraints.txt` constrains package selection for the supported Python versions. Reproducing a recorded result also requires the recorded source commit, inputs, arguments, and platform context.

## Fresh-machine smoke test

The smoke test verifies installation, generation, inference output, evaluation, tests, and release scanning. It does not recompute the 1,400-pair frozen result.

```bash
python generate_dataset.py \
  --architecture dram \
  --num-pairs 1 \
  --output-dir data/smoke \
  --seed 2026 \
  --supersample 1

python infer.py \
  --reference data/smoke/000000_dram_reference.png \
  --search data/smoke/000000_dram_search.png \
  --method full

python evaluate.py \
  --data-dir data/smoke \
  --method full \
  --output results/smoke.json

python -m pytest
python tools/scan_release.py . --history
```

Successful inference stdout contains two values formatted to six decimal places.

## Generate and evaluate a development set

The commands below create a new illustrative development study. They do not regenerate the checked-in development table.

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 100 \
  --output-dir data/generated/dev-iid \
  --seed 2026 \
  --difficulty medium \
  --suite iid

python evaluate.py \
  --data-dir data/generated/dev-iid \
  --method full \
  --output results/generated_full.json
```

Generation is reproducible for a fixed commit, dependency and numerical-backend environment, seed, and argument set.

## Shared-pipeline development study

```bash
python benchmark_pipeline.py \
  --data-dir data/generated/dev-iid \
  --output-dir results/dev_pipeline_study \
  --top-k-values 8 16 32 64 128
```

The output separates cumulative stages of the selected pipeline from representation, subpixel, and K alternatives. These measurements support configuration decisions only; they are not frozen held-out results.

The checked-in study in [`results/frozen/development/pipeline-study.json`](../results/frozen/development/pipeline-study.json) used the archived 100-pair IID development manifest with seed base `11000003`, manifest SHA-256 `195ff33fe6f433fe717d27899a11f3d2572070fe40667c0d4432465329ad80ba`, and dataset SHA-256 `f770e6baace4fd08ed7f8e62a796c49efe0111afccce002ae2920b223c9bd716`.

## Generator-component sensitivity

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

This is a paired leave-one-mechanism-out sensitivity study for a training-free localizer. It does not measure training benefit or establish physical calibration.

## Frozen reporting protocol

The archived release used:

- algorithm commit `c9363bfce535a812eb541417f3297602e97f619a`
- implementation SHA-256 `7819d767b5ab3aeadd40bb99addefcf28948bca9c07bb7a84b5fb20345f39881`
- both architectures
- seven suites
- 200 pairs per suite
- medium difficulty
- 1000×1000 search images
- supersample 2
- `full`, K=32, scale radius 0.006, rotation radius 3.0°, structural evidence, lattice diagnostics enabled, ambiguity rule enabled, and parabolic refinement

The orchestration command is:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method full \
  --confirm-report
```

`--confirm-report` guards report-split execution through `benchmark_suites.py`. It does not restrict arbitrary filesystem access. Do not replace the archived evidence with a later rerun.

## Evidence binding

Each schema-v2 report records:

- manifest SHA-256,
- per-image SHA-256 values,
- combined dataset SHA-256,
- implementation SHA-256,
- Git commit and dirty flag,
- exact evaluation configuration,
- environment versions,
- per-sample ground truth, prediction, error, runtime, and diagnostics,
- subgroup metrics and failure counts.

The aggregate report links all seven suite reports. Checked-in copies live under [`results/frozen/`](../results/frozen/); the exact inventory and original external-volume location are documented in [`ARTIFACTS.md`](../results/frozen/ARTIFACTS.md).

## Frozen machine record

- macOS 26.5.2 arm64
- CPython 3.14.6
- NumPy 2.5.2
- SciPy 1.18.0
- OpenCV 4.14.0
- Pillow 12.3.0
- CPU only

The primary `runtime_ms` field is evaluator wall time around `localize()`. It excludes image I/O and report construction.
