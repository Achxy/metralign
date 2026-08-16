# Command examples

Run these commands from the repository root after `python -m pip install -c constraints.txt .`. The examples generate their own inputs and do not depend on development data being present.

## Generate one pair, then infer it

```bash
python generate_dataset.py \
  --architecture finfet \
  --num-pairs 1 \
  --output-dir data/generated/example \
  --seed 4701

python infer.py \
  --reference data/generated/example/000000_finfet_reference.png \
  --search data/generated/example/000000_finfet_search.png
```

Inference stdout contains only the predicted `x y` center in search-image pixels. Add `--diagnostics` to receive method, scores, the phase-drift transform estimate, ambiguity state, confidence, residual-channel evidence, lattice-group coverage, and internal runtime as JSON on stderr. When a reliable real-space basis exists, threshold-qualified peaks are grouped by integer-lattice offset only if grouped representatives cover at least 65% of the raw tied peaks. Center-nearest selection additionally requires support from local peak perturbation, transform instability, or low residual evidence. The default rotation bound is 3.0°.

## Evaluate the generated pair

```bash
python evaluate.py \
  --data-dir data/generated/example \
  --method full \
  --output data/generated/example/report.json
```

Confirm the schema, manifest binding, and primary runtime definition before reading metrics:

```bash
REPORT=data/generated/example/report.json python - <<'PY'
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

In schema version 2, `runtime_ms` is evaluator wall time immediately around `localize()` and excludes image I/O and report construction. `localizer_runtime_ms` is the localizer's separate internal timer. `image_io_ms` and `sample_wall_ms` retain the other measured scopes.

## Disable selected synthetic mechanisms repeatably

Repeat `--disable-augmentation` to neutralize more than one mechanism. The seed, arguments, and disabled names recorded in sample metadata define the generated dataset:

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 12 \
  --output-dir data/generated/example-disabled \
  --seed 4702 \
  --suite iid \
  --disable-augmentation gaussian_noise \
  --disable-augmentation scan_jitter
```

This changes a synthetic stress test; it is not a statement that the disabled mechanisms or configured ranges match a physical instrument.

## Run paired generator-component sensitivity

```bash
python ablate_augmentations.py \
  --output-dir results/dev_example_augmentation_ablation \
  --num-pairs 12 \
  --seed 4703 \
  --architecture both \
  --suite iid \
  --supersample 2 \
  --method full
```

The wrapper generates the all-enabled case and same-seed leave-one-component-out cases, then writes `augmentation_ablation.json` with manifest hashes and report paths. It measures development robustness sensitivity for this training-free localizer, not training gain or frozen performance. The output directory must not already exist.

## Run the shared-pipeline development study

Generate a fixed input manifest first:

```bash
python generate_dataset.py \
  --architecture both \
  --num-pairs 12 \
  --output-dir data/generated/example-pipeline-dev \
  --seed 4704 \
  --suite iid

python benchmark_pipeline.py \
  --data-dir data/generated/example-pipeline-dev \
  --output-dir results/dev_example_pipeline_study \
  --top-k-values 8 16 32 64 128
```

`development_pipeline_study.json` contains cumulative stage rows for ZNCC, phase calibration, spatial residual representations, reliable-basis lattice grouping, the multi-evidence ambiguity rule, and parabolic refinement. Representation, refinement, and K sections are controlled alternatives. All are development-only; use `benchmark_suites.py` with the protected reporting split for frozen performance. The study output directory must not already exist.

## Run deterministic development suites

Preview a plan without generating data:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-dev \
  --split dev \
  --dev-pairs 100 \
  --method all \
  --dry-run
```

Remove `--dry-run` to run all seven suites. Repeat `--suite` to select a subset, for example:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-dev \
  --split dev \
  --suite iid \
  --suite cross_generator \
  --dev-pairs 100 \
  --method all
```

The orchestrator assigns disjoint split/suite seeds, verifies manifest contents, verifies each schema-version-2 report against its manifest hash and requested configuration, and writes an aggregate report under `reports/`.

## Run the protected reporting split

Freeze the method and arguments before this command. Reporting-split access is rejected unless `--confirm-report` is supplied explicitly:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method all \
  --confirm-report
```

The benchmark plan records `report_access_confirmed: true`. Do not tune against these results or rerun the reporting split while changing method settings. Use `--resume` only to reuse manifests and reports that the orchestrator verifies.

## Interpret renderer coverage

With the default supersampling arguments, the primary generator uses different downsampling paths for the two captures: the reference uses area integration and the search uses Lanczos. The alternate `cross_generator` capture path uses Kaiser-windowed polyphase sampling for the reference and Hann-windowed polyphase sampling for the search, together with different edge, blur, illumination, noise-ordering, and acquisition-distribution choices. Both renderers share latent architecture and coordinate-geometry code. Neither path inserts synthetic brightness barcodes.

Do not tune method settings on reporting seeds, and state the shared latent/geometry boundary when reporting cross-generator results.
