# Drift-Sense presentation outline

This deck is an evidence checklist, not a source of measurements. Slide 5 may use only the explicitly labeled development stage study produced by `benchmark_pipeline.py`; Slide 6 may use only frozen benchmark artifacts produced by `benchmark_suites.py`. Do not transcribe numbers from terminal output. Confirm that each underlying evaluation report is schema version 2 and bound to its manifest hash, and never relabel development evidence as frozen performance.

## Slide 1 — Drift-Sense

- Navigation-error recovery for repeated wafer structures.
- Input: one high-resolution reference image and one search image covering approximately 10× the physical width.
- Output: one center coordinate in search-image pixels.

## Slide 2 — Localization under periodic ambiguity

- Show the reference footprint and several high-correlation lattice-equivalent sites in one real evaluation pair.
- Define Euclidean coordinate error and the official center-nearest tie rule.
- State the practical failure mode: normalized template correlation can rank the wrong repeated site first.

Evidence status: representative frozen sample not yet selected.

## Slide 3 — Calibration and disambiguation

- Periodicity as calibration signal: phase drift across scan lines estimates lattice vectors, scale, rotation, and real-space pitch.
- Periodicity suppression for disambiguation: apply the same one-real-lattice-period spatial-difference transfer function to both captures.
- Match separable line-constant sequences only when their measured energy supports that model; otherwise preserve joint two-dimensional residual evidence.
- When site-specific information is insufficient, score the structural map and apply the required center-nearest rule over every threshold-qualified non-maximum-suppressed peak, including peaks beyond top-K.
- Raw tied maxima always drive the required center-nearest selection. When a reliable real-space basis is available, group the same peaks by nearest integer-lattice offset for family validation and offset diagnostics; trust that evidence only when its representatives cover at least 65% of the raw tied peaks.
- Invoke center-nearest selection only when a score tie is corroborated by local peak perturbation, transform-estimate instability, or low residual evidence. Record the raw/grouped counts, coverage, and supporting evidence in diagnostics. In the joint residual path, rank by the two-channel mean but require candidate-local support from both channels.

Evidence status: component ablation not yet measured on frozen suites.

## Slide 4 — Processing path

```text
continuous latent wafer field
        ↓
independent reference/search acquisition and resampling
        ↓
phase-drift lattice scale/rotation/pitch estimate
        ↓ (low confidence: bounded residual transform search)
        ↓
matched one-lattice-period spatial differences
        ↓
line-constant energy test
   ↙ separable 1-D sequences   ↘ joint 2-D residual correlation
        ↓
raw all-peak threshold scan → optional reliable-basis lattice-family diagnostics
        ↓
score tie + independent ambiguity evidence → center-nearest selection
        ↓
parabolic peak refinement → x y
```

- Annotate which transforms affect ground truth and which affect only image formation.
- State that the synthetic model is not calibrated to a named microscope or fabrication process.

## Slide 5 — Shared-pipeline development evidence

Populate this slide only from `development_pipeline_study.json` produced by `benchmark_pipeline.py` against one fixed development manifest. Do not construct it from comparisons between legacy method aliases. The first row is the ZNCC starting point; subsequent rows use the `full` implementation while progressively enabling production controls.

| Shared-pipeline stage | ≤1 px | ≤3 px | P95 error (px) | Mean inference wall (ms) | Decision |
|---|---:|---:|---:|---:|---|
| ZNCC starting point | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + phase calibration | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + raw spatial residual | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + structural spatial residual | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + reliable-basis lattice-family grouping | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + multi-evidence ambiguity rule | not yet measured | not yet measured | not yet measured | not yet measured | pending |
| + parabolic subpixel refinement | not yet measured | not yet measured | not yet measured | not yet measured | pending |

The representation, refinement, and top-K blocks in the same artifact are alternatives rather than cumulative stages. Label every Slide 5 number “development only”; this study supports configuration selection and does not replace the protected held-out results on Slide 6.

## Slide 6 — Frozen results and failures

Source all cells from the immutable aggregate created by a reporting-split benchmark with explicit `--confirm-report`.

| Frozen split | Count | ≤1 px | ≤3 px | Median error (px) | P95 error (px) | P95 inference wall (ms) |
|---|---:|---:|---:|---:|---:|---:|
| IID | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| High noise | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Geometry OOD | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Transform OOD | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Periodic ambiguity | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Scan distortion | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |
| Cross-generator (alternate capture renderer) | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured | not yet measured |

- Success image: not yet selected from the frozen report.
- Genuine failure image: not yet selected from the frozen report.
- State the failure category recorded for the selected failure.
- The primary renderer uses area versus Lanczos for reference/search by default. The alternate capture renderer uses Kaiser versus Hann polyphase paths and changes edge response, blur, illumination, noise ordering, and parameter distributions, while retaining the shared latent architecture and coordinate geometry. State this scope with its results.
- Confirm that `runtime_ms` is evaluator wall time around `localize()`; present internal `localizer_runtime_ms` separately if it is shown.

## Slide 7 — Exact execution environment

- Drift-Sense version: `0.1.0`.
- Python: obtain with `python --version`.
- NumPy, SciPy, OpenCV, and Pillow: obtain with `python -m pip freeze`.
- Operating system and CPU: record from the machine used for the frozen run.
- Accelerator: none used by the checked-in inference implementation.
- Learned weights: none.
- Method arguments, suite manifests and SHA-256 values, seed sets, commit identifier, report schema, and wall-clock policy: copy from the frozen experiment record.

Frozen hardware and software record: not yet captured.

## Slide 8 — Reproduction and demo

```bash
python -m pip install -c constraints.txt .
python generate_dataset.py --architecture dram --num-pairs 1 --output-dir data/generated/demo --seed 2026
python infer.py --reference data/generated/demo/000000_dram_reference.png --search data/generated/demo/000000_dram_search.png
```

- Show that stdout contains exactly two finite numbers.
- Link the release commit, README fresh-machine procedure, and machine-readable result artifact.
- Frozen release commit and repository URL: not yet recorded.

## Slide 9 — Verified sources and limits

- Pull publication metadata and mechanism mappings from `references.md` only after its citation audit passes.
- Separate literature support for a mechanism from the configured stress-test range.
- State generator limitations, including the latent architecture and coordinate geometry shared by the primary and alternate capture renderers.
- State that reference/search resampling paths differ and that the generator adds no synthetic brightness barcode.
- For generator sensitivity claims, show the same-seed leave-one-component-out configuration from `ablate_augmentations.py`; state that it is a synthetic stress-mechanism study, not a training ablation or physical calibration.
- Do not infer physical parameter ranges from synthetic defaults.

Citation audit status: complete. The augmentation-to-implementation evidence matrix and verified DOI bibliography are in `references.md`.

## Produce development studies

Use one generated development manifest for the shared-pipeline stage study:

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

For repeatable generator-mechanism sensitivity, `--disable-augmentation` may be repeated on `generate_dataset.py`, or the paired wrapper may be used:

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

Both outputs are development-only. Keep their manifests, configurations, and report bindings, but do not copy their measurements into the frozen-results table.

## Produce and populate the frozen report

Running the reporting split requires an explicit confirmation. Do this only after method selection and settings are frozen:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method all \
  --confirm-report
```

Set `RESULT` to the aggregate report path and run this extraction command. It refuses an unconfirmed or non-report plan and prints the exact fields needed by Slide 6.

```bash
RESULT=data/generated/benchmark-frozen/reports/benchmark_report.json python - <<'PY'
import json
import os
from hashlib import sha256
from pathlib import Path

report = json.loads(Path(os.environ["RESULT"]).read_text())
assert report["schema_version"] == 1
assert report["plan"]["protocol"] == "frozen-report"
assert report["plan"]["report_access_confirmed"] is True
for result in report["results"]:
    assert result["split"] == "report"
    manifest = Path(result["manifest"])
    assert sha256(manifest.read_bytes()).hexdigest() == result["manifest_sha256"]
    for method, method_result in result["methods"].items():
        metrics = method_result["metrics"]
        print(
            result["suite"],
            method,
            metrics["count"],
            metrics["success_le_1px"],
            metrics["success_le_3px"],
            metrics["median_error_px"],
            metrics["p95_error_px"],
            metrics["p95_runtime_ms"],
        )
PY
```

The per-suite evaluation artifacts under `reports/report/` must each declare schema version 2, their manifest SHA-256, and `runtime_ms` as evaluator wall time around `localize()`. Archive the aggregate, per-suite reports, and exact manifests with the release. If a value is absent, leave the corresponding cell as “not yet measured.”
