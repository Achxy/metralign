# Metralign presentation evidence map

The editable final deck is `DriftSense_LatticeLock_Frozen.pptx`; its rendered counterpart is `DriftSense_LatticeLock_Frozen.pdf`. Slide 5 uses only the explicitly labeled development study in `results/frozen/development/pipeline-study.json`. Slide 6 uses only the sealed reporting-split artifacts in `results/frozen/`. Every slide contains a `[Sources]` speaker-note block.

## Slide 1 — Metralign

- Navigation-error recovery for repeated wafer structures.
- Input: one high-resolution reference image and one search image covering approximately 10× the physical width.
- Output: one center coordinate in search-image pixels.

## Slide 2 — Localization under periodic ambiguity

- Show the reference footprint and several high-correlation lattice-equivalent sites in one real evaluation pair.
- Define Euclidean coordinate error and the official center-nearest tie rule.
- State the practical failure mode: normalized template correlation can rank the wrong repeated site first.

Frozen example: `iid/000002_dram`, seed `1011209461`, selected mechanically as the nearest suite-median IID success. Error: 0.094 px. The exact reference, search, and metadata are under `results/frozen/cases/success_iid_000002_dram/`.

## Slide 3 — Calibration and disambiguation

- Periodicity as calibration signal: phase drift across scan lines estimates lattice vectors, scale, rotation, and real-space pitch.
- Periodicity suppression for disambiguation: apply the same one-real-lattice-period spatial-difference transfer function to both captures.
- Match separable line-constant sequences only when their measured energy supports that model; otherwise preserve joint two-dimensional residual evidence.
- When site-specific information is insufficient, score the structural map and apply the required center-nearest rule over every threshold-qualified non-maximum-suppressed peak, including peaks beyond top-K.
- Raw tied maxima always drive the required center-nearest selection. When a reliable real-space basis is available, group the same peaks by nearest integer-lattice offset for family validation and offset diagnostics; trust that evidence only when its representatives cover at least 65% of the raw tied peaks.
- Invoke center-nearest selection only when a score tie is corroborated by local score-neighborhood variation, low transform-estimate confidence, or low residual evidence. Record the raw/grouped counts, coverage, and supporting evidence in diagnostics. In the joint residual path, rank by the two-channel mean and accept its global maximum as residual evidence only when both channels support that coordinate.

Evidence status: the shared-pipeline stage study is complete on one fixed 100-pair IID development manifest. It is development-only configuration evidence, not frozen-suite performance.

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

Populate this slide only from `development_pipeline_study.json` produced by `benchmark_pipeline.py` against one fixed development manifest. Do not construct it from comparisons between legacy method aliases. The first row is the ZNCC starting point; subsequent rows use the `full` implementation while progressively enabling the selected pipeline controls.

| Shared-pipeline stage | ≤1 px | ≤3 px | P95 error (px) | Mean inference wall (ms) | Decision |
|---|---:|---:|---:|---:|---|
| ZNCC starting point | 38% | 38% | 741.941 | 51.4 | insufficient under repetition |
| + phase calibration | 46% | 46% | 639.186 | 146.7 | transform alone is insufficient |
| + raw spatial residual | 95% | 95% | 13.435 | 167.5 | resolves most lattice aliases |
| + structural spatial residual | 100% | 100% | 0.643 | 220.0 | fixes remaining gross aliases |
| + reliable-basis lattice-family diagnostics | 100% | 100% | 0.643 | 222.9 | retain interpretable family evidence |
| + multi-evidence ambiguity rule | 100% | 100% | 0.643 | 220.0 | retain safe center-nearest fallback |
| + parabolic subpixel refinement | 100% | 100% | 0.259 | 217.3 | selected release refinement |

The representation, refinement, and top-K blocks in the same artifact are alternatives rather than cumulative stages. Label every Slide 5 number “development only”; this study supports configuration selection and does not replace the protected held-out results on Slide 6.

## Slide 6 — Frozen results and failures

Source all cells from the immutable aggregate created by a reporting-split benchmark with explicit `--confirm-report`.

| Frozen split | Count | ≤1 px | ≤3 px | Median error (px) | P95 error (px) | P95 inference wall (ms) |
|---|---:|---:|---:|---:|---:|---:|
| IID | 200 | 100.0% | 100.0% | 0.094 | 0.296 | 263.296 |
| High noise | 200 | 99.5% | 99.5% | 0.090 | 0.318 | 312.943 |
| Geometry OOD | 200 | 100.0% | 100.0% | 0.084 | 0.285 | 289.272 |
| Transform OOD | 200 | 100.0% | 100.0% | 0.090 | 0.313 | 276.439 |
| Periodic ambiguity | 200 | 100.0% | 100.0% | 0.025 | 0.086 | 381.369 |
| Scan distortion | 200 | 99.5% | 99.5% | 0.125 | 0.392 | 273.494 |
| Cross-generator (alternate capture renderer) | 200 | 100.0% | 100.0% | 0.110 | 0.366 | 259.503 |

- Pooled: 1,398/1,400 (99.8571%) within 1 px, median 0.0830 px, P95 0.3193 px, P99 0.4361 px, maximum 313.800 px, and mean runtime 239.405 ms. The mean error of 0.5152 px is outlier-inflated; the within-5-px mean is 0.1143 px.
- Success image: `iid/000002_dram`, 0.094 px.
- Failure image: `scan_distortion/000185_finfet`, 313.800 px. Weak site-specific FinFET residual, 4,428 periodic alternatives, 50.9% basis coverage, and strong row shift cause center-nearest fallback to select the wrong off-center lattice site.
- The other failure is `high_noise/000081_finfet`, 247.665 px, from the same identifiable low-residual periodic-ambiguity mode. Both are flagged ambiguous; no post-report tuning was performed.
- The primary renderer uses area versus Lanczos for reference/search by default. The alternate capture renderer uses Kaiser versus Hann polyphase paths and changes edge response, blur, illumination, noise ordering, and parameter distributions, while retaining the shared latent architecture and coordinate geometry. State this scope with its results.
- Confirm that `runtime_ms` is evaluator wall time around `localize()`; present internal `localizer_runtime_ms` separately if it is shown.

## Slide 7 — Exact execution environment

- Metralign version: `0.1.0`.
- Python: obtain with `python --version`.
- NumPy, SciPy, OpenCV, and Pillow: obtain with `python -m pip freeze`.
- Operating system and CPU: macOS 26.5.2, arm64.
- Accelerator: none used by the checked-in inference implementation.
- Learned weights: none.
- Method arguments, suite manifests and SHA-256 values, seed sets, commit identifier, report schema, and wall-clock policy: copy from the frozen experiment record.

Frozen record: Metralign 0.1.0; CPython 3.14.6; NumPy 2.5.2; SciPy 1.18.0; OpenCV 4.14.0; Pillow 12.3.0; CPU only. Algorithm commit `c9363bfce535a812eb541417f3297602e97f619a`; implementation SHA-256 `7819d767b5ab3aeadd40bb99addefcf28948bca9c07bb7a84b5fb20345f39881`; all reports clean and schema version 2.

## Slide 8 — Reproduction and demo

```bash
python -m pip install -c constraints.txt .
python generate_dataset.py --architecture dram --num-pairs 1 --output-dir data/generated/demo --seed 2026
python infer.py --reference data/generated/demo/000000_dram_reference.png --search data/generated/demo/000000_dram_search.png
```

- Show that stdout contains exactly two finite numbers.
- Link the release commit, README fresh-machine procedure, and machine-readable result artifact.
- Frozen algorithm commit: `c9363bfce535a812eb541417f3297602e97f619a`.
- Repository: <https://github.com/Achxy/metralign>
- Project site: <https://achxy.github.io/metralign/>
- Machine-readable evidence: `results/frozen/benchmark_report.json`; aggregate SHA-256 `a169bffa170707da166206640150702c87202670a1a22745ff6128ad46ff5b69`.

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

## Reproduce the sealed protocol

The official reporting split has already been evaluated once. The following command records the exact configuration for an independent reproduction in a new output directory; it is not an instruction to replace the archived release artifacts:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method full \
  --confirm-report
```

Run this extraction command against the archived aggregate. It refuses an unconfirmed or non-report plan and verifies the copied manifest hashes before printing the Slide 6 fields.

```bash
RESULT=results/frozen/benchmark_report.json python - <<'PY'
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
    manifest = Path("results/frozen/manifests") / f"{result['suite']}.jsonl"
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

The seven per-suite artifacts under `results/frozen/reports/` each declare schema version 2, their manifest and image bindings, clean commit state, and `runtime_ms` as evaluator wall time around `localize()`. `results/frozen/ARTIFACTS.md` indexes the release evidence and the two failures above 5 px.
