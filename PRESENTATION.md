# Metralign submission deck

The editable deck is `Metralign_Submission_Deck.pptx`; the rendered copy is
`Metralign_Submission_Deck.pdf`. Every slide contains a `[Sources]` block in its
speaker notes. The deck uses checked-in algorithm outputs and licensed
microscopy—not generated scientific illustration.

## Evidence policy

- The headline result is the sealed seven-suite synthetic report under
  `results/frozen/`.
- Real SEM/TEM rows are separate transfer checks with their source-defined
  ground-truth limits and declared fallback counts.
- The standalone renderer is a predeclared development transfer set, not part
  of the frozen result.
- External methods use fixed, documented adapters. XFeat is a retrospective
  task-mismatch control, not a claim about its intended benchmarks.
- Timing is evaluator wall time around localization unless a slide explicitly
  says otherwise.

## Slide map

### 1 — Identity

Metralign is a deterministic, training-free absolute-site localizer for
repeated wafer-like structures. The right-hand image is the report-bound real
microscopy plate from `real_imagery/results/`.

### 2 — Task

One fine-field reference and one wider-field search image are mapped to the
reference centre in search-image pixels. The frozen IID example is
`000002_dram`, seed `1011209461`: ground truth `(632.919, 234.237)`, prediction
`(632.832, 234.201)`, error `0.094 px`.

Source: `results/frozen/cases/success_iid_000002_dram/` and
`results/frozen/reports/iid.json`.

### 3 — Principle

The lattice first calibrates pitch, relative scale, and bounded rotation. The
same one-period difference operator is then applied to both acquisitions so
that site-specific variation can rank candidate locations. A supported local
maximum is refined to one subpixel coordinate.

Sources: `src/drift_sense/spectral.py`, `representations.py`, `localizer.py`,
and `ambiguity.py`.

### 4 — Default pipeline

The release path is phase calibration → matched period differences →
candidate-local support → ambiguity evidence → subpixel refinement. Low
transform confidence triggers a bounded residual search. Small unsupported
periodic templates complete through the existing baseline0 estimator while
reporting review status and zero absolute-site confidence.

Sources: `docs/method.md`, `docs/ambiguity-safety.md`, and
`src/drift_sense/safety.py`.

### 5 — Design choices

The slide explains the three choices that directly target absolute-site
failure: phase-drift calibration, matched periodic differences, and
conservative ambiguity handling. Development measurements used to select the
release configuration remain under `results/frozen/development/`; they are not
pooled with the frozen report.

### 6 — Measured evidence

Frozen synthetic report:

| Measure | Value |
|---|---:|
| Pairs within 1 px | 1,398 / 1,400 · 99.86% |
| Median error | 0.083 px |
| P95 error | 0.319 px |
| Mean localization wall time | 239 ms |

Separate transfer evidence:

| Population | Result | Declared fallback |
|---|---:|---:|
| Separate renderer, predeclared final 100 | 97 / 100 within 1 px | 0 / 100 |
| Carinthia wafer SEM, balanced digital crops | 23 / 24 within 1 px | 13 / 24 |
| Boiko FE-SEM digital crops | 29 / 30 within 1 px | 0 / 30 |
| Registered MiniTEM digital crops | 70 / 70 within 5 px | 70 / 70 |

The MiniTEM queries are too small for the one-period residual stage, so their
coordinates come from the declared fallback. Real-imagery selection and
ground-truth limits are documented in `real_imagery/README.md`.

External controls on the frozen 1,400 pairs:

- Best classical fixed adapter: OpenCV scale/rotation grid, 35.57% within 1 px.
- Official XFeat* + USAC_MAGSAC: 77.07% coverage and 0/1,400 within 5 px under
  the locked retrospective protocol.

Sources: `results/frozen/benchmark_report.json`,
`real_imagery/results/real-sem-report.json`,
`real_imagery/results/registered-tem-report.json`, and
`results/comparisons/`.

### 7 — Safety and integrity

The public interface exposes ambiguity, fallback, review reasons, and
alternative hypotheses alongside the coordinate. The retrospective selective
audit reviews exactly 2/1,400 archived cases, captures both large errors, and
leaves all 1,398 accepted cases within 1 px. Clean evidence generation used
commit `615806cfd5c4243ecca49cbf169324b33180f160`; 111 tests and the history-aware
release scan passed before bound measurements.

### 8 — Repository

Install and infer:

```bash
python -m pip install -c constraints.txt .
metralign --reference reference.png --search search.png
```

Standard output contains exactly two finite floating-point values. The public
repository is <https://github.com/Achxy/metralign>; the frozen artifact index is
`results/frozen/ARTIFACTS.md`.

### 9 — Primary sources

The final slide cites the four acquired-microscopy records, the core
registration sources, official OpenCV and scikit-image APIs, XFeat, and periodic
structure references. The full 34-DOI mechanism-to-code matrix is
`references.md`.

## Claim boundaries retained in the deck

- Synthetic datasets are phenomenological stress tests, not calibrated
  microscope or fabrication simulators.
- The alternate frozen capture renderer changes acquisition and resampling but
  retains the primary latent architecture and coordinate geometry.
- The standalone renderer has a separate geometry/capture implementation and
  a predeclared seed; its 97/100 result is development transfer evidence.
- Carinthia and Boiko accuracy use exact same-acquisition digital-crop ground
  truth. Native Boiko cross-magnification values are agreement to a fixed proxy,
  not physical-coordinate accuracy.
- MiniTEM uses publisher-registered Low/GT pairs plus deterministic digital
  crops; all 70 full-interface calls declare fallback.
- XFeat’s result describes this fixed unequal-FOV periodic task adapter only.

## Reproduce the sealed protocol

The archived report is immutable. A new independent report-split reproduction
can be generated with:

```bash
python benchmark_suites.py \
  --output-dir data/generated/benchmark-frozen \
  --split report \
  --report-pairs 200 \
  --method full \
  --confirm-report
```

Use `results/frozen/ARTIFACTS.md` for the original manifests, hashes, reports,
environment record, and the two archived periodic-ambiguity failures.
