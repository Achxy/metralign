# Frozen artifact index

These artifacts document the single protected reporting-split evaluation of the unchanged `full` method at algorithm commit `c9363bfce535a812eb541417f3297602e97f619a`.

## Integrity

- Aggregate report: `benchmark_report.json`
- Aggregate SHA-256: `a169bffa170707da166206640150702c87202670a1a22745ff6128ad46ff5b69`
- Localizer implementation SHA-256: `7819d767b5ab3aeadd40bb99addefcf28948bca9c07bb7a84b5fb20345f39881`
- Report state: `working_tree_dirty=false` in all seven schema-v2 reports
- Protocol: 200 pairs per suite, 7 suites, both architectures, report split, medium difficulty, 1000×1000 search images, supersample 2
- Result: 1,398/1,400 within 1 px; two genuine FinFET periodic-ambiguity failures

The absolute manifest and image paths embedded in the reports record the original sealed evaluation location on the external volume. Exact manifest copies are in `manifests/`; the complete 2,800-image archive remains at `/Volumes/External/drift-sense-release-2026-08-16/frozen-report/datasets/report/`. Representative image triples are checked in under `cases/` so the release remains inspectable without copying the full image corpus into Git.

## Contents

- `reports/`: seven complete per-suite schema-v2 evaluation reports, including per-sample predictions, diagnostics, image hashes, provenance, and grouped metrics
- `manifests/`: exact JSONL manifests for all seven reporting suites
- `frozen-ledger.jsonl`: measured experiment record and the explicit no-post-report-tuning decision
- `development/`: fixed-manifest pipeline, representation, subpixel, K, and generator-component studies; these are development evidence only
- `cases/success_iid_000002_dram/`: mechanically selected suite-median IID success, 0.094 px error
- `cases/failure_high_noise_000081_finfet/`: 247.665 px ambiguity failure
- `cases/failure_scan_distortion_000185_finfet/`: 313.800 px ambiguity failure
- `failure_high_noise.png` and `failure_scan_distortion.png`: failure montages generated from verified report rows

## Frozen environment

The benchmark ran on macOS 26.5.2 arm64 with CPython 3.14.6, NumPy 2.5.2, SciPy 1.18.0, OpenCV 4.14.0, and Pillow 12.3.0. No GPU, learned weights, or network service was used. The primary runtime metric is evaluator wall time immediately around `localize()` and excludes image decoding and report construction.

The results were independently rechecked against every manifest, sidecar, image SHA-256, report row, Euclidean error, subgroup, and aggregate before release. Later documentation, presentation, and website commits do not change the recorded implementation fingerprint.
