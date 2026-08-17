# Official XFeat external benchmark

This optional study adds a modern learned local-feature matcher to the classic
OpenCV and scikit-image comparison. It uses the official XFeat source and
checkpoint without vendoring or patching either one. The benchmark is
retrospective development evidence on frozen image bytes; it is not part of
Metralign's original frozen claim and is not described as a state-of-the-art
comparison.

## Locked method

The protocol was written before any XFeat prediction, match count, coverage, or
error was inspected. Its complete machine-readable form is
`evidence/external/xfeat-predeclared-protocol.json`.

- official repository: `verlab/accelerated_features`;
- commit: `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`;
- inference-code bundle SHA-256: `3ea50cd28a4f753efe7d296fabbdf067bf060c1bd79d7f2fb38a545abd4596ca`;
- checkpoint SHA-256: `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`;
- license: Apache-2.0 for the repository. The checkpoint is distributed in
  that repository; the pinned tree does not provide a separate checkpoint
  license.

The adapter downsamples the finer-sampling reference once using the known 0.1
sampling ratio and OpenCV area interpolation. It then calls the official
`match_xfeat_star` method with `top_k=8000`. The official registration notebook's
fixed homography stage is used unchanged: USAC_MAGSAC, 3.5 px threshold, 1,000
iterations, and 0.999 confidence. The downsampled reference's pixel center is
projected into the search image. A result with fewer than four matches, no
homography, fewer than four inliers, or a non-finite projection is unresolved
and counts as a failure.

These settings are an explicit task adapter, not a claim that XFeat was trained
for 10× unequal-field-of-view semiconductor localization. XFeat's paper studies
general local correspondence, pose estimation, and visual localization. The
known 0.1 resampling is necessary to put both acquisitions at approximately the
same physical sampling before matching.

Primary sources:

- [official XFeat repository](https://github.com/verlab/accelerated_features)
- [pinned Apache-2.0 license](https://github.com/verlab/accelerated_features/blob/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/LICENSE)
- [CVPR 2024 paper](https://doi.org/10.1109/CVPR52733.2024.00259)

## Population rule

The preferred population is all 1,400 records in the seven frozen reporting
manifests. A timing-only probe uses exactly one SHA-256-ranked record from each
suite/architecture cell. If its median runtime projects to more than 4,500
seconds for 1,400 records, the locked fallback is 20 SHA-256-ranked records per
suite/architecture cell (280 total). The gate reads runtime only; predictions,
statuses, matches, inliers, and errors remain hidden until the population
choice is recorded. The separate independent-renderer final-100 manifest is
then evaluated in full as a development transfer check.

The locked timing-only gate measured a 171.024 ms median across its 14 records,
projecting 239.434 seconds for 1,400 records. This is below the predeclared
4,500-second ceiling, so the full 1,400-record population was selected. The
compact decision is recorded in
`evidence/external/xfeat-population-decision.json`; no accuracy, coverage,
match, or inlier outcome was emitted or inspected before that choice.

## Reproduction

Create a separate Python 3.12 environment; the optional PyTorch stack is not a
core inference dependency.

```bash
python3.12 -m venv /tmp/metralign-xfeat-venv
/tmp/metralign-xfeat-venv/bin/python -m pip install -r requirements-xfeat.txt
```

Download or clone the official repository at the pinned commit. The adapter
fails closed if the three inference source files or checkpoint do not match the
locked hashes.

```bash
PYTHONPATH=src /tmp/metralign-xfeat-venv/bin/python compare_xfeat.py \
  --data-dir /path/to/frozen-report/datasets/report/iid \
  --official-source /path/to/accelerated_features \
  --selection timing-probe \
  --hardware-label "Apple M3 CPU" \
  --output /tmp/xfeat-timing-probe.json \
  --quiet
```

The timing probe records the complete protocol and source bindings, selected-ID
hash, every input-image hash, hardware label, model-load time, and per-record
runtime, but intentionally omits matches, predictions, and error outcomes. The
final measured reports add matches, inliers, predictions, errors, and resolved
distributions. Timing excludes image decoding and one-time model loading.

## Measured result

Both populations were measured from clean commit `615806cfd5c4243ecca49cbf169324b33180f160`, with `dirty=false`, adapter SHA-256 binding, official-source binding, checkpoint binding, and complete input hashes. No setting was changed after outcomes were read.

| Population | Coverage | ≤1 px | ≤5 px | Median resolved error | P95 resolved error | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| Frozen synthetic, all 1,400 | 1,079 / 1,400 · 77.07% | 0 / 1,400 | 0 / 1,400 | 629.221 px | 1,037.635 px | 295.692 ms |
| Independent renderer, all 100 | 78 / 100 · 78.00% | 0 / 100 | 0 / 100 | 690.465 px | 1,098.103 px | 149.318 ms |

The outputs are `results/comparisons/xfeat-frozen-all1400-development.json` and `results/comparisons/xfeat-independent-final100-development.json`. XFeat* finds enough local correspondences to estimate many homographies, but strong repetition makes those correspondences non-unique and the resulting homographies project incorrect absolute locations. This is a fixed task-adapter mismatch, not a claim about XFeat on its intended general correspondence, pose-estimation, and visual-localization benchmarks. No outcome-informed plausibility gate or tuning was added.

The isolated report's distribution inventory contains a stale `drift-sense 0.1.0` entry from an ignored `src/drift_sense.egg-info` directory exposed by `PYTHONPATH`. That distribution was not installed in the XFeat environment and is not used by inference. The imported package reports 0.2.0; the actual adapter/source is independently bound by commit and SHA-256.
