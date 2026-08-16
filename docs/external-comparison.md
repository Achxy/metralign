# External registration comparison protocol

This study compares the archived Metralign result with general-purpose registration methods on the same frozen image bytes and coordinates. It is a method comparison, not a claim that every baseline was designed for periodic semiconductor localization.

## Task

Each record contains a 1000×1000 finer-sampling reference image and a 1000×1000 wider-field search image. The known nominal sampling ratio is `0.1`. A method must return the reference field-of-view center in search-image pixels. Error is Euclidean distance from `center_x, center_y` in the manifest.

Every method receives identical images and the same nominal ratio. External methods do not receive Metralign coordinates. Settings are fixed across suites. An unresolved estimate counts against the all-sample success rate; resolved-only accuracy is reported separately.

## Archived development result

The same fixed adapters were also run on all seven archived development manifests (100 records per suite, 700 total). This is a completeness and reproducibility check, not a second test set.

| Method | Coverage | ≤0.5 px | ≤1 px | Median error, resolved | P95 error, resolved | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| Metralign, archived | 100.00% | 99.43% | 100.00% | 0.081 px | 0.305 px | 238.1 ms |
| OpenCV grid template | 100.00% | 31.71% | 33.14% | 268.997 px | 724.095 px | 1,254.1 ms |
| scikit-image template + phase | 100.00% | 30.29% | 30.29% | 273.385 px | 722.781 px | 163.3 ms |
| OpenCV template + phase | 100.00% | 30.14% | 30.14% | 279.905 px | 725.211 px | 17.9 ms |
| OpenCV template | 100.00% | 28.71% | 30.14% | 280.077 px | 723.332 px | 18.1 ms |
| OpenCV ECC affine | 82.14% | 0.14% | 28.43% | 243.369 px | 713.780 px | 931.9 ms |
| OpenCV SIFT + RANSAC | 0.29% | 0.00% | 0.00% | 650.292 px | 918.334 px | 1,869.5 ms |

## Frozen report result

The primary comparison uses the seven untouched reporting manifests: 200 records per suite, 1,400 total. Metralign numbers are read from the archived reports; they are not recomputed or replaced. External rows were evaluated later against the manifest-bound image bytes.

| Method | Coverage | ≤0.5 px | ≤1 px | Median error, resolved | P95 error, resolved | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|
| Metralign, archived | 100.00% | 99.57% | 99.86% | 0.083 px | 0.319 px | 239.4 ms |
| OpenCV grid template | 100.00% | 33.86% | 35.57% | 250.644 px | 719.140 px | 3,328.5 ms |
| OpenCV template + phase | 100.00% | 32.50% | 32.57% | 266.028 px | 719.598 px | 47.6 ms |
| OpenCV template | 100.00% | 30.93% | 32.57% | 265.172 px | 718.396 px | 65.7 ms |
| OpenCV ECC affine | 80.71% | 0.07% | 30.36% | 220.515 px | 702.013 px | 1,177.0 ms |
| scikit-image template + phase | 100.00% | 32.14% | 32.14% | 267.710 px | 719.192 px | 393.7 ms |
| OpenCV SIFT + RANSAC | 0.29% | 0.00% | 0.00% | 613.630 px | 804.181 px | 1,981.0 ms |

Success uses all 1,400 records as the denominator; unresolved estimates therefore count as failures. Error summaries use resolved estimates only, as labeled. Runtime is adapter wall time without image decoding. The Metralign timing comes from its archived single-method evaluator, whereas the external suite jobs ran later in parallel; accuracy is comparable, but these timings are not an isolated speed benchmark.

The per-suite ≤1 px rates expose the task boundary rather than hiding it in one pooled number:

| Method | IID | High noise | Scan distortion | Periodic ambiguity | Transform OOD | Geometry OOD | Cross-generator |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metralign, archived | 100.0% | 99.5% | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| OpenCV grid template | 48.5% | 38.5% | 38.0% | 0.5% | 38.5% | 39.0% | 46.0% |
| OpenCV template + phase | 48.0% | 38.0% | 36.5% | 0.5% | 20.0% | 39.0% | 46.0% |
| OpenCV template | 48.0% | 38.0% | 36.5% | 0.5% | 20.0% | 39.0% | 46.0% |
| OpenCV ECC affine | 44.5% | 37.5% | 32.0% | 0.5% | 19.5% | 37.5% | 41.0% |
| scikit-image template + phase | 48.0% | 37.5% | 35.5% | 0.0% | 19.5% | 39.0% | 45.5% |
| OpenCV SIFT + RANSAC | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Independent-renderer development probe

A separately implemented 40-pair renderer (seed `3105198707`, dataset SHA-256 `aff1b9639de164ee8648b5d1d1527ae8171584127b6fdbbb1f29794ebd3dd6da`) provides a development-only check outside the primary generator. It was not used as a frozen claim or to tune the remaining failure.

| Method | Coverage | ≤0.5 px | ≤1 px | Median error, resolved | P95 error, resolved |
|---|---:|---:|---:|---:|---:|
| Metralign | 100.0% | 97.5% | 97.5% | 0.045 px | 0.202 px |
| OpenCV template + phase | 100.0% | 70.0% | 70.0% | 0.164 px | 612.779 px |
| scikit-image template + phase | 100.0% | 70.0% | 70.0% | 0.058 px | 613.248 px |
| OpenCV grid template | 100.0% | 67.5% | 70.0% | 0.383 px | 613.171 px |
| OpenCV template | 100.0% | 67.5% | 70.0% | 0.388 px | 613.171 px |
| OpenCV ECC affine | 100.0% | 0.0% | 52.5% | 0.735 px | 616.818 px |
| OpenCV SIFT + RANSAC | 2.5% | 2.5% | 2.5% | 0.394 px | 0.394 px |

The subpixel external methods are precise when their coarse peak lands on the correct lattice member, but their high P95 errors show that they do not resolve absolute-site aliasing. SIFT's one resolved record must be read together with its 2.5% coverage.

## Implementations

| Adapter | Official primitive | Fixed adaptation |
|---|---|---|
| `opencv_template` | `cv2.matchTemplate`, `TM_CCOEFF_NORMED` | nominal 0.1× area-resized template; integer peak |
| `opencv_grid_template` | `cv2.matchTemplate`, `TM_CCOEFF_NORMED` | 5 scales × 5 rotations over the stated bounds; integer peak |
| `opencv_template_phase` | `cv2.matchTemplate`, `cv2.phaseCorrelate` | nominal coarse crop; Hanning-window phase refinement |
| `opencv_ecc_affine` | `cv2.findTransformECC`, affine motion | OpenCV template initialization; 0.5× working images; 100 iterations; epsilon 1e-6; Gaussian size 5 |
| `opencv_sift_ransac` | SIFT, L2 BF matching, `estimateAffinePartial2D` | 5,000 features; 0.75 ratio; 3 px RANSAC; 5,000 iterations; at least 4 inliers |
| `skimage_template_phase` | `skimage.feature.match_template`, `phase_cross_correlation` | independent nominal coarse crop; 20× DFT upsampling; unnormalized correlation |

Direct whole-image phase correlation is not a valid comparison because the inputs cover different physical fields of view and differ by roughly 10× in sampling. The scikit-image adapter therefore uses its independent normalized correlation to select a same-sized crop before subpixel phase registration.

The adapters follow the official [OpenCV template-matching](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html), [phase-correlation](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html), [ECC](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html), and [SIFT matching](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html) documentation, plus the official scikit-image [`match_template`](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.match_template) and [`phase_cross_correlation`](https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation) APIs.

The measured release environment records OpenCV and scikit-image versions in every report. OpenCV is used under Apache-2.0. scikit-image is used under BSD-3-Clause. The adapters call public library APIs; their orchestration is checked in at `src/drift_sense/external_baselines.py`.

## Reproduction

The independent library is optional and does not enlarge the inference install. Python 3.10 resolves scikit-image 0.25.2; Python 3.11–3.14 resolve 0.26.0. The transitive pins in `requirements-comparison.txt` have matching version markers and were dry-resolved for each supported Python version.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-comparison.txt
```

Run one suite:

```bash
python compare_external_baselines.py \
  --data-dir /path/to/report/iid \
  --method all \
  --metralign-report-dir results/frozen/reports \
  --output results/comparisons/external/iid.json \
  --quiet
```

Each schema-v2 result contains the full settings, software licenses and versions, manifest and image hashes, per-sample predictions, unresolved reasons, and per-suite metrics. When an archived Metralign report is supplied, the comparison also embeds a compact per-sample extraction (ID, suite, architecture, truth, prediction, error, and runtime) plus the SHA-256 of the untouched source report. Paths are repository-relative or input-relative locators; aggregation never needs the original absolute path.

Merge the OpenCV and independent-library suite reports only after every expected adapter is present:

```bash
python aggregate_external_comparison.py \
  --input-dir results/comparisons/external \
  --label frozen \
  --output results/comparisons/external-registration-frozen.json
```

The default expected set is all six external adapters. A custom study must repeat `--expected-method` for its complete predeclared set. Aggregation rejects a missing or unexpected method, duplicate samples, unequal sample coverage, arithmetic mismatches, inconsistent dataset bindings, and partial archived-Metralign evidence instead of silently intersecting whatever happened to finish.

Recorded outputs:

- `results/comparisons/external-registration-frozen.json`: pooled frozen result and per-suite metrics;
- `results/comparisons/external-registration-development.json`: pooled archived development result and per-suite metrics;
- `results/comparisons/external/*.json`: OpenCV per-sample frozen records;
- `results/comparisons/independent/*.json`: scikit-image per-sample frozen records;
- `results/comparisons/development-archived/*.json`: complete per-sample archived development records;
- `results/comparisons/external-registration-independent-fresh40-development.json`: development-only independent-renderer aggregate;
- `results/comparisons/development/independent-fresh-40*.json`: its complete per-sample records.

The schema-v2 aggregate files retain SHA-256 hashes of every source result and recompute archived Metralign metrics from the embedded compact rows. Numeric search bounds live in each source report's `configuration`; method-specific constants and license sources live in `method_metadata` and `external_software`. CI installs the optional comparison environment on Python 3.10 and 3.14 and exercises the adapters and aggregation guards.

## Considered but not included

SimpleITK/Elastix was evaluated as a candidate second independent framework. It was not included in this fixed comparison. SimpleITK exposes a configurable intensity-registration framework in physical space, while Elastix requires a parameter map. The PNG inputs do not carry calibrated physical spacing or an initial transform, and the two images cover unequal fields of view. A credible adapter would therefore require new choices for initialization, metric, pyramid, optimizer, masks, transform family, and per-axis spacing. Those choices create a tuning study rather than an off-the-shelf baseline. Adding one after inspecting the reporting results would overstate breadth. The boundary and rationale follow the [SimpleITK registration model](https://simpleitk.org/index.html) and its [Elastix parameter-file interface](https://simpleitk.org/doxygen/latest/html/Elastix_2Registration_2elx_8cxx-example.html).

## Interpretation boundary

These are established general-purpose registration primitives with transparent task adapters. They are not asserted to be the best possible tuning of each library. Periodic structures violate the distinct-feature assumption behind SIFT/RANSAC and create many valid correlation maxima. Conversely, Metralign was designed for this structure. The comparison quantifies that specialization; it does not establish superiority on unrelated registration tasks or real microscopy.
