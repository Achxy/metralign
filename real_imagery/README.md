# Real-microscopy development evaluation

This directory is a source-bound evaluation track for real acquired microscopy. It is deliberately separate from the frozen synthetic reporting split. Its results must not be pooled with, substituted for, or described as additional cases from that split.

## Evidence levels

The protocols support different conclusions:

1. **Carinthia production-wafer SEM self-consistency.** A fixed, class-balanced selection of 24 acquired images supplies deterministic digital crops. This is exact digital construction truth, not population-representative defect sampling or independent reacquisition.
2. **Boiko real SEM digital-crop self-consistency.** A deterministic crop from an acquired SEM image is enlarged and localized back into that same image. Coordinates are exact digital construction truth. This probes real texture but not independent reacquisition.
3. **Registered real TEM pair/crop localization.** A reference is constructed from the publisher-provided sharp `GT` member and localized in an independently acquired, motion-blurred `Low` member. The publisher selected and registered those frames before release. Coordinates are exact in that registered image coordinate system and digital crop construction, but residual registration error is unknown. This is not stage ground truth or a native cross-magnification test.
4. **Native SEM multimag agreement.** Same-area publisher labels at 50k×, 100k×, and 200k× are tested against a fixed SIFT/RANSAC proxy. The proxy has an explicit quality gate. Agreement is neither coordinate ground truth nor an accuracy measurement.

No source, candidate grid, scoring rule, threshold, or representative-visual rule is selected from localization outputs. Because the publisher's registered TEM intersections are short horizontal strips with blank padding, TEM query crops are selected reproducibly from each sharp GT alone: a fixed 9×9 interior grid is ranked by mean absolute Gaussian high-pass residual, then the five highest-scoring non-overlapping crops are assigned to the five sorted Low frames. The public success plate selects the median and nearest-P95 successful case per prespecified source group from a completed report and says so explicitly.

## Sources and licenses

| Track | Primary source | License | Pinned download |
|---|---|---|---|
| Production-wafer SEM | Kofler et al., [dataset DOI 10.5281/zenodo.10715190](https://doi.org/10.5281/zenodo.10715190) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `data.zip`, 133,840,870 bytes, SHA-256 `02436de8c2d6b0c7eabdcdcf133ab5f17e59a0e3de56ebffc4cb6b2acb771490` |
| Registered MiniTEM Low/GT | Wieslander, Wählby, and Sintorn, [dataset DOI 10.5281/zenodo.4113244](https://doi.org/10.5281/zenodo.4113244); [paper DOI 10.1371/journal.pone.0246336](https://doi.org/10.1371/journal.pone.0246336) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `TrainTestVal.zip`, 629,335,440 bytes, MD5 `2128033df6437e5d9bcdc0d4796a7b94` |
| Ordered real FE-SEM | Boiko et al., [dataset DOI 10.6084/m9.figshare.11783661.v1](https://doi.org/10.6084/m9.figshare.11783661.v1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Nine individually pinned TIFFs |
| Disordered real FE-SEM | Boiko et al., [dataset DOI 10.6084/m9.figshare.11783667.v1](https://doi.org/10.6084/m9.figshare.11783667.v1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Nine individually pinned TIFFs |

The FE-SEM acquisition and same-area magnification organization are described in the [Scientific Data article, DOI 10.1038/s41597-020-0439-1](https://doi.org/10.1038/s41597-020-0439-1). Exact source identities and selection rules are in `carinthia_source.json`, `sources.json`, and `paired_tem_source.json`.

## Reproduce

Run from the repository root with the project environment. Raw data should live outside Git; these examples use an external volume.

```bash
export PYTHONPATH=src:.
export REAL_DATA=/Volumes/External/metralign-real-imagery-2026-08-17
export CARINTHIA_ARCHIVE=/Volumes/External/metralign-real-imagery-2026-08-17/carinthia/data.zip

.venv/bin/python -m real_imagery.download_subset \
  --output-dir "$REAL_DATA/boiko-sem"

.venv/bin/python -m real_imagery.download_paired_tem \
  --output-dir "$REAL_DATA"

.venv/bin/python -m real_imagery.evaluate_real_imagery \
  --data-dir "$REAL_DATA/boiko-sem" \
  --carinthia-archive "$CARINTHIA_ARCHIVE" \
  --output real_imagery/results/real-sem-report.json

.venv/bin/python -m real_imagery.evaluate_registered_tem \
  --archive "$REAL_DATA/TrainTestVal.zip" \
  --output real_imagery/results/registered-tem-report.json

.venv/bin/python -m real_imagery.make_real_imagery_evidence_plate \
  --sem-report real_imagery/results/real-sem-report.json \
  --tem-report real_imagery/results/registered-tem-report.json \
  --sem-data-dir "$REAL_DATA/boiko-sem" \
  --tem-archive "$REAL_DATA/TrainTestVal.zip" \
  --carinthia-archive "$CARINTHIA_ARCHIVE" \
  --output real_imagery/results/real-microscopy-success-plate.png

.venv/bin/python -m real_imagery.verify_report \
  --report real_imagery/results/real-sem-report.json \
  --sem-data-dir "$REAL_DATA/boiko-sem" \
  --carinthia-archive "$CARINTHIA_ARCHIVE"

.venv/bin/python -m real_imagery.verify_report \
  --report real_imagery/results/registered-tem-report.json \
  --tem-archive "$REAL_DATA/TrainTestVal.zip"

.venv/bin/python -m pytest -q real_imagery/tests
```

The 629 MB Zenodo transfer is resumable in `download_paired_tem.py`. Where `aria2c` is available, the same pinned target can be fetched faster with parallel HTTP ranges before running the downloader in verification mode by pointing it at the populated directory:

```bash
aria2c --continue=true --max-connection-per-server=16 --split=16 \
  --min-split-size=4M --file-allocation=none --auto-file-renaming=false \
  --allow-overwrite=false --dir="$REAL_DATA" --out=TrainTestVal.zip \
  https://zenodo.org/api/records/4113244/files/TrainTestVal.zip/content

.venv/bin/python -m real_imagery.download_paired_tem --output-dir "$REAL_DATA"
```

Each evaluator refuses to overwrite a report, verifies all downloaded bytes against the pinned source manifest, records every selected input member's cryptographic digest, records the implementation/protocol fingerprints and environment, executes each prediction twice, and exposes execution failures in aggregate counts. Each PNG has a JSON sidecar and embedded report digest; the plate builder re-verifies every shown source against the report.

## Results

These are development-track results and are not pooled with the frozen synthetic split. `≤1 px` and `≤5 px` are evaluated in the registered/search image coordinate system. Native SEM rows are proxy disagreement, not error against physical ground truth.

| Evidence | Method | Completed / attempted | Median px | P95 px | Maximum px | ≤1 px | ≤5 px | Declared fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Registered real TEM | full interface | 70 / 70 | 0.803 | 2.086 | 2.899 | 43 / 70 | 70 / 70 | 70 / 70 |
| Registered real TEM | baseline0 | 70 / 70 | 0.803 | 2.086 | 2.899 | 43 / 70 | 70 / 70 | 0 / 70 |
| Carinthia wafer SEM | full interface | 24 / 24 | 0.025 | 0.274 | 3.058 | 23 / 24 | 24 / 24 | 13 / 24 |
| Carinthia wafer SEM | baseline0 | 24 / 24 | 0.008 | 0.192 | 3.058 | 23 / 24 | 24 / 24 | 0 / 24 |
| Boiko SEM exact digital crop | full interface | 30 / 30 | 0.015 | 0.758 | 1.121 | 29 / 30 | 30 / 30 | 0 / 30 |
| Boiko SEM exact digital crop | baseline0 | 30 / 30 | 0.006 | 0.057 | 0.103 | 30 / 30 | 30 / 30 | 0 / 30 |
| Native SEM multimag proxy agreement | full interface | 18 / 18 | 0.400 | 2.364 | 2.644 | — | 18 / 18 | 0 / 18 |
| Native SEM multimag proxy agreement | baseline0 | 18 / 18 | 0.413 | 2.016 | 3.092 | — | 18 / 18 | 0 / 18 |

Every nominal 0.1× TEM query is only 20–28 pixels high and cannot support one-period residual estimation. The `full` interface therefore records `periodic_model_unsupported`, sets review and conservative-abstention diagnostics, assigns zero absolute-site confidence, and dispatches coordinate estimation to the existing `baseline0` matcher. Its TEM coordinates are consequently identical to baseline0. This proves deterministic interface completion and registered-coordinate performance; it does not show benefit from the periodic stage.

Machine-readable evidence:

- `results/real-sem-report.json` — 24 balanced Carinthia crops, 30 Boiko exact digital crops, and 18 native multimag pairs per method.
- `results/registered-tem-report.json` — 70 registered Test pairs, 84 unique bound archive members, both methods, fallback diagnostics, and complete error rows.
- `results/real-microscopy-success-plate.png` with its JSON sidecar — six mechanically selected successful cases spanning Carinthia, ordered/disordered Boiko SEM, calibration-grid TEM, and kidney TEM.

## Limitations

- The TEM dataset was released for deblurring, not relocalization. Its same-field `Low`/`GT` pairing becomes a localization problem only through the declared digital crop.
- The GT-only informativeness rule avoids blank intersection padding but favors locally structured query regions. It does not measure performance on uniform queries and is not a random spatial sample.
- The publisher reports manual selection, manually initialized ECC registration, intersection cropping, and GT downsampling for the prepared TEM data. The evaluator retains the archive's single-channel `uint16` arrays without clipping or per-image rescaling. It cannot quantify residual pair-registration error.
- The SEM exact-coordinate cases reuse one acquisition and therefore cannot estimate reacquisition robustness. The native multimag comparison uses a feature proxy, not physical truth, and rejected proxy pairs remain visible rather than being silently treated as failures or successes.
- These four public dataset records contain limited instruments, specimens, acquisition settings, and laboratories. They do not establish broad real-world deployment performance.
