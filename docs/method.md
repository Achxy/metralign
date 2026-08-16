# Method

Metralign's `full` localizer estimates the center of a finely sampled reference field in a wider-field, coarser-sampling search image. It is training-free. Coordinates are reproducible within a fixed CPU/software environment and numerical backend.

## Coordinate contract

On success, the inference CLI prints two values formatted to six decimal places:

```text
x y
```

Coordinates use the search image's top-left origin. `x` increases rightward and `y` increases downward. Diagnostics go to stderr only when `--diagnostics` is set.

## Default `full` pipeline

### 1. Robust normalization

Each image is clipped to its 1st–99th percentile range. The implementation normalizes by the median and median absolute deviation, with a standard-deviation fallback for low-MAD inputs.

### 2. Lattice calibration

Zero-padded one-dimensional spectra estimate the two lattice fundamentals. Phase drift across scan lines supplies relative scale, rotation, search-image pitch, confidence, and a separable-axis test.

The default bounds are:

- nominal reference-to-search scale: `0.1`
- absolute scale radius: `±0.006`
- rotation radius: `±3.0°`

Low-confidence or inconsistent phase estimates trigger a bounded coarse-to-fine residual transform search. Refinement never expands the configured scale or rotation bounds.

### 3. Periodic-backbone cancellation

The method warps reference templates at candidate transforms within the configured bounds. It then applies the same symmetric transfer function to reference and search. Each image is differenced against the average of samples shifted by `±round(pitch_x)` columns or `±round(pitch_y)` rows.

Separate `period_x` and `period_y` channels suppress the repeated backbone without image-specific Fourier-notch masks. Pitch estimates can come from the two-dimensional spectrum or a nominal fallback when spectral evidence is insufficient.

### 4. Residual matching

The localizer measures the fraction of residual energy explained by a sequence that is constant along each line.

- When the lattice axes are separable and both channels are sufficiently line-constant, it correlates one-dimensional sequences and corrects for measured lattice slant.
- Otherwise it preserves two-dimensional evidence, correlates both residual channels, and ranks coordinates by their arithmetic mean.

The global maximum of the fused map is accepted as residual evidence only if both channel scores at that coordinate clear the evidence floor. Otherwise the method uses robust-normalized backbone fallback. This prevents an isolated one-channel response from validating the fused maximum.

### 5. Ambiguity handling

Robust-normalized backbone correlation is used when spatial residuals are disabled, when template residual strength is at most `0.03`, or when the weakest channel at the joint-path maximum is below `0.25`. The ambiguity decision scans every non-maximum-suppressed peak inside the configured score band, including peaks beyond the ordinary top-K list.

Raw peaks remain authoritative for center-nearest selection. If the estimated real-space basis is reliable, the same peaks are grouped by nearest integer-lattice offset for diagnostics. Group evidence is marked reliable only when its representatives cover at least 65% of raw tied peaks.

A score tie alone does not trigger center-nearest selection. It must also be supported by at least one secondary signal:

- local score-neighborhood variation,
- low transform-estimate confidence, or
- low residual evidence.

Diagnostics report raw and grouped counts, group coverage, residual evidence, selected score, best score, runner-up score, ambiguity status, and any reliable lattice offset. The schema field `transform_stability` contains the spectral phase-confidence value; `local_perturbation` contains the standard deviation of the best peak's 3×3 score neighborhood.

### 6. Subpixel refinement

The default path attempts bounded independent three-sample parabolic interpolation along the horizontal and vertical axes through the selected peak. Edge, flat, or non-concave neighborhoods remain discrete. The separable path refines its one-dimensional axis scores and corrects for measured slant. `dft` and `none` are controlled development alternatives.

## Methods exposed by the CLI

| Method | Purpose |
|---|---|
| `full` | Default release pipeline described above. |
| `baseline0` | Robust normalization, nominal 0.1× area resize, normalized template correlation, and peak refinement. |
| `multiscale` | Legacy scale/rotation grid ablation. |
| `structure_gradient` | Legacy structural-plus-gradient ablation. |
| `structure_residual` | Legacy structural/Fourier-residual ablation. |

The legacy methods are not cumulative stages of `full`. Use `benchmark_pipeline.py` for shared-pipeline stage controls.

## Configuration

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

Pipeline controls exposed through evaluation and benchmarking include phase calibration, evidence representation, spatial residuals, lattice diagnostics, ambiguity handling, and `parabolic`/`dft`/`none` subpixel refinement.

## Computational properties

- CPU implementation; no GPU path is required.
- No learned weights or network service.
- Coordinate results are reproducible for fixed input bytes, source implementation, dependencies, arguments, CPU/software environment, and numerical backend. Cross-platform bitwise identity is not promised.
- Primary evaluation runtime is wall time immediately around `localize()` and excludes image decoding and report construction.
