# Standalone generator-transfer suite

The `independent_renderer` suite tests whether Metralign transfers to a second
synthetic implementation. It is development evidence. It is not part of the
frozen 1,400-pair report and does not replace evaluation on acquired SEM/TEM
imagery.

## Separation boundary

The suite lives in
[`src/drift_sense/independent_renderer.py`](../src/drift_sense/independent_renderer.py).
That module does not import the primary generator's architecture, coordinate,
distortion, capture, alternate-capture, or dataset modules:

- `drift_sense.architectures`
- `drift_sense.geometry`
- `drift_sense.distortions`
- `drift_sense.sem_render`
- `drift_sense.sem_render_alt`
- `drift_sense.dataset`

An AST-level regression test enforces this boundary. The generator instead
defines its own physical-nanometre homography, signed-distance layouts,
per-feature integer noise, detector transfer functions, acquisition ordering,
and reference/search sampling paths.

The implementations still share the task contract: DRAM/FinFET category names,
grayscale reference/search PNGs, a nominal 0.1 scale ratio, and evaluator field
names. The images remain synthetic. Code-path separation should not be read as
external experimental independence.

## Capture differences

| Component | Primary frozen generator | Standalone suite |
|---|---|---|
| Coordinates | Analytic sensor/world transform | 3 × 3 physical-nanometre homography with shear and perspective |
| Persistent variation | Sine-hashed cells/segments | SplitMix64 feature keys and PCG64 global processes |
| DRAM layout | Staggered circular contacts | Anisotropic brick-array pads, sparse bridges and missing pads |
| FinFET layout | Orthogonal smooth fin/gate field | Wandering lines, cut blocks and sparse intersection contacts |
| Reference sampling | Area or alternate polyphase path | Raised-cosine intra-pixel integration |
| Search sampling | Lanczos or alternate polyphase path | Charge-spread filtering and phase-centred cubic sampling |
| Detector model | Shared acquisition parameter model | Role-specific PSF, edge response, charging, stripes, shot noise and read noise |

## Generate and bind a suite

```bash
python generate_independent_suite.py \
  --architecture both \
  --num-pairs 100 \
  --image-size 1000 \
  --supersample 2 \
  --difficulty medium \
  --seed 2026081701 \
  --output-dir data/independent-renderer
```

The command writes:

- `manifest.jsonl`, with the source and configuration hash in every record;
- one JSON sidecar and two image files per pair;
- `suite-metadata.json`, with the source, configuration, manifest, image, and
  aggregate dataset hashes;
- `SHA256SUMS`, binding every manifest, metadata, sidecar, and image artifact.

`verify_independent_suite()` recomputes these bindings and rejects source,
configuration, manifest, image, or ledger drift.

## Evaluate

```bash
python evaluate.py \
  --data-dir data/independent-renderer \
  --method full \
  --output results/independent-renderer/report.json \
  --quiet
```

Report this suite separately from the frozen benchmark. Record the generator
commit, suite seed, dataset digest, sample count, and complete error
distribution. Do not tune the localizer on this suite and then call the result
held out.
