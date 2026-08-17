# Metralign film visual language

This film is an explanatory animation, not a narrated slide deck. Every shot must
make a technical relationship visible through continuity: an object moves,
splits, transforms, or is measured. Static evidence remains source-bound, but it
is introduced at a scale at which the evidence can actually be read.

## Composition

- One dominant idea per shot. Secondary facts are disclosed only after the
  primary relationship is established.
- The 1920×1080 safe frame is 96 px on every edge. Titles occupy a dedicated
  top band; no panel label may enter it.
- Evidence must occupy at least half the frame when it is the subject. Dense
  plates are never shown side by side. Their authentic pixels are cropped or
  visited sequentially at readable scale.
- Titles use sentence case. All-caps is reserved for brief scope labels.
- Tables are audit artifacts, not film graphics. Comparisons use direct labels,
  axes, and progressive reveals.

## Typography

- Avenir Next is the text family; mathematical notation is rendered through
  LaTeX. Menlo is reserved for real terminal output and commands.
- Five sizes cover the film: 52 title, 38 statement, 30 body, 22 annotation,
  17 provenance. No text intended to carry the argument is smaller than 22.
- Text blocks are left aligned unless the mathematical geometry itself calls
  for centering. Line breaks are authored, never obtained by squeezing a long
  sentence into a fixed rectangle.
- Mathematical variables retain color across diagrams, equations, and plots.

## Motion

- Use `TransformMatchingTex` when an equation changes form, and use
  `ReplacementTransform` when one representation becomes another.
- Use `ValueTracker` or direct coordinate interpolation when a measured value or
  selected point changes. Do not animate a decorative substitute.
- Camera motion is semantic: zoom into a measured peak, a candidate, or a real
  microscopy case. It is never ambient.
- Entrances overlap gently; exits are faster. Avoid bounce, elastic motion,
  spinning, and ornamental movement.
- Hard cuts separate concepts. Within a concept, preserve the object that
  carries the explanation.

## Evidence rules

- No generated scientific imagery, decorative plots, or illustrative spectra.
- Every plotted value is read from a checked-in report or exported evidence
  record. Every overlay is attached through the same coordinate transform used
  to display its source image.
- Synthetic, acquired, publisher-registered, proxy, fallback, and development
  scopes remain visibly distinct.
- The transfer scene shows representative evidence sequentially and names the
  population and fallback boundary; it never implies microscope-stage truth.

## Frame review

Every scene must pass these checks at 1920×1080 and on a 480×270 contact sheet:

1. no object crosses the safe frame;
2. no title, label, or annotation overlaps another object;
3. the primary image or plot remains readable without pausing and zooming;
4. equations use LaTeX and correct source notation;
5. the most important element remains obvious under a squint test;
6. the last frame of each narration cue is a coherent pause frame.
