# Ambiguity safety and external priors

Metralign reports two different confidence questions:

- `confidence` describes the selected match peak.
- `decision_support.absolute_site_confidence` describes whether the image identifies a unique absolute lattice site.

The latter is deliberately `0.0` for a detected periodic tie. It is a diagnostic score, not an empirical probability. A sharp local match can still select the wrong member of a repeated lattice family.

## Default behavior

The inference contract is unchanged: a default invocation still writes exactly two numbers to stdout, and the default selection prior is the image center. The center/prior fallback is applied only when a score tie is corroborated by low residual evidence or transform instability. Local peak perturbation is derived from the same score map as the tie, so it remains diagnostic but cannot authorize an absolute-site override by itself.

This gate leaves every coordinate and every accuracy metric in the frozen 1,400-pair report exactly unchanged. On the separately implemented 20-pair development renderer, it corrects four score-best DRAM predictions that had previously been replaced by a distant center-nearest site: success within 0.5 px rises from 13/20 to 17/20. The remaining three errors have an incorrect score-best site; they were not tuned away, and both review policies flag their score ties.

```bash
python infer.py \
  --reference reference.png \
  --search search.png \
  --diagnostics
```

Diagnostics are written to stderr. Additions are backward-compatible fields:

- `decision_support.status`: `resolved`, `ambiguous`, or `review`;
- `decision_support.review_recommended`: selective review policy;
- `decision_support.conservative_abstention_recommended`: true for any detected absolute-site tie (`ambiguity_evidence.score_tied`), whether or not the center/prior fallback was applied;
- `hypothesis_count`: number of tied score-map maxima;
- `hypotheses`: a bounded list containing the selected, score-best, and prior-nearest alternatives;
- `hypotheses_truncated`: whether more alternatives exist.

## Versioned review policy

Policy `metralign-absolute-site-review-v1` reviews an ambiguous result when either condition holds:

- residual evidence is at least `0.15`; or
- transform stability is below `0.95`.

The residual threshold is the archived development maximum among tied cases (`0.1382027417`), rounded outward to the next `0.05`. The transform threshold is the archived development minimum (`0.9544444166`), rounded down to `0.95`. (`score_tied` and the historical `ambiguity_flag` agree on every archived development and frozen sample.) The reporting split was not used to calculate either number. However, the decision to investigate these diagnostics followed inspection of the released failures. The study is therefore a transparent post-release safety audit, not a new frozen accuracy result.

Recompute the audit with:

```bash
python audit_safety_policy.py \
  --output results/comparisons/safety-audit.json
```

The default inputs are public: `evidence/safety/development-diagnostics.json` contains the exact eight columns consumed from all 700 archived development rows, together with the SHA-256 and record count of each untouched source report; the frozen reports and two bound cases live under `results/frozen/`. The audit validates the compact schema, source bindings, per-source counts, row uniqueness, and finite values before deriving the thresholds. Supplying `--development-reports` remains available when the original archived reports are present. The checked-in audit leaves all frozen reports unchanged.

## Optional stage-position prior

An application can supply an approximate reference-center location in search-image pixels:

```bash
python infer.py \
  --reference reference.png \
  --search search.png \
  --prior-center-x 261.5 \
  --prior-center-y 427.1 \
  --diagnostics
```

Both coordinates are required, finite, and inside the search image. The prior affects only the selection among image-supported tied maxima when the independent-evidence gate permits fallback. It does not create new image evidence, suppress a detected tie, or cancel a review recommendation. Diagnostics identify it as `selection_prior_source: user_supplied`.

On the two archived large-error cases, a fixed stage-like prior offset of `(+3, -4)` pixels selects the correct lattice member. Errors fall from `247.665` to `0.114` px and from `313.800` to `0.022` px. This is an oracle-anchored sensitivity check. It is not evidence about the accuracy of any real stage sensor. A prior can select an adjacent period when its error crosses a lattice-cell boundary.
