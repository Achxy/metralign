# Safety-audit input

`development-diagnostics.json` is a lossless extraction of the fields consumed by `audit_safety_policy.py` from the seven archived development reports: source report name, sample ID, coordinate error, match confidence, ambiguity flag, score-tie flag, residual evidence, and transform stability.

The 700 rows were copied without recomputation. The artifact records the SHA-256 and exact row count of every source report; those hashes match the provenance recorded in `results/comparisons/safety-audit.json`. It is an audit input, not a new benchmark report, and it contains no reconstructed or simulated measurements.
