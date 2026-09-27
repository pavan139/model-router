# Routing verification ledger

[reports/verification.json](../reports/verification.json) is the current check record. It contains the source digest, Python/tool versions, test counts, lint/format results, CLI checks, and cold-wheel installation outcomes.

The current suite checks structural isolation, calibration and threshold budgets, response provenance, exact budget oracles, propensity arithmetic, fold exclusion, missing support, and extreme weights. Selected source-fault probes are recorded in `runs/review-fault-probes.json`; they are not whole-project mutation coverage.

Run `python tools/check_routing.py --require-lint --wheel` after edits. Historical fixture results in `runs/` remain separate from regenerated check metadata so a layout change cannot be mistaken for a new model experiment.
