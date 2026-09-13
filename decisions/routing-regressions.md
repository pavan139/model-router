# Independent review and verification

The audit inspected train/calibration/test isolation, pre-dispatch budgets, exact-oracle correctness and scaling, calibration claims, response provenance and bounded imports. All 17 original tests passed before the following defects were reproduced.

| Finding and original proof | Fix and regression evidence |
| --- | --- |
| Wrapping a training expression in extra parentheses let the same AST enter test without triggering structural leakage. | Canonical AST signatures ignore whitespace, redundant parentheses and integer literal spelling. Parenthesized and hexadecimal/underscore variants are covered by regression tests. |
| Marking one training response RECORDED made `model_evaluation=true` even when every test response was SIMULATED. | Determine evaluation mode from test candidates; fully recorded test sets alone set that flag. Mixed sets and actually selected provenance are counted explicitly. |
| A simulated response accepted arbitrary dictionary `source_model` metadata, causing later report construction to fail. | Reject invented recorded metadata on simulated responses; bound and type-check identities at construction. |
| JSONL iteration allocated an entire untrusted line before checking its length. Artifact loading was also unbounded. | Bounded reads before parsing, per-line/total-character/record caps, bounded artifacts, duplicate-key/non-finite JSON rejection. An instrumented stream rejects unbounded read requests in tests. |
| Threshold artifacts omitted the calibration budget, concealing transfer to a different budget regime. | Schema-2 artifacts retain that budget; evaluation reports mismatches. Schema-1 artifacts fail with a retraining instruction. |
| The exact oracle repeatedly rescored labels inside a pseudo-polynomial budget-state dynamic program. | Exploit the exact unit-value objective: select the cheapest correct response per case, then buy cheapest cases first in O(n log n). Export the selected plan and spend. |

The new frozen-router `sweep` capability measures budget transfer and arrival-order sensitivity without selecting parameters from test outcomes. Test calibration diagnostics now include raw-versus-calibrated Brier/ECE/log loss; no claim that calibration always helps is embedded in the implementation.

## Commands actually run

From this standalone directory after editable installation (the collection used its `.venv/bin/python`):

```bash
python -m unittest discover -s test -v
python -m model_router generate --output policies/response_fixture.jsonl
python -m model_router train --records policies/response_fixture.jsonl --budget-units 180 --output runs/router.json
python -m model_router demo --router runs/router.json --output runs/demo.json
python -m model_router evaluate --records policies/response_fixture.jsonl --router runs/router.json --budget-units 180 --output runs/evaluation.json
python -m model_router sweep --router runs/router.json --budget-units 180 --budgets 0 40 80 180 --orderings 5 --output runs/sensitivity.json
python tools/probe_routing_edges.py --output runs/review-fault-probes.json
```

Results: **31 tests pass**; all commands above return **0**. The oracle matches **60/60 exhaustive tiny joint-choice comparisons** across 12 generated four-case problems and five budgets, including exact equality and zero budget. Another test verifies each response is scored once independently of the number of potential budget states. Root Ruff `check --select E9,F` and `format --check` pass on this project.

On the actual simulated seed-17 test split (36 cases, budget 180): small **13 correct/54 units**, large **23/180**, heuristic **23/180**, learned **25/180**, exact offline oracle **33**. Calibrated test Brier is **0.1772929447**, ECE is **0.0990296976**, and the Brier difference versus raw prediction is **−0.0243235916**. These numbers describe the executed synthetic replay only.

With the router frozen across five arrival orders, learned correct-count ranges are **0–0 at budget 0**, **5–8 at 40**, **10–12 at 80**, and **24–27 at 180**. At budget 40 fixed-small ranges **8–12**. These results expose budget/order dependence rather than establishing universal routing improvement.

## Test-quality evidence and remaining limits

TestProof was not installed; `npx --no-install testproof --version` failed without downloading it. The initial collection audit had no HEAD commit for a git-based comparison. No TestProof score is claimed.

The explicit isolated-copy source-fault experiment caught **5/5 selected probes**: lexical rather than AST split checks, off-by-one exact-oracle budget, off-by-one dispatch affordability, training-only recorded provenance inflating evaluation claims, and unbounded line reading. Baselines stayed enabled; there were no timeouts. This is a small behavior-focused experiment, not whole-project mutation coverage.

Calibration fitting and threshold selection still reuse the calibration split, which may overfit it; the test split remains untouched until scoring. AST shape isolation does not establish algebraic/semantic independence. Artifact metadata is declarative, not attested external training history. The greedy oracle is exact only for independent cases with equal one-point reward; unequal rewards require another algorithm. Imported recorded metadata is supplied by collectors, not independently authenticated. All published default responses/costs are simulated, no provider was called, no model was downloaded, no full ML training ran, and no production savings or deployment readiness is claimed.
