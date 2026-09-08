# Two offline evaluation protocols

## Batch budgets

Train on the structural training split and fit calibration/threshold selection on the calibration split. Freeze the artifact before opening test outcomes.

```bash
python -m model_router generate --seed 17 --output artifacts/records.jsonl
python -m model_router train --records artifacts/records.jsonl --budget-units 180 --output artifacts/router.json
python -m model_router evaluate --records artifacts/records.jsonl --router artifacts/router.json --budget-units 180 --output artifacts/evaluation.json
```

The report includes all test cases, including budget abstentions. Read correct count, answered accuracy, cost units, cost per correct answer, probability calibration, and regret to the label-informed upper bound separately. Keep one arrival order fixed across policies. `sweep` freezes the router while changing budgets and arrival orders; it does not select a best configuration using test outcomes.

The checked-in 36-case result solves 25 cases with learned routing at budget 180 and 33 with the offline oracle. The stored sensitivity run includes budgets where fixed-small does better. Those failures are part of the result.

## Logged per-request policies

```bash
python -m model_router bandit-generate --samples 1500 --seed 17 --epsilon 0.3 --output artifacts/action-log.jsonl
python -m model_router bandit-evaluate --logs artifacts/action-log.jsonl --policy adaptive --seed 17 --folds 5 --output artifacts/ope.json
```

The input contains context, the actual logging probability vector, chosen action, and observed outcome. Unchosen synthetic outcomes stay outside estimation. Compare IPS, SNIPS, cross-fitted DR, ESS, support diagnostics, and the applicable intervals. A nuisance prediction cannot repair absent support.

The stored adaptive-policy run has quality truth 0.477333 and DR 0.475168. This is one synthetic realization, not a coverage study or a provider benchmark. Independent action-space enumeration and held-out-label perturbations check estimator behavior. Read [the bandit record](bandit-evaluation-record.md) for assumptions and extreme-weight regressions.
