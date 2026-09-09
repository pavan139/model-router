# Logged-bandit evaluation record

`bandit.py` implements off-policy evaluation for fixed stateless routing/admission policies with explicit per-request cost caps. This is separate from the existing sequential batch-budget allocator; no one-step importance weighting is applied to that allocator.

The vertical includes epsilon-randomized logging with actual probability vectors, observed-outcome-only JSONL logs, IPS/SNIPS and cross-fitted doubly robust quality/cost estimates, per-action ridge outcome models, support and ESS diagnostics, confidence intervals, cap enforcement, logged replay CLI, full-information synthetic target values and a label-informed per-request quality oracle. Unsupported actions produce no estimates; low evidence produces an exported unusable report and exit status 1.

## Reproduce actual runs

After `python -m pip install -e .`:

```bash
python -m model_router bandit-generate --samples 1500 --seed 17 --epsilon 0.3 --output runs/bandit-logs.jsonl
python -m model_router bandit-evaluate --logs runs/bandit-logs.jsonl --policy adaptive --seed 17 --folds 5 --output runs/bandit-adaptive.json
python -m model_router bandit-demo --samples 1500 --seed 17 --epsilon 0.3 --folds 5 --output runs/bandit-experiment.json
python -m unittest discover -s test -v
```

All commands returned **0**. The suite passed **47 tests**, retaining all 31 earlier tests. No model or API was called. The experiment generates bounded synthetic potential outcomes, randomizes only one logged action per request, and keeps unchosen outcomes away from the estimator.

For the fixed adaptive target, the 1,500-request run produced:

| Metric | Exact synthetic target value | IPS | SNIPS | Cross-fitted DR |
| --- | ---: | ---: | ---: | ---: |
| Mean quality | 0.477333 | 0.470392 | 0.473689 | 0.475168 |
| Mean configured cost units | 1.812846 | 1.797935 | 1.810538 | 1.814636 |

The effective sample size was **1,305.59**, with **1,317** matched events, maximum observed weight **1.25**, and no support violations. The quality DR Wald interval was **[0.448765, 0.501572]**; the conditional finite-sample IPS quality interval was **[0.435623, 0.505161]**. Cost DR's Wald interval was **[1.706107, 1.923164]**. These are different interval types with different assumptions, not evidence from a multi-run coverage study.

The full-information per-request quality oracle was **0.498667**. It knows all synthetic potential outcomes and respects each request's cap; it has no batch budget and is unavailable to an actual router. All five target policies are reported, including an example where DR's absolute quality error is not better than IPS. No universal variance-reduction claim is inferred from one seed.

## Independent estimator checks

A new test enumerates all nine joint logging-action assignments for two requests under nonuniform propensities, independently computes their probabilities, and verifies expected IPS and cross-fitted DR quality/cost equal full-information target values. This directly exercises the randomized estimator and fold-exclusion formula, including empty-action nuisance folds. Other tests check exact IPS/SNIPS arithmetic, unclipped estimates outside natural reward ranges, held-out label perturbation, no-support refusal, zero-weight undefined SNIPS, zero-cap abstention, impossible target admission, malformed propensities, order-stable folds and an observed-only CLI replay.

## Assumptions and limits

Logging probabilities must be true randomized probabilities; external declarations are not attested. Requests must be independent, without interference, and target policies fixed before outcome inspection. Recorded-context diagnostics cannot prove overlap for unseen contexts. Wald intervals are marginal and asymptotic; cross-fitted nuisance regularity and overlap matter. Conditional IPS Hoeffding intervals cover finite realized contexts/potential outcomes under independent randomized actions, not distribution shift. No automatic simultaneous-policy, cluster, repeated-peeking or sequential-bandit guarantee is provided.

The cost unit is an explicit synthetic/replay policy, not provider currency. Missing support is never repaired by a nuisance-model guess, and no weight clipping silently changes the estimand. See [estimators-and-routing.md](estimators-and-routing.md) for equations, complexity and failure semantics.

## Extreme-weight regression

A bounded cross-review reproduced an overflow with positive propensity `1e-200`. The numerical implementation now computes scaled ESS and moments: that exact two-request case returns ESS **1**, IPS quality **5e199**, finite confidence intervals and JSON without non-finite numbers. A four-request case with propensity `1e-308` retains ESS **4** and SNIPS **1** despite its raw weight sum overflowing; the sum is explicitly null and its logarithm remains finite. Nonrepresentable `1e-320` ratios and weighted-cost overflow are refused explicitly. Three added regression tests cover these cases without clipping or changing the estimand.
