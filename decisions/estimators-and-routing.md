# Stateless contextual-bandit off-policy evaluation

The new subsystem estimates the quality and cost of a fixed, stateless routing/admission policy from genuinely randomized action logs. Its inputs contain only the chosen action's observed outcome. Synthetic full-information outcomes are kept outside the estimator and used only for experiment diagnostics.

## Scope and contract

A request context contains a unique ID, bounded pre-action features, upper cost quotes for `abstain`, `small` and `large`, and an explicit per-request cap. Both logging and target policies assign a normalized action distribution with zero probability beyond that cap. Abstention has exactly zero quality and cost. Realized quality lies in `[0,1]`; realized cost lies between zero and the selected pre-dispatch quote.

The logging vector records the actual probabilities used to draw the action, not an estimated classifier score. The synthetic logger mixes fixed adaptive exploitation with epsilon-uniform exploration over admitted actions, using a separate random stream for action sampling. External collectors must supply truthful propensities; the loader cannot attest their honesty.

Requests must be independent, without interference, and target policies must be fixed before inspecting these evaluation outcomes. Duplicate request IDs are rejected, but different IDs do not prove independence. User/customer clusters need cluster-level splitting and inference. Support is required throughout the target context distribution; diagnostics can detect violations on recorded contexts, not certify all unseen contexts.

**The existing batch-budget router is excluded.** Its remaining budget changes with earlier selected actions, so one-step contextual-bandit weights do not evaluate that sequential policy. This module never accepts batch state or invokes its allocator. It requires an explicit `StatelessPolicy` and enforces each request's cap separately.

## Estimators

For logged action `A_i`, propensity `q_i(A_i|X_i)`, target distribution `pi_i`, and chosen outcome `Y_i`, define `w_i = pi_i(A_i)/q_i(A_i)`.

`IPS = mean(w_i Y_i)`.

`SNIPS = sum(w_i Y_i) / sum(w_i)`.

For out-of-fold outcome estimates `m_hat_{-fold(i),a}(X_i)`, define

`DR_i = sum_a pi_i(a) m_hat_{-fold(i),a}(X_i) + w_i (Y_i - m_hat_{-fold(i),A_i}(X_i))`.

The doubly robust estimate is `mean(DR_i)`. The same formulas are applied separately to quality and cost. Known correct propensities identify IPS and the DR correction even if the linear outcome model is misspecified. More generally DR consistency requires either a correct propensity mechanism or a suitable consistent outcome model, plus support and regularity; wrong external propensities are not repaired by the name of the estimator. SNIPS has finite-sample ratio bias and is undefined when total weight is zero.

No importance weights or point estimates are clipped. IPS or DR may lie outside a reward's natural range at finite sample sizes; silently clipping them would change the estimator. Missing support returns no IPS/SNIPS/DR values, including no model-only fallback. Low effective sample size and insufficient request counts retain diagnostic estimates but mark the report unusable, causing `bandit-evaluate` to exit 1 after exporting it.

## Cross fitting

Request IDs are sorted, shuffled with an explicit seed and assigned to balanced nonempty folds. Each held-out request receives predictions from per-action ridge models fitted on other folds only. Quality predictions are clipped to `[0,1]`. Cost is modeled as a fraction of that action's known quote, then multiplied by the held-out quote; predictions therefore respect known bounds.

Training uses only observed outcomes for the chosen action. Missing-action training folds use a declared zero nuisance prior and emit diagnostics. They do not invent observations. Positive ridge regularization and partial-pivot Gaussian elimination solve the small normal equations. Abstention's outcome model is exactly zero by contract. Fold/action counts and assignment provenance are exported. Tests perturb a held-out outcome and verify its entire fold's predictions remain unchanged.

## Overlap and confidence intervals

The report includes support violations, minimum supported logging probability, matched events, mean/sum/max observed importance weights, and `ESS = (sum w)^2 / sum(w^2)`. ESS is a diagnostic, not a proof of identification or coverage.

Wald intervals use the empirical variance of IPS scores, ratio influence scores `w_i(Y_i-SNIPS)/mean(w)`, or cross-fitted DR scores. They are marginal, asymptotic intervals. DR inference additionally needs appropriate nuisance convergence/regularity and overlap. Several policies or metrics are not automatically simultaneous; there is no sequential monitoring guarantee.

IPS additionally has a conditional finite-sample Hoeffding interval. Condition on realized contexts and potential outcomes, with fixed target policy and independent randomized logging actions. A valid score range bound for request `i` is

`c_i = max_{supported admitted a != abstain} pi_i(a)/q_i(a) * upper_outcome_i(a)`.

The quality upper outcome is one; cost uses the predeclared quote. Abstention is known zero. Radius

`r = sqrt(log(2/alpha) * sum(c_i^2) / (2 n^2))`

bounds deviation from the finite-context target mean with error probability at most alpha. This is not a bound for unseen-context population shift, and it relies on actual propensities and outcome bounds. Wider intervals under weak overlap are retained honestly. It does not apply to DR scores with cross-fold dependence.

## Experiment separation and oracle

`bandit-demo` generates contexts, potential quality/cost outcomes, randomized chosen actions and observed logs. `evaluate_logged` receives only the observed logs. A separate full-information calculation computes each fixed target's exact value on those same synthetic contexts/potential outcomes. A label-informed quality oracle selects the best admitted action independently per request; it is not a deployable policy or a batch-budget oracle.

Dataset, target-distribution, inference-configuration and implementation hashes bind evaluation artifacts. Fold seed, ridge, alpha and diagnostic thresholds are recorded. Logging seed, epsilon and synthetic potential-outcome hash bind the full experiment. No test outcomes select a preferred policy or tune parameters in the supplied experiment.

## Complexity and failure behavior

With `n` requests, `k` folds, `a=3` actions and feature width `d`, ridge fitting costs `O(k n d^2 + k a d^3)` and storage `O(n d + n a)`. Scalar estimators cost `O(n a)`. Defaults use 1,500 requests, five folds and two features; hard bounds are 5,000 events, ten folds and sixteen features. Imports bound line allocation and total characters before parsing. Invalid propensities, duplicate requests, malformed costs and cap violations fail explicitly. No model downloads, provider calls, or full-model training are performed.

## Numerical representability

Importance weights are not clipped. ESS is computed from weights divided by their maximum, and SNIPS uses the same common scaling. Means and Wald standard errors are accumulated at unit scale; Hoeffding norms divide by sample count before a scaled `hypot`. This avoids squaring representable weights such as `1e200`, and it does not alter any estimator or estimand.

An overflowing raw weight sum is reported as `null` with `weight_sum_overflowed=true` and a finite `log_sum_weights`; ESS and normalized estimators remain available when representable. If a ratio, weighted score, standard error or interval endpoint cannot be represented safely in float64, the report returns `numerical_status=unsupported_float64_range`, `usable=false` and `estimates=null`. Statistical support and numerical representation are reported separately. No infinity/NaN is exported and no propensity floor or substitute model estimate hides the failure.
