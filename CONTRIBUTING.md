# Changes to routing decisions

Decide which of the two problems a change addresses: the sequential batch-budget allocator or the stateless logged-bandit evaluator. Their data and estimands differ. Do not apply one-step IPS to the allocator's changing remaining-budget state.

For batch routing, add a structurally distinct expression case when changing features or splitting. The router must not inspect answers or candidate output text before dispatch. Calibration may choose the threshold; held-out test labels may not. Preserve exact pre-dispatch budget equality cases.

For logged evaluation, use a hand-calculated propensity example or enumerate a tiny action space. Check support refusal, zero weights, fold exclusion, numerical overflow, and both quality and cost. An estimator outside the natural reward range can be valid; silently clipping it changes the estimand.

Run `python -m unittest discover -s test -v`, then `python tools/check_routing.py --require-lint --wheel`. `tools/probe_routing_edges.py` runs selected regressions in copied source. When updating `runs/`, keep the source log, frozen policy, split identities, budget, confidence-interval assumptions, and unfavorable results together.
