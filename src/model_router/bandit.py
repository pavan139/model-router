"""Logged contextual bandits for STATELESS per-request routing.

Never feed the existing budget-coupled sequential router into these one-step
estimators: its state and action-dependent future allocation violate this API.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from statistics import NormalDist
from typing import Callable
from .data import strict_json

ACTIONS = ("abstain", "small", "large")
MAX_EVENTS = 5000
VERSION = "stateless-ope-v1.1"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value, lo, hi):
    return type(value) in (int, float) and math.isfinite(value) and lo <= value <= hi


@dataclass(frozen=True)
class Context:
    request_id: str
    features: tuple[float, ...]
    quoted_costs: tuple[float, float, float]
    cost_cap: float

    def __post_init__(self):
        if type(self.request_id) is not str or not 1 <= len(self.request_id) <= 128:
            raise ValueError("Request IDs must be bounded strings")
        if (
            not isinstance(self.features, tuple)
            or not 1 <= len(self.features) <= 16
            or any(not _number(x, -10, 10) for x in self.features)
        ):
            raise ValueError("Context requires 1 to 16 finite features in [-10,10]")
        if (
            not isinstance(self.quoted_costs, tuple)
            or len(self.quoted_costs) != 3
            or self.quoted_costs[0] != 0
            or any(not _number(x, 0, 100000) for x in self.quoted_costs)
        ):
            raise ValueError("Quotes must be finite costs for abstain, small and large, with abstain zero")
        if self.quoted_costs[1] <= 0 or self.quoted_costs[2] <= 0 or not _number(self.cost_cap, 0, 100000):
            raise ValueError("Candidate quotes must be positive and the per-request cap nonnegative")

    def admitted(self, action: int) -> bool:
        return self.quoted_costs[action] <= self.cost_cap


def validate_distribution(context: Context, probabilities) -> tuple[float, float, float]:
    if (
        not isinstance(probabilities, (tuple, list))
        or len(probabilities) != 3
        or any(not _number(p, 0, 1) for p in probabilities)
    ):
        raise ValueError("Action probabilities must be three finite values in [0,1]")
    if not math.isclose(sum(probabilities), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Action probabilities must sum to one")
    if any(p > 0 and not context.admitted(a) for a, p in enumerate(probabilities)):
        raise ValueError("Policy assigns probability to an action beyond the per-request cap")
    return tuple(probabilities)


@dataclass(frozen=True)
class LoggedEvent:
    context: Context
    action: str
    logging_probabilities: tuple[float, float, float]
    quality: float
    cost: float

    def __post_init__(self):
        if not isinstance(self.context, Context) or self.action not in ACTIONS:
            raise ValueError("Invalid context or logged action")
        validate_distribution(self.context, self.logging_probabilities)
        selected = ACTIONS.index(self.action)
        if self.logging_probabilities[selected] <= 0:
            raise ValueError("Observed action must have positive logging propensity")
        if not _number(self.quality, 0, 1) or not _number(self.cost, 0, self.context.quoted_costs[selected]):
            raise ValueError("Observed quality/cost violate their predeclared bounds")
        if selected == 0 and (self.quality != 0 or self.cost != 0):
            raise ValueError("Abstention must have zero quality and cost")


@dataclass(frozen=True)
class StatelessPolicy:
    policy_id: str
    distribution: Callable[[Context], tuple[float, float, float]]

    def __post_init__(self):
        if type(self.policy_id) is not str or not 1 <= len(self.policy_id) <= 128 or not callable(self.distribution):
            raise ValueError("Stateless policies need an explicit bounded version ID and context-only callable")

    def probabilities(self, context: Context):
        return validate_distribution(context, self.distribution(context))


def target_policy(name: str) -> StatelessPolicy:
    if name not in {"small", "large", "adaptive", "soft_adaptive", "admit"}:
        raise ValueError("Unknown stateless target policy")

    def distribution(context):
        difficulty = min(1.0, max(0.0, context.features[0]))
        if name == "soft_adaptive":
            mass = [0.0, 0.9 - 0.8 * difficulty, 0.1 + 0.8 * difficulty]
            mass = [p if context.admitted(a) else 0.0 for a, p in enumerate(mass)]
            if sum(mass) == 0:
                return (1.0, 0.0, 0.0)
            return tuple(p / sum(mass) for p in mass)
        preferred = 1 if name == "small" else 2 if name == "large" else 1 if difficulty < 0.5 else 2
        if name == "admit" and difficulty > 0.75 and not context.admitted(2):
            preferred = 0
        if not context.admitted(preferred):
            preferred = 1 if name in {"adaptive", "admit"} and context.admitted(1) else 0
        return tuple(float(a == preferred) for a in range(3))

    return StatelessPolicy(f"stateless-{name}-v1", distribution)


def logging_distribution(context: Context, epsilon: float = 0.3) -> tuple[float, float, float]:
    if not _number(epsilon, 0.01, 1):
        raise ValueError("Exploration epsilon must be in [0.01,1]")
    admitted = [a for a in range(3) if context.admitted(a)]
    exploit = target_policy("adaptive").probabilities(context)
    return tuple((1 - epsilon) * exploit[a] + (epsilon / len(admitted) if a in admitted else 0) for a in range(3))


def _validate_events(events):
    if (
        not isinstance(events, list)
        or not 2 <= len(events) <= MAX_EVENTS
        or not all(isinstance(row, LoggedEvent) for row in events)
    ):
        raise ValueError("OPE requires 2 to 5000 validated independent request events")
    if len({row.context.request_id for row in events}) != len(events):
        raise ValueError("Duplicate request IDs would invalidate independent-request inference")
    if len({len(row.context.features) for row in events}) != 1:
        raise ValueError("All contexts must use the same feature dimension")


def _solve(matrix, vector):
    """Partial-pivot Gaussian elimination; positive ridge makes Gram systems nonsingular."""
    rows = [list(row) + [value] for row, value in zip(matrix, vector)]
    n = len(vector)
    for column in range(n):
        pivot = max(range(column, n), key=lambda index: abs(rows[index][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        if abs(scale) < 1e-15:
            raise ValueError("Nuisance regression is numerically singular")
        rows[column] = [value / scale for value in rows[column]]
        for index in range(n):
            if index == column:
                continue
            factor = rows[index][column]
            rows[index] = [a - factor * b for a, b in zip(rows[index], rows[column])]
    return [row[-1] for row in rows]


def _fit(rows, targets, width, ridge):
    gram = [[ridge if i == j else 0.0 for j in range(width)] for i in range(width)]
    rhs = [0.0] * width
    for x, y in zip(rows, targets):
        for i in range(width):
            rhs[i] += x[i] * y
            for j in range(width):
                gram[i][j] += x[i] * x[j]
    return _solve(gram, rhs)


def cross_fitted_predictions(events: list[LoggedEvent], folds: int = 5, seed: int = 17, ridge: float = 0.1) -> dict:
    _validate_events(events)
    if (
        type(folds) is not int
        or not 2 <= folds <= min(10, len(events))
        or type(seed) is not int
        or not _number(ridge, 1e-8, 100)
    ):
        raise ValueError("Cross fitting needs 2 to 10 nonempty folds, integer seed and positive finite ridge")
    order = sorted(range(len(events)), key=lambda i: events[i].context.request_id)
    random.Random(seed).shuffle(order)
    assignment = [0] * len(events)
    for position, index in enumerate(order):
        assignment[index] = position % folds
    quality, cost = [[0.0] * 3 for _ in events], [[0.0] * 3 for _ in events]
    diagnostics = []
    width = len(events[0].context.features) + 1
    for fold in range(folds):
        training = [i for i in range(len(events)) if assignment[i] != fold]
        heldout = [i for i in range(len(events)) if assignment[i] == fold]
        counts = {}
        for action in (1, 2):
            chosen = [i for i in training if events[i].action == ACTIONS[action]]
            counts[ACTIONS[action]] = len(chosen)
            x = [[1.0, *events[i].context.features] for i in chosen]
            # Empty-action folds use an explicit zero nuisance prior. Known
            # randomized propensities still identify DR; diagnostics expose it.
            y_quality = [events[i].quality for i in chosen]
            y_cost_fraction = [events[i].cost / events[i].context.quoted_costs[action] for i in chosen]
            beta_quality = _fit(x, y_quality, width, ridge)
            beta_cost = _fit(x, y_cost_fraction, width, ridge)
            for i in heldout:
                vector = [1.0, *events[i].context.features]
                quality[i][action] = min(1.0, max(0.0, sum(a * b for a, b in zip(vector, beta_quality))))
                fraction = min(1.0, max(0.0, sum(a * b for a, b in zip(vector, beta_cost))))
                cost[i][action] = fraction * events[i].context.quoted_costs[action]
        diagnostics.append(
            {
                "fold": fold,
                "training_count": len(training),
                "heldout_count": len(heldout),
                "training_action_counts": counts,
                "heldout_ids_sha256": digest(sorted(events[i].context.request_id for i in heldout)),
            }
        )
    return {
        "quality": quality,
        "cost": cost,
        "assignment": assignment,
        "folds": diagnostics,
        "seed": seed,
        "ridge": ridge,
        "method": "per-action ridge on other folds only; cost-fraction regression",
    }


class NumericalRangeError(RuntimeError):
    """An estimator score or interval cannot be represented safely in float64."""


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise NumericalRangeError("Nonrepresentable importance-weighted value")
    return value


def _product(left: float, right: float) -> float:
    result = _finite(left * right)
    if result == 0 and left != 0 and right != 0:
        raise NumericalRangeError("Importance-weighted value underflows float64")
    return result


def _mean(values: list[float]) -> float:
    scale = max(abs(_finite(value)) for value in values)
    if scale == 0:
        return 0.0
    return _product(scale, math.fsum(value / scale for value in values) / len(values))


def _wald(values: list[float], estimate: float, alpha: float):
    if len(values) < 2:
        return None
    # Squaring a perfectly representable 1e200 score overflows. Compute
    # centered moments at unit scale, then restore only the standard error.
    scale = max(abs(_finite(value)) for value in values)
    normalized = [value / scale for value in values] if scale else [0.0] * len(values)
    center = math.fsum(normalized) / len(values)
    standard_error = _product(
        scale, math.sqrt(math.fsum((value - center) ** 2 for value in normalized) / ((len(values) - 1) * len(values)))
    )
    radius = _product(NormalDist().inv_cdf(1 - alpha / 2), standard_error)
    return {
        "lower": _finite(estimate - radius),
        "upper": _finite(estimate + radius),
        "standard_error": standard_error,
        "confidence_level": 1 - alpha,
        "method": "asymptotic influence-function Wald; not finite-sample coverage",
    }


def _numerical_refusal(report):
    report["numerical_status"] = "unsupported_float64_range"
    report["usable"] = False
    report["estimates"] = None
    report["warnings"].append(
        "Importance-weighted scores or confidence bounds are outside representable float64 range; estimates are refused without clipping."
    )
    return report


def _estimate_metrics(events, distributions, weights, scaled_weights, scaled_total, nuisance, alpha):
    estimates = {}
    n = len(events)
    for metric in ("quality", "cost"):
        observed = [getattr(row, metric) for row in events]
        ips_values = [_product(w, y) for w, y in zip(weights, observed)]
        ips = _mean(ips_values)
        snips = math.fsum(w * y for w, y in zip(scaled_weights, observed)) / scaled_total if scaled_total else None
        dr_values, normalized_ranges = [], []
        for i, (row, pi, weight) in enumerate(zip(events, distributions, weights)):
            action = ACTIONS.index(row.action)
            predictions = nuisance[metric][i]
            correction = _product(weight, observed[i] - predictions[action])
            dr_values.append(_finite(math.fsum(p * m for p, m in zip(pi, predictions)) + correction))
            # Normalize before multiplication and use hypot rather than sum
            # of squares, avoiding overflow in representable Hoeffding radii.
            ranges = [
                _product(
                    _finite(pi[a] / row.logging_probabilities[a]) / n,
                    1.0 if metric == "quality" else row.context.quoted_costs[a],
                )
                for a in (1, 2)
                if pi[a] > 0
            ]
            normalized_ranges.append(max(ranges, default=0.0))
        dr = _mean(dr_values)
        radius = _product(math.hypot(*normalized_ranges), math.sqrt(math.log(2 / alpha) / 2))
        snips_ci = None
        if snips is not None:
            # This ratio is invariant to the common weight scaling, unlike
            # w*(y-theta)/mean(w), whose intermediate product can overflow.
            scaled_mean_weight = scaled_total / n
            influence = [(w / scaled_mean_weight) * (y - snips) for w, y in zip(scaled_weights, observed)]
            snips_ci = _wald(influence, snips, alpha)
        estimates[metric] = {
            "ips": {
                "estimate": ips,
                "wald_ci": _wald(ips_values, ips, alpha),
                "conditional_hoeffding_ci": {
                    "lower": _finite(ips - radius),
                    "upper": _finite(ips + radius),
                    "confidence_level": 1 - alpha,
                    "method": "finite-sample conditional independent randomized actions; fixed target and contexts",
                },
            },
            "snips": {"estimate": snips, "wald_ci": snips_ci},
            "cross_fitted_dr": {"estimate": dr, "wald_ci": _wald(dr_values, dr, alpha)},
        }
    return estimates


def evaluate_logged(
    events: list[LoggedEvent],
    policy: StatelessPolicy,
    *,
    folds=5,
    seed=17,
    ridge=0.1,
    alpha=0.05,
    min_ess=30.0,
    min_cases=100,
) -> dict:
    _validate_events(events)
    if not isinstance(policy, StatelessPolicy):
        raise ValueError("Only an explicit stateless per-request policy is supported; not a sequential batch router")
    if (
        not _number(alpha, 1e-6, 0.2)
        or not _number(min_ess, 0, MAX_EVENTS)
        or type(min_cases) is not int
        or not 1 <= min_cases <= MAX_EVENTS
    ):
        raise ValueError("Invalid inference/diagnostic thresholds")
    distributions = [policy.probabilities(row.context) for row in events]
    violations = [
        {"request_id": row.context.request_id, "action": ACTIONS[a], "target_mass": pi[a]}
        for row, pi in zip(events, distributions)
        for a in range(3)
        if pi[a] > 0 and row.logging_probabilities[a] == 0
    ]
    report = {
        "schema_version": 1,
        "implementation": VERSION,
        "estimand": "stateless per-request target quality and cost",
        "dataset_sha256": digest([asdict(row) for row in events]),
        "policy_id": policy.policy_id,
        "target_distributions_sha256": digest(distributions),
        "case_count": len(events),
        "support_violations": violations,
        "identifiable_on_logged_contexts": not violations,
        "usable": False,
        "warnings": [],
        "estimates": None,
        "inference_config": {
            "folds": folds,
            "seed": seed,
            "ridge": ridge,
            "alpha": alpha,
            "minimum_ess": min_ess,
            "minimum_cases": min_cases,
        },
    }
    report["artifact_sha256"] = digest(
        {
            "implementation": VERSION,
            "dataset": report["dataset_sha256"],
            "target": report["target_distributions_sha256"],
            "policy": policy.policy_id,
            "inference_config": report["inference_config"],
        }
    )
    if violations:
        report["warnings"].append("Target has unsupported action probability; no IPS/SNIPS/DR estimate is returned.")
        return report
    try:
        weights = [
            _finite(pi[ACTIONS.index(row.action)] / row.logging_probabilities[ACTIONS.index(row.action)])
            for row, pi in zip(events, distributions)
        ]
    except NumericalRangeError:
        return _numerical_refusal(report)
    maximum_weight = max(weights)
    scaled_weights = [weight / maximum_weight for weight in weights] if maximum_weight else [0.0] * len(weights)
    scaled_total = math.fsum(scaled_weights)
    ess = scaled_total**2 / math.fsum(weight * weight for weight in scaled_weights) if scaled_total else 0.0
    total_weight = maximum_weight * scaled_total
    report["overlap"] = {
        "effective_sample_size": ess,
        "sum_weights": total_weight if math.isfinite(total_weight) else None,
        "log_sum_weights": math.log(maximum_weight) + math.log(scaled_total) if scaled_total else None,
        "weight_sum_overflowed": not math.isfinite(total_weight),
        "mean_weight": maximum_weight * (scaled_total / len(events)),
        "maximum_observed_weight": maximum_weight,
        "matched_events": sum(weight > 0 for weight in weights),
        "minimum_target_supported_logging_probability": min(
            q
            for row, pi in zip(events, distributions)
            for q, target in zip(row.logging_probabilities, pi)
            if target > 0
        ),
    }
    if ess < min_ess:
        report["warnings"].append("Low effective sample size; nominal intervals can be unreliable.")
    if len(events) < min_cases:
        report["warnings"].append("Fewer independent requests than the declared minimum.")
    nuisance = cross_fitted_predictions(events, folds, seed, ridge)
    if any(count == 0 for fold in nuisance["folds"] for count in fold["training_action_counts"].values()):
        report["warnings"].append(
            "Some fold/action nuisance models use the declared zero prior because no training action was observed."
        )
    report["cross_fitting"] = {key: value for key, value in nuisance.items() if key not in {"quality", "cost"}}
    try:
        report["estimates"] = _estimate_metrics(
            events, distributions, weights, scaled_weights, scaled_total, nuisance, alpha
        )
    except (NumericalRangeError, OverflowError):
        return _numerical_refusal(report)
    report["numerical_status"] = "finite_float64"
    report["usable"] = ess >= min_ess and len(events) >= min_cases
    report["assumptions"] = [
        "Independent requests; no interference; target policy fixed before inspecting these outcomes.",
        "Logged probabilities are the actual randomized action probabilities, not fitted scores.",
        "Every target-admitted action has logging support on the target context distribution.",
        "Per-request costs are bounded by their pre-dispatch quotes and cap.",
        "Wald intervals are asymptotic and marginal; DR nuisance convergence and overlap are required.",
        "Conditional IPS Hoeffding bounds concern the realized contexts/potential outcomes, not unseen-context coverage.",
        "No weight clipping is performed; support failures never become model-only estimates.",
    ]
    return report


def synthetic_log(samples=1500, seed=17, epsilon=0.3):
    if type(samples) is not int or not 2 <= samples <= MAX_EVENTS or type(seed) is not int:
        raise ValueError("Synthetic logs need 2 to 5000 requests and an integer seed")
    environment, logger = random.Random(seed), random.Random(seed + 1_000_003)
    events, potential = [], []
    for i in range(samples):
        difficulty, domain = environment.random(), float(environment.randrange(2))
        small_quote, large_quote = float(environment.randint(1, 3)), float(environment.randint(5, 9))
        cap = environment.choice((0.0, 2.0, 5.0, 9.0))
        context = Context(f"synthetic-{i:05}", (difficulty, domain), (0.0, small_quote, large_quote), cap)
        shared = environment.random()
        quality = (0.0, float(shared < 0.95 - 0.65 * difficulty), float(shared < 0.98 - 0.15 * difficulty))
        costs = (
            0.0,
            small_quote * (0.8 + 0.2 * environment.random()),
            large_quote * (0.8 + 0.2 * environment.random()),
        )
        probabilities = logging_distribution(context, epsilon)
        u, cumulative, selected = logger.random(), 0.0, 2
        for a, p in enumerate(probabilities):
            cumulative += p
            if u < cumulative:
                selected = a
                break
        events.append(LoggedEvent(context, ACTIONS[selected], probabilities, quality[selected], costs[selected]))
        potential.append({"request_id": context.request_id, "quality": quality, "cost": costs})
    return events, potential


def full_information_value(events, potential, policy):
    if len(events) != len(potential) or any(
        row.context.request_id != truth["request_id"] for row, truth in zip(events, potential)
    ):
        raise ValueError("Potential outcomes must match the exact logged context ordering")
    return {
        metric: sum(
            sum(p * y for p, y in zip(policy.probabilities(row.context), truth[metric]))
            for row, truth in zip(events, potential)
        )
        / len(events)
        for metric in ("quality", "cost")
    }


def bandit_demo(samples=1500, seed=17, epsilon=0.3, folds=5):
    events, potential = synthetic_log(samples, seed, epsilon)
    outcomes = []
    for name in ("small", "large", "adaptive", "soft_adaptive", "admit"):
        policy = target_policy(name)
        evaluation = evaluate_logged(events, policy, folds=folds, seed=seed)
        truth = full_information_value(events, potential, policy)
        evaluation["synthetic_full_information_target_value"] = truth
        evaluation["absolute_error"] = {
            metric: {
                estimator: abs(value["estimate"] - truth[metric]) if value["estimate"] is not None else None
                for estimator, value in evaluation["estimates"][metric].items()
            }
            for metric in ("quality", "cost")
        }
        outcomes.append(evaluation)
    oracle_quality = (
        sum(
            max(truth["quality"][a] for a in range(3) if row.context.admitted(a))
            for row, truth in zip(events, potential)
        )
        / samples
    )
    return {
        "schema_version": 1,
        "experiment": "synthetic_stateless_logged_bandit_v1",
        "simulation": True,
        "model_evaluation": False,
        "samples": samples,
        "seed": seed,
        "epsilon": epsilon,
        "folds": folds,
        "logging": "epsilon exploration over cap-admitted actions plus fixed adaptive exploitation",
        "potential_outcomes_sha256": digest(potential),
        "outcomes": outcomes,
        "full_information_per_request_quality_oracle": oracle_quality,
        "oracle_scope": "Label-informed best admitted action independently per request; no batch budget.",
        "excluded_scope": "The sequential batch-budget router is not evaluated with these one-step estimators.",
    }


def load_logged(path):
    events, total = [], 0
    with open(path, encoding="utf-8") as source:
        while True:
            line = source.readline(100001)
            if not line:
                break
            total += len(line)
            if len(line) > 100000 or total > 20_000_000:
                raise ValueError("Logged-event import character limit exceeded")
            if not line.strip():
                continue
            if len(events) >= MAX_EVENTS:
                raise ValueError("Logged-event record limit exceeded")
            row = strict_json(line)
            if not isinstance(row, dict) or not isinstance(row.get("context"), dict):
                raise ValueError("Logged events require a context object")
            try:
                context = dict(row["context"])
                context["features"], context["quoted_costs"] = (
                    tuple(context["features"]),
                    tuple(context["quoted_costs"]),
                )
                row["context"] = Context(**context)
                row["logging_probabilities"] = tuple(row["logging_probabilities"])
                events.append(LoggedEvent(**row))
            except (TypeError, KeyError) as exc:
                raise ValueError("Invalid logged-event schema") from exc
    _validate_events(events)
    return events
