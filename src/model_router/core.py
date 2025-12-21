"""Learned routing, held-out calibration diagnostics, and exact replay budget bounds."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
import re
from typing import Any
from .data import MAX_RECORDS, Record, correct, dataset_hash, features, strict_json, validate_splits


def _budget(value: int) -> None:
    if type(value) is not int or not 0 <= value <= 100_000:
        raise ValueError("Budget must be integer units in [0, 100000]")


def sigmoid(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Logit must be finite")
    value = max(-40.0, min(40.0, value))
    return 1 / (1 + math.exp(-value))


def fit_logistic(
    rows: list[list[float]],
    targets: list[int],
    epochs: int = 800,
    learning_rate: float = 0.3,
    regularization: float = 0.005,
) -> list[float]:
    if not rows or len(rows) != len(targets) or len(rows) > MAX_RECORDS:
        raise ValueError("Training requires aligned bounded nonempty examples")
    if type(epochs) is not int or not 1 <= epochs <= 5000:
        raise ValueError("Epochs must be an integer in [1, 5000]")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (learning_rate, regularization)):
        raise ValueError("Training hyperparameters must be finite")
    if not 0 < learning_rate <= 1 or not 0 <= regularization <= 1:
        raise ValueError("Invalid learning rate or regularization")
    width = len(rows[0])
    if not 1 <= width <= 64 or any(len(row) != width for row in rows):
        raise ValueError("Invalid feature dimensions")
    if any(type(value) not in (float, int) or not math.isfinite(value) for row in rows for value in row):
        raise ValueError("Feature values must be finite numbers")
    if any(type(target) is not int or target not in (0, 1) for target in targets):
        raise ValueError("Labels must be integer zero or one")
    weights = [0.0] * width
    for _ in range(epochs):
        gradient = [0.0] * width
        for row, target in zip(rows, targets):
            residual = sigmoid(sum(a * b for a, b in zip(row, weights))) - target
            for index, value in enumerate(row):
                gradient[index] += residual * value / len(rows)
        for index in range(width):
            penalty = regularization * weights[index] if index else 0
            weights[index] -= learning_rate * (gradient[index] + penalty)
    return weights


@dataclass(frozen=True)
class Router:
    weights: tuple[float, ...]
    calibration: tuple[float, float]
    threshold: float
    training_hash: str
    calibration_hash: str
    training_template_ids: tuple[str, ...]
    calibration_template_ids: tuple[str, ...]
    calibration_budget_units: int = 180

    def __post_init__(self):
        numbers = (*self.weights, *self.calibration, self.threshold)
        if (
            len(self.weights) != 6
            or len(self.calibration) != 2
            or not all(type(x) in (int, float) and math.isfinite(x) for x in numbers)
        ):
            raise ValueError("Invalid router coefficients")
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be a probability")
        _budget(self.calibration_budget_units)
        for value in (self.training_hash, self.calibration_hash):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Artifact provenance hashes must be SHA-256 strings")
        for groups in (self.training_template_ids, self.calibration_template_ids):
            if (
                not isinstance(groups, tuple)
                or not 1 <= len(groups) <= MAX_RECORDS
                or any(type(group) is not str or not 1 <= len(group) <= 128 for group in groups)
                or len(set(groups)) != len(groups)
            ):
                raise ValueError("Artifact groups must be unique bounded identifiers")
        if set(self.training_template_ids) & set(self.calibration_template_ids):
            raise ValueError("Training/calibration leakage")

    def raw_probability(self, expression: str) -> float:
        return sigmoid(sum(a * b for a, b in zip(self.weights, features(expression))))

    def probability(self, expression: str) -> float:
        raw = sum(a * b for a, b in zip(self.weights, features(expression)))
        return sigmoid(self.calibration[0] + self.calibration[1] * raw)

    def choose(
        self, expression: str, quoted_costs: dict[str, int], remaining: int, threshold: float | None = None
    ) -> str | None:
        _budget(remaining)
        if set(quoted_costs) != {"small", "large"} or any(
            type(cost) is not int or not 1 <= cost <= 100_000 for cost in quoted_costs.values()
        ):
            raise ValueError("Both bounded positive pre-dispatch quotes are required")
        cutoff = self.threshold if threshold is None else threshold
        if type(cutoff) not in (float, int) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
            raise ValueError("Invalid threshold")
        preferred = "small" if self.probability(expression) >= cutoff else "large"
        if quoted_costs[preferred] <= remaining:
            return preferred
        other = "large" if preferred == "small" else "small"
        return other if quoted_costs[other] <= remaining else None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 2, "model_type": "logistic_success_predictor_with_platt_calibration", **asdict(self)}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Router:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != 2
            or row.get("model_type") != "logistic_success_predictor_with_platt_calibration"
        ):
            raise ValueError("Unsupported router artifact; retrain schema-1 artifacts with the current command")
        keys = {
            "weights",
            "calibration",
            "threshold",
            "training_hash",
            "calibration_hash",
            "training_template_ids",
            "calibration_template_ids",
            "calibration_budget_units",
        }
        if set(row) != keys | {"schema_version", "model_type"}:
            raise ValueError("Unexpected router fields")
        values = {key: row[key] for key in keys}
        for key in ("weights", "calibration", "training_template_ids", "calibration_template_ids"):
            if not isinstance(values[key], (list, tuple)) or len(values[key]) > MAX_RECORDS:
                raise ValueError("Artifact arrays must be bounded")
            values[key] = tuple(values[key])
        return cls(**values)


def load_router(path: str) -> Router:
    with Path(path).open(encoding="utf-8") as source:
        text = source.read(1_048_577)
    if len(text) > 1_048_576:
        raise ValueError("Router artifact exceeds the 1 MiB character limit")
    return Router.from_dict(strict_json(text))


def _records(records: list[Record]):
    if (
        not isinstance(records, list)
        or len(records) > MAX_RECORDS
        or not all(isinstance(row, Record) for row in records)
    ):
        raise ValueError("Evaluation requires a bounded list of validated records")


def evaluate_policy(
    records: list[Record], router: Router, policy: str, budget: int, threshold: float | None = None
) -> dict[str, Any]:
    _budget(budget)
    _records(records)
    if policy not in {"small", "large", "heuristic", "learned"}:
        raise ValueError("Unknown routing policy")
    remaining, solved, answered = budget, 0, 0
    decisions = []
    measured_latency, latency_complete = 0.0, True
    provenance_counts = {"SIMULATED": 0, "RECORDED": 0}
    for record in records:
        quotes = {response.model_id: response.cost_units for response in record.responses}
        if policy == "learned":
            selected = router.choose(record.expression, quotes, remaining, threshold)
        else:
            selected = (
                ("small" if features(record.expression)[2] <= 0.1 else "large") if policy == "heuristic" else policy
            )
            if quotes[selected] > remaining:
                selected = None
        if selected is None:
            decisions.append(
                {
                    "case_id": record.case_id,
                    "model_id": None,
                    "status": "abstained_budget",
                    "cost_units": 0,
                    "correct": False,
                }
            )
            continue
        response = record.response(selected)
        if response.cost_units > remaining:
            raise AssertionError("Pre-dispatch budget invariant violated")
        remaining -= response.cost_units
        is_correct = correct(record, selected)
        solved += is_correct
        answered += 1
        provenance_counts[response.provenance] += 1
        if response.latency_ms is None:
            latency_complete = False
        else:
            measured_latency += response.latency_ms
        decisions.append(
            {
                "case_id": record.case_id,
                "model_id": selected,
                "status": "answered",
                "cost_units": response.cost_units,
                "correct": is_correct,
                "response_provenance": response.provenance,
                "small_success_probability": router.probability(record.expression),
            }
        )
    return {
        "policy": policy,
        "case_count": len(records),
        "correct": solved,
        "answered": answered,
        "accuracy": solved / len(records) if records else None,
        "answered_accuracy": solved / answered if answered else None,
        "coverage": answered / len(records) if records else None,
        "budget_units": budget,
        "spent_units": budget - remaining,
        "cost_per_correct": (budget - remaining) / solved if solved else None,
        "selected_response_provenance_counts": provenance_counts,
        "recorded_latency_sum_ms": measured_latency if latency_complete and answered else None,
        "decisions": decisions,
    }


def oracle_plan(records: list[Record], budget: int) -> dict[str, Any]:
    """Exact unit-value bound in O(n log n), not pseudo-polynomial knapsack.

    Each case is worth one correct answer. A wrong response is dominated by
    free abstention; of correct responses only the cheapest can be optimal.
    Buying the cheapest remaining correct cases therefore maximizes count.
    """
    _budget(budget)
    _records(records)
    options = []
    for row in records:
        correct_options = [
            (response.cost_units, response.model_id) for response in row.responses if correct(row, response.model_id)
        ]
        if correct_options:
            cost, model = min(correct_options)
            options.append((cost, row.case_id, model))
    selected, spent = [], 0
    for cost, case_id, model in sorted(options):
        if spent + cost > budget:
            break
        spent += cost
        selected.append({"case_id": case_id, "model_id": model, "cost_units": cost})
    return {
        "correct": len(selected),
        "spent_units": spent,
        "budget_units": budget,
        "case_count": len(records),
        "selected": selected,
        "algorithm": "exact_sorted_cheapest_correct_case_unit_value",
    }


def oracle_score(records: list[Record], budget: int) -> int:
    return oracle_plan(records, budget)["correct"]


def train(records: list[Record], calibration_budget: int) -> Router:
    _budget(calibration_budget)
    validate_splits(records)
    training = [record for record in records if record.split == "train"]
    calibration = [record for record in records if record.split == "calibration"]
    weights = fit_logistic(
        [features(row.expression) for row in training], [int(correct(row, "small")) for row in training]
    )
    raw = [sum(a * b for a, b in zip(weights, features(row.expression))) for row in calibration]
    platt = fit_logistic(
        [[1.0, value] for value in raw],
        [int(correct(row, "small")) for row in calibration],
        epochs=500,
        learning_rate=0.1,
    )
    candidate = Router(
        tuple(weights),
        tuple(platt),
        0.5,
        dataset_hash(training),
        dataset_hash(calibration),
        tuple(sorted({row.template_id for row in training})),
        tuple(sorted({row.template_id for row in calibration})),
        calibration_budget,
    )
    options = []
    for step in range(21):
        threshold = step / 20
        report = evaluate_policy(calibration, candidate, "learned", calibration_budget, threshold)
        options.append((report["correct"], -report["spent_units"], -threshold, threshold))
    return Router(
        candidate.weights,
        candidate.calibration,
        max(options)[-1],
        candidate.training_hash,
        candidate.calibration_hash,
        candidate.training_template_ids,
        candidate.calibration_template_ids,
        calibration_budget,
    )


def calibration_metrics(probabilities: list[float], targets: list[int]) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(targets):
        raise ValueError("Calibration metrics require aligned nonempty samples")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities
    ):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if any(type(target) is not int or target not in (0, 1) for target in targets):
        raise ValueError("Targets must be integer zero or one")
    bins, ece = [], 0.0
    for index in range(5):
        selected = [(p, target) for p, target in zip(probabilities, targets) if min(int(p * 5), 4) == index]
        if selected:
            confidence = sum(p for p, _ in selected) / len(selected)
            observed = sum(target for _, target in selected) / len(selected)
            ece += len(selected) / len(targets) * abs(confidence - observed)
            bins.append(
                {"bin": index, "count": len(selected), "mean_probability": confidence, "observed_success": observed}
            )
    return {
        "case_count": len(targets),
        "brier_score": sum((p - y) ** 2 for p, y in zip(probabilities, targets)) / len(targets),
        "log_loss": -sum(
            y * math.log(max(1e-15, p)) + (1 - y) * math.log(max(1e-15, 1 - p)) for p, y in zip(probabilities, targets)
        )
        / len(targets),
        "ece_5_bins": ece,
        "bins": bins,
    }


def _checked_router(records: list[Record], budget: int, router: Router | None) -> tuple[Router, list[Record], int]:
    _budget(budget)
    validate_splits(records)
    test = [row for row in records if row.split == "test"]
    training = [row for row in records if row.split == "train"]
    calibration = [row for row in records if row.split == "calibration"]
    calibration_budget = round(budget * len(calibration) / len(test))
    _budget(calibration_budget)
    router = router or train(records, calibration_budget)
    forbidden = set(router.training_template_ids) | set(router.calibration_template_ids)
    if forbidden & {row.template_id for row in test}:
        raise ValueError("Router artifact has seen held-out templates")
    if router.training_hash != dataset_hash(training):
        raise ValueError("Router training provenance does not match the supplied training split")
    if router.calibration_hash != dataset_hash(calibration):
        raise ValueError("Router calibration provenance does not match the supplied calibration split")
    if set(router.training_template_ids) != {row.template_id for row in training} or set(
        router.calibration_template_ids
    ) != {row.template_id for row in calibration}:
        raise ValueError("Router declared group provenance does not match the supplied splits")
    return router, test, calibration_budget


def evaluate(records: list[Record], budget: int = 180, router: Router | None = None) -> dict[str, Any]:
    router, test, calibration_budget = _checked_router(records, budget, router)
    oracle = oracle_plan(test, budget)
    policies = [evaluate_policy(test, router, name, budget) for name in ("small", "large", "heuristic", "learned")]
    for report in policies:
        report["budget_oracle_correct"] = oracle["correct"]
        report["correct_count_regret"] = oracle["correct"] - report["correct"]
    targets = [int(correct(row, "small")) for row in test]
    calibrated = calibration_metrics([router.probability(row.expression) for row in test], targets)
    raw = calibration_metrics([router.raw_probability(row.expression) for row in test], targets)
    calibrated["uncalibrated"] = raw
    calibrated["brier_delta_calibrated_minus_raw"] = calibrated["brier_score"] - raw["brier_score"]
    test_provenance = sorted({response.provenance for row in test for response in row.responses})
    mode = "mixed_recorded_and_simulated" if len(test_provenance) > 1 else test_provenance[0].lower()
    return {
        "schema_version": 2,
        "dataset_sha256": dataset_hash(records),
        "response_provenance": sorted({response.provenance for row in records for response in row.responses}),
        "test_response_provenance": test_provenance,
        "evaluation_mode": mode,
        "source_models": sorted(
            {response.source_model for row in records for response in row.responses if response.source_model}
        ),
        "live_provider_calls_executed": False,
        "cost_unit_description": "Configured integer replay units, not currency or provider billing.",
        "split_counts": {
            split: sum(row.split == split for row in records) for split in ("train", "calibration", "test")
        },
        "router": router.to_dict(),
        "policies": policies,
        "budget_oracle": oracle,
        "calibration": calibrated,
        "budget_transfer": {
            "artifact_calibration_budget_units": router.calibration_budget_units,
            "matching_per_case_calibration_budget_units": calibration_budget,
            "different_budget_regime": router.calibration_budget_units != calibration_budget,
        },
        "limitations": [
            "Default responses/costs are SIMULATED; scores are not provider quality or savings.",
            "Oracle knows labels and assumes each correct case has unit value.",
            "Probability fitting and threshold selection reuse calibration labels; test remains held out.",
            "Artifact metadata is declarative, not proof of an external collector's training process.",
        ],
        "model_evaluation": test_provenance == ["RECORDED"],
    }


def evaluate_sweep(
    records: list[Record],
    budgets: list[int],
    *,
    reference_budget: int = 180,
    orderings: int = 5,
    seed: int = 17,
    router: Router | None = None,
) -> dict[str, Any]:
    """Freeze one calibrated router; inspect budget transfer and arrival-order sensitivity."""
    if not isinstance(budgets, list) or not 1 <= len(budgets) <= 16 or len(set(budgets)) != len(budgets):
        raise ValueError("Supply 1 to 16 distinct budgets")
    for budget in budgets:
        _budget(budget)
    if type(orderings) is not int or not 1 <= orderings <= 20 or type(seed) is not int:
        raise ValueError("Orderings must be 1 to 20 and seed an integer")
    router, test, _ = _checked_router(records, reference_budget, router)
    outcomes = []
    for budget in sorted(budgets):
        oracle = oracle_score(test, budget)
        policies = []
        for policy in ("small", "large", "heuristic", "learned"):
            runs = []
            for ordering in range(orderings):
                ordered = list(test)
                if ordering:
                    random.Random(seed + ordering).shuffle(ordered)
                report = evaluate_policy(ordered, router, policy, budget)
                runs.append(
                    {
                        "ordering": ordering,
                        "correct": report["correct"],
                        "answered": report["answered"],
                        "spent_units": report["spent_units"],
                        "regret": oracle - report["correct"],
                    }
                )
            policies.append(
                {
                    "policy": policy,
                    "runs": runs,
                    "correct_min": min(row["correct"] for row in runs),
                    "correct_max": max(row["correct"] for row in runs),
                    "correct_mean": sum(row["correct"] for row in runs) / orderings,
                }
            )
        outcomes.append(
            {"budget_units": budget, "case_count": len(test), "oracle_correct": oracle, "policies": policies}
        )
    return {
        "schema_version": 1,
        "experiment": "frozen_router_budget_and_arrival_order_sweep",
        "dataset_sha256": dataset_hash(records),
        "router": router.to_dict(),
        "reference_test_budget_units": reference_budget,
        "orderings": orderings,
        "shuffle_seed": seed,
        "test_used_for_parameter_selection": False,
        "outcomes": outcomes,
        "limitations": [
            "A sensitivity experiment, not a test-selected optimal router or budget.",
            "Default outcomes use simulated candidate responses and configured cost units.",
        ],
    }
