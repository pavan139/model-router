from dataclasses import asdict, replace
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from model_router.bandit import (
    ACTIONS,
    Context,
    LoggedEvent,
    StatelessPolicy,
    bandit_demo,
    cross_fitted_predictions,
    evaluate_logged,
    full_information_value,
    load_logged,
    logging_distribution,
    synthetic_log,
    target_policy,
)


class BanditTests(unittest.TestCase):
    def test_logging_is_randomized_normalized_and_respects_each_request_cap(self):
        for cap in (0, 1, 3, 10):
            context = Context(str(cap), (0.8,), (0.0, 1.0, 5.0), cap)
            probabilities = logging_distribution(context, 0.3)
            self.assertAlmostEqual(sum(probabilities), 1)
            for action, p in enumerate(probabilities):
                self.assertEqual(p > 0, context.admitted(action))
            for name in ("small", "large", "adaptive", "soft_adaptive", "admit"):
                target = target_policy(name).probabilities(context)
                self.assertAlmostEqual(sum(target), 1)
                self.assertTrue(all(not p or context.admitted(a) for a, p in enumerate(target)))

    def test_ips_and_cross_fitted_dr_are_unbiased_under_exhaustive_action_assignments(self):
        contexts = [Context("one", (0.2,), (0.0, 1.0, 3.0), 3), Context("two", (0.8,), (0.0, 1.0, 3.0), 3)]
        potentials = [
            dict(request_id="one", quality=(0, 0.5, 1), cost=(0, 1, 2)),
            dict(request_id="two", quality=(0, 1, 0.5), cost=(0, 0.5, 3)),
        ]
        behavior = (0.2, 0.3, 0.5)
        policy = StatelessPolicy("fixed-soft-target-v1", lambda context: (0.1, 0.3, 0.6))
        weighted = {metric: {"ips": 0.0, "cross_fitted_dr": 0.0} for metric in ("quality", "cost")}
        for choices in itertools.product(range(3), repeat=2):
            probability = math.prod(behavior[action] for action in choices)
            events = [
                LoggedEvent(
                    contexts[i],
                    ACTIONS[action],
                    behavior,
                    potentials[i]["quality"][action],
                    potentials[i]["cost"][action],
                )
                for i, action in enumerate(choices)
            ]
            result = evaluate_logged(events, policy, folds=2, min_cases=2, min_ess=0)
            for metric in weighted:
                for estimator in weighted[metric]:
                    weighted[metric][estimator] += probability * result["estimates"][metric][estimator]["estimate"]
        # Independent full-information enumeration; no fitted outcome model is used here.
        truth = {
            metric: sum(sum(p * v for p, v in zip((0.1, 0.3, 0.6), potential[metric])) for potential in potentials) / 2
            for metric in ("quality", "cost")
        }
        for metric in weighted:
            for estimator, estimate in weighted[metric].items():
                with self.subTest(metric=metric, estimator=estimator):
                    self.assertAlmostEqual(estimate, truth[metric], places=11)

    def test_ips_snips_arithmetic_and_unclipped_estimates(self):
        contexts = [Context(str(i), (0.2,), (0.0, 2.0, 4.0), 4) for i in range(2)]
        events = [
            LoggedEvent(contexts[0], "small", (0.0, 0.25, 0.75), 1.0, 2.0),
            LoggedEvent(contexts[1], "large", (0.0, 0.25, 0.75), 0.0, 3.0),
        ]
        report = evaluate_logged(events, target_policy("small"), folds=2, min_cases=2, min_ess=0)
        self.assertEqual(report["estimates"]["quality"]["ips"]["estimate"], 2)
        self.assertEqual(report["estimates"]["quality"]["snips"]["estimate"], 1)
        self.assertEqual(report["estimates"]["cost"]["ips"]["estimate"], 4)
        self.assertEqual(report["estimates"]["cost"]["snips"]["estimate"], 2)
        self.assertEqual(report["overlap"]["effective_sample_size"], 1)
        self.assertEqual(report["overlap"]["mean_weight"], 2)

    def test_cross_fitting_excludes_the_heldout_outcome(self):
        events, _ = synthetic_log(50, seed=13)
        before = cross_fitted_predictions(events, folds=5, seed=7)
        changed = list(events)
        index = next(i for i, row in enumerate(changed) if row.action != "abstain")
        changed[index] = replace(changed[index], quality=1 - changed[index].quality)
        after = cross_fitted_predictions(changed, folds=5, seed=7)
        heldout_fold = before["assignment"][index]
        for i, fold in enumerate(before["assignment"]):
            if fold == heldout_fold:
                self.assertEqual(before["quality"][i], after["quality"][i])
                self.assertEqual(before["cost"][i], after["cost"][i])
        self.assertEqual(sum(fold["heldout_count"] for fold in before["folds"]), 50)

    def test_no_support_returns_no_model_only_fallback(self):
        events = [
            LoggedEvent(Context(str(i), (0.8,), (0.0, 1.0, 3.0), 3), "small", (0.0, 1.0, 0.0), 1, 1) for i in range(4)
        ]
        result = evaluate_logged(events, target_policy("large"), folds=2)
        self.assertFalse(result["identifiable_on_logged_contexts"])
        self.assertFalse(result["usable"])
        self.assertIsNone(result["estimates"])
        self.assertEqual(len(result["support_violations"]), 4)

    def test_no_realized_target_matches_yields_zero_ess_and_undefined_snips(self):
        events = [
            LoggedEvent(Context(str(i), (0.2,), (0.0, 1.0, 3.0), 3), "large", (0.0, 0.1, 0.9), 1, 3) for i in range(6)
        ]
        result = evaluate_logged(events, target_policy("small"), folds=2)
        self.assertTrue(result["identifiable_on_logged_contexts"])
        self.assertFalse(result["usable"])
        self.assertEqual(result["overlap"]["effective_sample_size"], 0)
        self.assertIsNone(result["estimates"]["quality"]["snips"]["estimate"])
        self.assertIsNone(result["estimates"]["quality"]["snips"]["wald_ci"])
        self.assertEqual(result["estimates"]["quality"]["ips"]["estimate"], 0)
        self.assertGreater(result["estimates"]["quality"]["ips"]["conditional_hoeffding_ci"]["upper"], 1)

    def test_abstention_cap_zero_has_known_zero_values_and_zero_radius(self):
        events = [
            LoggedEvent(Context(str(i), (0.5,), (0.0, 1.0, 3.0), 0), "abstain", (1.0, 0.0, 0.0), 0, 0) for i in range(6)
        ]
        result = evaluate_logged(events, target_policy("adaptive"), folds=2, min_cases=2, min_ess=0)
        self.assertTrue(result["usable"])
        for metric in ("quality", "cost"):
            for estimator in ("ips", "snips", "cross_fitted_dr"):
                self.assertEqual(result["estimates"][metric][estimator]["estimate"], 0)
            ci = result["estimates"][metric]["ips"]["conditional_hoeffding_ci"]
            self.assertEqual((ci["lower"], ci["upper"]), (0, 0))

    def test_malicious_target_cannot_overrule_per_request_cap(self):
        events = [
            LoggedEvent(Context(str(i), (0.5,), (0.0, 1.0, 3.0), 1), "small", (0.5, 0.5, 0.0), 1, 1) for i in range(6)
        ]
        policy = StatelessPolicy("invalid-cap-bypass", lambda context: (0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "per-request cap"):
            evaluate_logged(events, policy, folds=2)
        with self.assertRaisesRegex(ValueError, "sequential batch"):
            evaluate_logged(events, object(), folds=2)

    def test_propensities_observed_costs_and_independence_contract_are_checked(self):
        context = Context("sample", (0.5,), (0.0, 1.0, 3.0), 3)
        for probabilities in ((0, 0, 0), (0.2, 0.2, 0.2), (0, float("nan"), 1), (1.0, 0.0, 0.0)):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                LoggedEvent(context, "small", probabilities, 1, 1)
        with self.assertRaises(ValueError):
            LoggedEvent(context, "small", (0.2, 0.4, 0.4), 1, 1.01)
        with self.assertRaises(ValueError):
            LoggedEvent(context, "abstain", (0.2, 0.4, 0.4), 1, 0)
        row = LoggedEvent(context, "small", (0.2, 0.4, 0.4), 1, 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_logged([row, row], target_policy("small"), folds=2)

    def test_randomized_logs_and_fold_assignments_are_reproducible(self):
        first, truth = synthetic_log(100, seed=92)
        self.assertEqual((first, truth), synthetic_log(100, seed=92))
        predictions = cross_fitted_predictions(first, folds=5, seed=4)
        reversed_predictions = cross_fitted_predictions(list(reversed(first)), folds=5, seed=4)
        assignment = {row.context.request_id: fold for row, fold in zip(first, predictions["assignment"])}
        self.assertEqual(
            assignment,
            {row.context.request_id: fold for row, fold in zip(reversed(first), reversed_predictions["assignment"])},
        )
        for row in first:
            self.assertEqual(row.logging_probabilities, logging_distribution(row.context))

    def test_full_information_values_are_separate_from_logged_estimation(self):
        events, truth = synthetic_log(100, seed=72)
        policy = target_policy("soft_adaptive")
        expected = full_information_value(events, truth, policy)
        hand_quality = (
            sum(
                sum(p * y for p, y in zip(policy.probabilities(row.context), outcome["quality"]))
                for row, outcome in zip(events, truth)
            )
            / 100
        )
        self.assertAlmostEqual(expected["quality"], hand_quality)
        with self.assertRaises(ValueError):
            full_information_value(events, list(reversed(truth)), policy)
        result = evaluate_logged(events, policy, folds=2)
        self.assertNotIn("synthetic_full_information_target_value", result)

    def test_import_and_cli_replay_work_without_counterfactual_labels(self):
        events, _ = synthetic_log(100)
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "logs.jsonl", Path(directory) / "evaluation.json"
            path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in events))
            self.assertEqual(load_logged(path), events)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "model_router",
                    "bandit-evaluate",
                    "--logs",
                    str(path),
                    "--policy",
                    "adaptive",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(json.loads(output.read_text())["usable"])
            path.write_text("x" * 100001)
            with self.assertRaises(ValueError):
                load_logged(path)

    def test_small_executed_experiment_reports_truth_and_does_not_claim_provider_quality(self):
        result = bandit_demo(100, seed=12, folds=2)
        self.assertTrue(result["simulation"])
        self.assertFalse(result["model_evaluation"])
        self.assertEqual(len(result["outcomes"]), 5)
        self.assertIn("sequential", result["excluded_scope"])
        for report in result["outcomes"]:
            self.assertEqual(report["support_violations"], [])
            for metric in ("quality", "cost"):
                for estimator in ("ips", "snips", "cross_fitted_dr"):
                    self.assertIn("estimate", report["estimates"][metric][estimator])

    def test_extreme_finite_weights_use_stable_ess_and_moments(self):
        rows = [
            LoggedEvent(
                Context(str(i), (0.2,), (0.0, 1.0, 3.0), 3.0),
                "small" if i == 0 else "large",
                (0.0, 1e-200, 1.0),
                1.0,
                1.0 if i == 0 else 3.0,
            )
            for i in range(2)
        ]
        result = evaluate_logged(rows, target_policy("small"), folds=2, min_cases=2, min_ess=0)
        self.assertEqual(result["numerical_status"], "finite_float64")
        self.assertEqual(result["overlap"]["effective_sample_size"], 1)
        self.assertEqual(result["estimates"]["quality"]["ips"]["estimate"], 5e199)
        self.assertEqual(result["estimates"]["quality"]["snips"]["estimate"], 1)
        for metric in ("quality", "cost"):
            for estimator in ("ips", "snips", "cross_fitted_dr"):
                self.assertTrue(math.isfinite(result["estimates"][metric][estimator]["estimate"]))
                ci = result["estimates"][metric][estimator]["wald_ci"]
                self.assertTrue(all(math.isfinite(ci[key]) for key in ("lower", "upper", "standard_error")))
        json.dumps(result, allow_nan=False)

    def test_overflowing_raw_weight_sum_does_not_change_ess_or_snips(self):
        rows = [
            LoggedEvent(Context(str(i), (0.2,), (0.0, 1.0, 3.0), 3.0), "small", (0.0, 1e-308, 1.0), 1.0, 1.0)
            for i in range(4)
        ]
        result = evaluate_logged(rows, target_policy("small"), folds=2, min_cases=2, min_ess=0)
        self.assertEqual(result["numerical_status"], "finite_float64")
        self.assertEqual(result["overlap"]["effective_sample_size"], 4)
        self.assertTrue(result["overlap"]["weight_sum_overflowed"])
        self.assertIsNone(result["overlap"]["sum_weights"])
        self.assertTrue(math.isfinite(result["overlap"]["log_sum_weights"]))
        self.assertEqual(result["estimates"]["quality"]["snips"]["estimate"], 1)
        self.assertEqual(result["estimates"]["quality"]["ips"]["estimate"], 1e308)
        json.dumps(result, allow_nan=False)

    def test_nonrepresentable_ratios_and_weighted_costs_are_explicitly_refused(self):
        for propensity, quote in ((1e-320, 1.0), (1e-308, 100000.0)):
            rows = [
                LoggedEvent(
                    Context(str(i), (0.2,), (0.0, quote, quote), quote),
                    "small" if i == 0 else "large",
                    (0.0, propensity, 1.0),
                    1.0,
                    quote,
                )
                for i in range(2)
            ]
            result = evaluate_logged(rows, target_policy("small"), folds=2, min_cases=2, min_ess=0)
            with self.subTest(propensity=propensity, quote=quote):
                self.assertEqual(result["numerical_status"], "unsupported_float64_range")
                self.assertFalse(result["usable"])
                self.assertIsNone(result["estimates"])
                self.assertTrue(result["identifiable_on_logged_contexts"])
                self.assertTrue(any("without clipping" in warning for warning in result["warnings"]))
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
