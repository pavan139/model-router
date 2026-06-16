from dataclasses import replace
import json
import itertools
import io
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_router.core import (
    calibration_metrics,
    evaluate_sweep,
    load_router,
    oracle_plan,
    Router,
    evaluate,
    evaluate_policy,
    fit_logistic,
    oracle_score,
    sigmoid,
    train,
)
from model_router.data import (
    correct,
    expression_structure,
    Record,
    Response,
    answer,
    features,
    generate,
    load,
    serialized,
    validate_splits,
)


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = generate()
        cls.router = train(cls.records, 180)
        cls.test = [row for row in cls.records if row.split == "test"]

    def test_generator_is_reproducible_with_all_splits(self):
        self.assertEqual(serialized(generate()), serialized(self.records))
        self.assertEqual(len(self.records), 180)
        self.assertEqual(len(self.test), 36)
        self.assertTrue(
            all(
                response.provenance == "SIMULATED" and response.latency_ms is None
                for row in self.records
                for response in row.responses
            )
        )

    def test_arithmetic_oracle_rejects_code_and_limits(self):
        self.assertEqual(answer("(3 + 4) * 5 - 2"), "33")
        for expression in (
            "__import__('os').system('echo unsafe')",
            "2 ** 100",
            "1 / 0",
            "True + 2",
            "10001 + 1",
            "x" * 257,
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    answer(expression)

    def test_template_leakage_rejected(self):
        rows = list(self.records)
        rows[-1] = replace(rows[-1], template_id=rows[0].template_id)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(rows)

    def test_expression_structure_leakage_rejected(self):
        rows = list(self.records)
        training = next(row for row in rows if row.split == "train")
        index = next(index for index, row in enumerate(rows) if row.split == "test")
        rows[index] = replace(rows[index], expression=training.expression)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(rows)

    def test_training_uses_no_test_response_labels(self):
        modified = [
            replace(row, responses=tuple(replace(response, text="incorrect") for response in row.responses))
            if row.split == "test"
            else row
            for row in self.records
        ]
        self.assertEqual(train(modified, 180), self.router)

    def test_probability_and_routing_do_not_read_answers(self):
        row = self.test[0]
        quotes = {response.model_id: response.cost_units for response in row.responses}
        self.assertEqual(self.router.choose(row.expression, quotes, 20), self.router.choose(row.expression, quotes, 20))
        self.assertTrue(0 <= self.router.probability(row.expression) <= 1)
        with patch("model_router.data.answer", side_effect=AssertionError("Oracle must not run during routing")):
            self.assertTrue(features(row.expression))
            self.router.choose(row.expression, quotes, 20)

    def test_zero_budget_abstains(self):
        for policy in ("small", "large", "heuristic", "learned"):
            report = evaluate_policy(self.test, self.router, policy, 0)
            self.assertEqual(report["spent_units"], 0)
            self.assertEqual(report["answered"], 0)
            self.assertIsNone(report["answered_accuracy"])

    def test_budget_is_never_exceeded(self):
        for budget in (1, 2, 7, 15, 80, 180):
            for policy in ("small", "large", "heuristic", "learned"):
                with self.subTest(budget=budget, policy=policy):
                    report = evaluate_policy(self.test, self.router, policy, budget)
                    self.assertLessEqual(report["spent_units"], budget)
                    self.assertEqual(sum(row["cost_units"] for row in report["decisions"]), report["spent_units"])

    def test_router_abstains_before_unaffordable_dispatch(self):
        self.assertIsNone(self.router.choose("1 + 2", {"small": 3, "large": 7}, 2))
        self.assertEqual(self.router.choose("1 + 2", {"small": 3, "large": 7}, 3), "small")
        for budget in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                self.router.choose("1 + 2", {"small": 1, "large": 7}, budget)

    def test_model_roundtrip_and_invalid_weights(self):
        artifact = self.router.to_dict()
        self.assertEqual(Router.from_dict(json.loads(json.dumps(artifact))), self.router)
        artifact["weights"] = [float("nan")] * 6
        with self.assertRaises(ValueError):
            Router.from_dict(artifact)

    def test_exact_budget_oracle(self):
        record = Record("a", "a", "test", "2 + 3", (Response("small", "0", 1), Response("large", "5", 4)))
        other = Record("b", "b", "test", "3 * 2", (Response("small", "6", 2), Response("large", "6", 5)))
        self.assertEqual(oracle_score([record, other], 3), 1)
        self.assertEqual(oracle_score([record, other], 6), 2)
        self.assertEqual(oracle_score([record, other], 0), 0)

    def test_untrusted_numeric_output_cannot_trigger_huge_fraction_expansion(self):
        row = Record(
            "numeric-limit",
            "numeric-limit",
            "test",
            "1 + 1",
            (Response("small", "1e999999999", 1), Response("large", "2", 7)),
        )
        report = evaluate_policy([row], self.router, "small", 1)
        self.assertEqual(report["correct"], 0)

    def test_test_set_cannot_be_declared_training(self):
        router = replace(
            self.router, training_template_ids=(*self.router.training_template_ids, self.test[0].template_id)
        )
        with self.assertRaisesRegex(ValueError, "held-out"):
            evaluate(self.records, router=router)

    def test_renaming_groups_cannot_hide_training_set_reuse(self):
        swap = {"train": "test", "test": "train", "calibration": "calibration"}
        remapped = [
            replace(row, split=swap[row.split], template_id="renamed-" + row.template_id) for row in self.records
        ]
        validate_splits(remapped)
        with self.assertRaisesRegex(ValueError, "training provenance"):
            evaluate(remapped, router=self.router)

    def test_report_distinguishes_simulation_from_live_results(self):
        report = evaluate(self.records, router=self.router)
        self.assertFalse(report["model_evaluation"])
        self.assertEqual(report["response_provenance"], ["SIMULATED"])
        for policy in report["policies"]:
            self.assertIsNone(policy["recorded_latency_sum_ms"])
            self.assertGreaterEqual(policy["correct_count_regret"], 0)

    def test_record_validation_and_replay_roundtrip(self):
        with self.assertRaises(ValueError):
            Response("small", "2", 0)
        with self.assertRaises(ValueError):
            Response("small", "2", 1, latency_ms=12)
        with self.assertRaises(ValueError):
            Response("small", "2", 1, provenance="RECORDED")
        recorded = Response(
            "small",
            "2",
            1,
            latency_ms=12,
            provenance="RECORDED",
            source_model="local-checkpoint@sha256:example",
            recorded_at="2026-11-13T00:00:00+00:00",
        )
        self.assertEqual(recorded.latency_ms, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(serialized(self.records))
            self.assertEqual(load(str(path)), self.records)

    def test_learned_classifier_fits_separable_data(self):
        weights = fit_logistic([[1.0, -1.0], [1.0, 1.0]], [0, 1])
        self.assertGreater(sigmoid(weights[0] + weights[1]), 0.9)
        self.assertLess(sigmoid(weights[0] - weights[1]), 0.1)

    def test_canonical_ast_shape_blocks_parentheses_and_literal_spelling_leaks(self):
        self.assertEqual(expression_structure("1 + 2"), expression_structure("(((0x3 + 4_0)))"))
        self.assertNotEqual(expression_structure("1 + 2"), expression_structure("1 * 2"))
        rows = list(self.records)
        training = next(row for row in rows if row.split == "train")
        index = next(index for index, row in enumerate(rows) if row.split == "test")
        rows[index] = replace(rows[index], expression="((" + training.expression + "))")
        with self.assertRaisesRegex(ValueError, "structure leakage"):
            validate_splits(rows)

    def test_calibration_labels_do_not_change_predictor_training(self):
        changed = [
            replace(row, responses=tuple(replace(response, text="invalid") for response in row.responses))
            if row.split == "calibration"
            else row
            for row in self.records
        ]
        retrained = train(changed, 180)
        self.assertEqual(retrained.weights, self.router.weights)
        self.assertEqual(retrained.training_hash, self.router.training_hash)
        self.assertNotEqual(retrained.calibration_hash, self.router.calibration_hash)

    def test_training_recorded_metadata_does_not_claim_recorded_test_evaluation(self):
        rows = list(self.records)
        index = next(index for index, row in enumerate(rows) if row.split == "train")
        row = rows[index]
        recorded = replace(
            row.responses[0],
            provenance="RECORDED",
            source_model="fixture@sha256:declared",
            recorded_at="2026-11-13T00:00:00+00:00",
        )
        rows[index] = replace(row, responses=(recorded, row.responses[1]))
        report = evaluate(rows)
        self.assertFalse(report["model_evaluation"])
        self.assertEqual(report["evaluation_mode"], "simulated")
        self.assertEqual(report["test_response_provenance"], ["SIMULATED"])

    def test_mixed_test_replay_provenance_is_counted_per_selected_response(self):
        rows = list(self.records)
        index = next(index for index, row in enumerate(rows) if row.split == "test")
        row = rows[index]
        recorded = replace(
            row.responses[0],
            provenance="RECORDED",
            source_model="fixture@sha256:declared",
            recorded_at="2026-11-13T00:00:00+00:00",
        )
        rows[index] = replace(row, responses=(recorded, row.responses[1]))
        report = evaluate(rows, router=self.router)
        self.assertEqual(report["evaluation_mode"], "mixed_recorded_and_simulated")
        self.assertFalse(report["model_evaluation"])
        small = next(policy for policy in report["policies"] if policy["policy"] == "small")
        self.assertEqual(small["selected_response_provenance_counts"]["RECORDED"], 1)
        self.assertEqual(sum(small["selected_response_provenance_counts"].values()), small["answered"])

    def test_budget_oracle_matches_exhaustive_joint_choices(self):
        rng = random.Random(72)
        for trial in range(12):
            rows = []
            for index in range(4):
                responses = tuple(
                    Response(model, "2" if rng.random() < 0.6 else "3", rng.randint(1, 7))
                    for model in ("small", "large")
                )
                rows.append(Record(f"{trial}-{index}", f"g{index}", "test", "1 + 1", responses))
            for budget in (0, 1, 4, 8, 16):
                best = (0, 0)
                for choices in itertools.product((None, "small", "large"), repeat=len(rows)):
                    spent = sum(row.response(choice).cost_units for row, choice in zip(rows, choices) if choice)
                    if spent <= budget:
                        solved = sum(correct(row, choice) for row, choice in zip(rows, choices) if choice)
                        best = max(best, (solved, -spent))
                plan = oracle_plan(rows, budget)
                with self.subTest(trial=trial, budget=budget):
                    self.assertEqual((plan["correct"], -plan["spent_units"]), best)
                    self.assertEqual(sum(row["cost_units"] for row in plan["selected"]), plan["spent_units"])
                    self.assertEqual(len({row["case_id"] for row in plan["selected"]}), plan["correct"])

    def test_oracle_scores_each_response_once_independent_of_budget_states(self):
        rows = [
            Record(
                str(index),
                str(index),
                "test",
                "1 + 1",
                (Response("small", "2", index + 1), Response("large", "2", 100 + index)),
            )
            for index in range(40)
        ]
        with patch("model_router.core.correct", wraps=correct) as scorer:
            result = oracle_plan(rows, 100_000)
        self.assertEqual(result["correct"], 40)
        self.assertEqual(scorer.call_count, 80)

    def test_calibration_diagnostics_include_raw_comparison_and_exact_denominators(self):
        perfect = calibration_metrics([0.0, 1.0], [0, 1])
        self.assertEqual(perfect["brier_score"], 0)
        self.assertEqual(perfect["ece_5_bins"], 0)
        self.assertEqual(sum(row["count"] for row in perfect["bins"]), 2)
        wrong = calibration_metrics([1.0, 0.0], [0, 1])
        self.assertEqual(wrong["brier_score"], 1)
        self.assertEqual(wrong["ece_5_bins"], 1)
        report = evaluate(self.records, router=self.router)
        self.assertAlmostEqual(
            report["calibration"]["brier_delta_calibrated_minus_raw"],
            report["calibration"]["brier_score"] - report["calibration"]["uncalibrated"]["brier_score"],
        )
        for probabilities, targets in (([float("nan")], [1]), ([1.1], [1]), ([0.5], [True])):
            with self.assertRaises(ValueError):
                calibration_metrics(probabilities, targets)

    def test_budget_sweep_freezes_router_and_reports_order_sensitive_bounds(self):
        before = self.router.to_dict()
        with patch("model_router.core.train", side_effect=AssertionError("Sweep must not refit test data")):
            report = evaluate_sweep(self.records, [0, 40, 180], orderings=4, router=self.router)
            repeated = evaluate_sweep(self.records, [0, 40, 180], orderings=4, router=self.router)
        self.assertEqual(report, repeated)
        self.assertEqual(self.router.to_dict(), before)
        self.assertFalse(report["test_used_for_parameter_selection"])
        for outcome in report["outcomes"]:
            for policy in outcome["policies"]:
                self.assertEqual(len(policy["runs"]), 4)
                for run in policy["runs"]:
                    self.assertLessEqual(run["spent_units"], outcome["budget_units"])
                    self.assertLessEqual(run["correct"], outcome["oracle_correct"])
                    self.assertEqual(run["regret"], outcome["oracle_correct"] - run["correct"])

    def test_artifact_records_calibration_budget_and_transfer_is_explicit(self):
        self.assertEqual(self.router.to_dict()["schema_version"], 2)
        self.assertEqual(self.router.calibration_budget_units, 180)
        report = evaluate(self.records, budget=80, router=self.router)
        self.assertTrue(report["budget_transfer"]["different_budget_regime"])
        self.assertEqual(report["router"]["threshold"], self.router.threshold)
        old = self.router.to_dict()
        old["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "retrain"):
            Router.from_dict(old)

    def test_invalid_provenance_and_metadata_fail_at_boundary(self):
        for options in ({"source_model": {"bad": "metadata"}}, {"recorded_at": "fake"}, {"source_model": "invented"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Response("small", "2", 1, **options)
        for identity in (123, "x" * 129):
            with self.assertRaises(ValueError):
                Record(identity, "group", "test", "1 + 1", (Response("small", "2", 1), Response("large", "2", 2)))

    def test_imports_reject_oversized_nonobject_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            for text in ("[]\n", "null\n", "x" * 100_001, '{"responses": [], "responses": []}', '{"cost": NaN}'):
                path.write_text(text)
                with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                    load(str(path))
            path.write_text(serialized(self.records))
            with patch("model_router.data.MAX_DATASET_CHARS", 100), self.assertRaises(ValueError):
                load(str(path))
            path.write_text("x" * 1_048_577)
            with self.assertRaisesRegex(ValueError, "1 MiB"):
                load_router(str(path))
            path.write_text(json.dumps(self.router.to_dict()))
            self.assertEqual(load_router(str(path)), self.router)

    def test_nonfinite_training_and_artifact_inputs_are_rejected(self):
        for rows, targets, options in (
            ([[1.0, float("nan")]], [1], {}),
            ([[1.0]], [True], {}),
            ([[1.0]], [1], {"epochs": 0}),
            ([[1.0]], [1], {"learning_rate": float("inf")}),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit_logistic(rows, targets, **options)
        artifact = self.router.to_dict()
        artifact["weights"] = [True] * 6
        with self.assertRaises(ValueError):
            Router.from_dict(artifact)
        artifact = self.router.to_dict()
        artifact["training_hash"] = "unverifiable"
        with self.assertRaises(ValueError):
            Router.from_dict(artifact)
        with self.assertRaises(ValueError):
            self.router.choose("1 + 1", {"small": 1, "large": 7}, 10, threshold=True)

    def test_negative_budget_is_rejected_before_any_training(self):
        with patch("model_router.core.fit_logistic", side_effect=AssertionError("Must validate budget first")):
            with self.assertRaises(ValueError):
                evaluate(self.records, budget=-1)

    def test_stream_import_never_requests_an_unbounded_line(self):
        class BoundedStream(io.StringIO):
            def readline(self, size=-1):
                if not 0 < size <= 100_001:
                    raise AssertionError("Importer must bound allocations before reading untrusted lines")
                return super().readline(size)

        with patch("builtins.open", return_value=BoundedStream(serialized(self.records))):
            self.assertEqual(load("untrusted-stream"), self.records)


if __name__ == "__main__":
    unittest.main()
