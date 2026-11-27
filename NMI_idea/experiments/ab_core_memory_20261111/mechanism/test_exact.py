#!/usr/bin/env python3
"""Small independent regression tests for the exact mechanism model."""

from __future__ import annotations

import math
import unittest

import run_exact as ex


class ExactMechanismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ex.Model()

    def test_one_probe_does_not_identify_s(self) -> None:
        for action in ("probe_R", "probe_P"):
            for observation in ex.SIGNS:
                post = ex.posterior(((action, observation),), self.model.accuracy)
                self.assertAlmostEqual(ex.side_probability(post, -1), 0.5)
                self.assertAlmostEqual(ex.side_probability(post, 1), 0.5)

    def test_two_probe_formula_and_enumeration(self) -> None:
        expected = self.model.accuracy**2 + (1.0 - self.model.accuracy) ** 2
        self.assertAlmostEqual(expected, 0.9232)
        node = ex.fixed_open_loop_plan(2, self.model)
        exact, elapsed = ex.evaluate_policy_exact(node, self.model)
        self.assertAlmostEqual(node.value, expected)
        self.assertAlmostEqual(exact, expected)
        self.assertEqual(elapsed, 5.0)

    def test_horizon_threshold_and_common_deadline(self) -> None:
        values = {}
        for h in (1, 2, 3, 4):
            node = ex.adaptive_plan(h, self.model)
            value, elapsed = ex.evaluate_policy_exact(node, self.model)
            values[h] = value
            self.assertLessEqual(ex.max_elapsed(node, self.model), self.model.budget)
            self.assertEqual(ex.max_terminals_on_path(node), 1)
            self.assertAlmostEqual(node.value, value)
            self.assertAlmostEqual(node.expected_time, elapsed)
        self.assertAlmostEqual(values[1], 0.67)
        self.assertAlmostEqual(values[2], 0.67)
        self.assertAlmostEqual(values[3], 0.9232)
        self.assertAlmostEqual(values[4], 0.9232)

    def test_fixed_window_and_ignore_diagnostic(self) -> None:
        fixed0 = ex.fixed_open_loop_plan(0, self.model)
        fixed1 = ex.fixed_open_loop_plan(1, self.model)
        fixed2 = ex.fixed_open_loop_plan(2, self.model)
        self.assertAlmostEqual(ex.evaluate_policy_exact(fixed0, self.model)[0], 0.67)
        self.assertAlmostEqual(ex.evaluate_policy_exact(fixed1, self.model)[0], 0.67)
        self.assertAlmostEqual(ex.evaluate_policy_exact(fixed2, self.model)[0], 0.9232)

        ignored = ex.adaptive_plan(4, self.model, assumed_accuracy=0.5)
        actual, elapsed = ex.evaluate_policy_exact(ignored, self.model)
        self.assertEqual(ignored.action, "terminal")
        self.assertEqual(ignored.terminal_choice, "fallback")
        self.assertAlmostEqual(actual, 0.67)
        self.assertEqual(elapsed, 3.0)

    def test_prefix_memory_is_the_only_difference(self) -> None:
        result = ex.prefix_results(self.model)
        rows = {row["memory_policy"]: row for row in result["rows"]}
        self.assertAlmostEqual(rows["full_history"]["exact_enumerated_success"], 0.9232)
        self.assertAlmostEqual(rows["replay"]["exact_enumerated_success"], 0.9232)
        self.assertAlmostEqual(rows["address_cache"]["exact_enumerated_success"], 0.9232)
        self.assertAlmostEqual(rows["window1"]["exact_enumerated_success"], 0.67)
        self.assertAlmostEqual(rows["window2"]["exact_enumerated_success"], 0.67)
        self.assertEqual(rows["window1"]["target_action"], "terminal:fallback")
        self.assertEqual(rows["window2"]["target_action"], "terminal:fallback")
        self.assertEqual(rows["full_history"]["target_action"], "probe_P")
        self.assertTrue(result["protocol"]["shared_public_exit_and_time"])
        self.assertTrue(result["protocol"]["no_old_patch_revisit"])

    def test_counterexamples(self) -> None:
        result = ex.counterexample_results(self.model)
        self.assertAlmostEqual(
            result["no_information_accuracy_0.5"]["adaptive_h3_exact_success"], 0.67
        )
        self.assertAlmostEqual(
            result["memory_epoch_reset_stale_R"]["stale_old_R_times_current_P_success"],
            0.5,
        )
        self.assertAlmostEqual(
            result["model_mismatch_actual_accuracy_0.55_assumed_0.96"]["actual_exact_success"],
            0.505,
        )
        negative = result["minimal_single_probe_negative_control"]
        self.assertAlmostEqual(negative["adaptive_exact_success"], 0.5)
        self.assertAlmostEqual(negative["fixed_one_probe_exact_success"], 0.5)
        self.assertAlmostEqual(negative["A_and_B_upper_bound"], 0.5)


if __name__ == "__main__":
    unittest.main()

