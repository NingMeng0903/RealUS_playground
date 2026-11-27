"""Non-mirror contract tests for the SENSOR_V2 belief model."""
from __future__ import annotations

import inspect
import time
import unittest

import numpy as np

from acoustics import ProbeAction, PublicObservation
from belief_model import CompleteOption, ModelParams, _geometry_terms, new_belief
from train_calibrate import feature_rows, fit_model, render_sensor_v2


def packet(event_id, *, detected=False, quality=float("nan"), center=float("nan"), axis=float("nan")):
    return PublicObservation(event_id, "public:asymmetric", "window_left", .30,
                             "checked", detected, .0, quality, center, axis,
                             "candidate" if detected else "missing")


class SensorV2BeliefTests(unittest.TestCase):
    def setUp(self):
        self.action = ProbeAction("window_left", (-.011, .014), .043, .30, "probe")
        self.other = ProbeAction("window_right", (.009, .017), -.031, .60, "probe")

    def test_public_likelihood_normalizes_and_has_no_labels(self):
        belief = new_belief()
        prediction = belief.public_predict(self.action)
        self.assertAlmostEqual(prediction.p_candidate + prediction.p_missing, 1.0, places=12)
        self.assertAlmostEqual(sum(component.weight for component in prediction.components), prediction.p_candidate, places=10)
        self.assertTrue(all(not hasattr(component, "target") for component in prediction.components))
        self.assertGreaterEqual(prediction.p_candidate, 0.)
        self.assertLessEqual(prediction.p_candidate, 1.)
        self.assertAlmostEqual(prediction.p_public_accept,
                               prediction.p_correct_accept + prediction.p_false_accept, places=12)
        self.assertLessEqual(prediction.p_public_accept, prediction.p_candidate + 1e-12)

    def test_duplicate_id_replay_and_fork(self):
        belief = new_belief()
        miss = packet("e0")
        first = belief.update("e0", self.action, miss)
        snapshot = belief.snapshot()
        duplicate = belief.update("e0", self.action, miss)
        self.assertTrue(first.applied)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(snapshot, belief.snapshot())
        with self.assertRaises(ValueError):
            belief.update("e0", self.other, miss)
        branch = belief.clone()
        branch.update("e1", self.other, packet("e1", detected=True, quality=.62, center=.001, axis=.01))
        self.assertNotIn("e1", belief.event_ids)

    def test_missing_keeps_response_moments_but_updates_loss_mode(self):
        belief = new_belief()
        before_mean, before_cov = belief.posterior_moments()
        belief.update("miss", self.action, packet("miss"))
        after_mean, after_cov = belief.posterior_moments()
        np.testing.assert_allclose(before_mean, after_mean, atol=1e-10)
        np.testing.assert_allclose(before_cov, after_cov, atol=1e-10)
        self.assertGreater(belief.mode_weights[:, 1].mean(), 0.)

    def test_candidate_uses_quality_center_axis_and_keeps_clutter_branch(self):
        belief = new_belief()
        result = belief.update("candidate", self.action,
                               packet("candidate", detected=True, quality=.61, center=.0007, axis=.018))
        self.assertTrue(result.applied)
        self.assertGreater(result.component_count, 0)
        self.assertTrue(np.isfinite(result.log_evidence))
        self.assertEqual(sorted(belief.event_ids), ["candidate"])

    def test_complete_option_contains_joint_gates_and_retry(self):
        belief = new_belief()
        one = belief.predict_complete_option(CompleteOption((self.action,), max_retries=1, gh_order=9))
        two = belief.predict_complete_option(CompleteOption((self.action, self.action), max_retries=2, gh_order=9))
        for result in (one, two):
            self.assertTrue(0. <= result.raw_complete_p <= 1.)
            self.assertTrue(0. <= result.p_false_early_stop <= 1.)
            self.assertTrue(0. <= result.p_target_identity <= 1.)
        self.assertGreaterEqual(two.raw_complete_p, one.raw_complete_p)
        self.assertGreater(two.p_false_early_stop, 0.)

    def test_non_mirror_dwell_and_batched_predictions(self):
        belief = new_belief()
        short = ProbeAction("window_left", (-.002, .010), .017, .15, "probe")
        long = ProbeAction("window_left", (-.019, .021), .061, .60, "probe")
        outputs = belief.batched_outcomes((short, long))
        self.assertEqual(len(outputs), 2)
        self.assertNotEqual(outputs[0].components[0].quality_variance,
                            outputs[1].components[0].quality_variance)

    def test_signed_geometry_is_not_abs_folded(self):
        geometry = np.array([[0.001, 0.02, 0.0, .022]], dtype=float)
        lower = ProbeAction("window_left", (0., .012), .0, .30, "probe")
        upper = ProbeAction("window_left", (0., .024), .0, .30, "probe")
        lower_center = float(_geometry_terms(geometry, lower)[0][0])
        upper_center = float(_geometry_terms(geometry, upper)[0][0])
        self.assertLess(lower_center, 0.)
        self.assertGreater(upper_center, 0.)

    def test_training_generator_and_no_policy_truth(self):
        rows = feature_rows(721501, 3, repeats=1)
        self.assertTrue(rows)
        self.assertTrue(all("packet" in row and "truth" in row for row in rows))
        model = fit_model(rows)
        self.assertIsInstance(model, ModelParams)
        source = inspect.getsource(__import__("belief_model"))
        self.assertNotIn("PrivateScenario", source)
        self.assertNotIn("HiddenFrame", source)

    def test_rendered_packet_is_publicly_unlabeled(self):
        rows = feature_rows(721501, 1)
        public = rows[0]["packet"]
        self.assertIn(public.candidate_type, ("candidate", "missing"))
        self.assertFalse(hasattr(public, "target_id"))

    def test_update_and_complete_are_bounded(self):
        belief = new_belief()
        started = time.perf_counter()
        for index in range(3):
            action = self.action if index % 2 == 0 else self.other
            belief.update(f"event-{index}", action, packet(f"event-{index}"))
        update_time = time.perf_counter() - started
        started = time.perf_counter()
        belief.predict_complete_option(CompleteOption((self.action, self.action), gh_order=33))
        complete_time = time.perf_counter() - started
        # Guard only against accidental quadratic regressions; this is not a
        # real-time claim and is reported separately from task cost.
        self.assertLess(update_time, 2.0)
        self.assertLess(complete_time, 2.0)


if __name__ == "__main__":
    unittest.main()
