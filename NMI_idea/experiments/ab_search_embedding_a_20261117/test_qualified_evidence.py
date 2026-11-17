"""Check coverage, replacement and revocation using the real online update."""

import unittest
from dataclasses import replace

import numpy as np

from adaptive_controller import (BINS, EDGES, END, LEGACY_EVIDENCE, QUALIFIED_EVIDENCE, VISIBLE_EVIDENCE,
                                 AdaptiveController, ScanState, final_metrics)


class PublicPackets:
    def __init__(self):
        self.mass = np.ones(2)
        self.responsibility = np.array([1., .4])
        self.centers = np.array([[-6., 8.], [-6., 8.]])

    def observe(self, scene, frame, s, x):
        return np.array([[0., 8.]]), np.array([0]), self.centers[0]

    def likelihood(self, packet, s, x, q):
        return self.mass, self.responsibility[:, None]

    def target_grid(self, s, q):
        return self.centers

    def identity_prob(self, belief, s, q):
        return .96


class QualifiedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.controller = AdaptiveController.__new__(AdaptiveController)
        self.controller.world = PublicPackets()
        self.controller.evidence_mode = QUALIFIED_EVIDENCE
        self.state = ScanState(np.array([.96, .04]), 3., 0.)
        self.record = self.controller.virtual_record(self.state, detailed=True)

    def observe(self):
        self.controller.observe(None, self.state, self.record)
        self.state.frame += 1

    def test_pending_candidate_is_replaced_by_a_centered_image(self):
        self.observe()
        self.assertIsNotNone(self.state.ledger[0])
        self.assertIn(0, self.state.uncovered())
        self.assertIsNone(self.state.last_trusted)
        self.state.x = -6.
        self.observe()
        self.assertNotIn(0, self.state.uncovered())
        self.assertEqual(self.state.ledger[0].frame, 1)
        self.assertEqual(self.record.retractions[0]["reason"], "higher_centered_probability")

    def test_identity_contradiction_revokes_a_witness(self):
        self.state.x = -6.
        self.observe()
        self.controller.world.mass = np.array([.001, 1.])
        self.observe()
        self.assertIsNone(self.state.ledger[0])
        self.assertIn(0, self.state.uncovered())
        self.assertEqual(self.record.retractions[0]["witness"], 0)

    def test_updated_geometric_uncertainty_reopens_a_bin_at_the_endpoint(self):
        self.state.s = END
        self.state.x = -6.
        self.controller.world.centers[1, 0] = -12.
        self.controller.world.responsibility[:] = 1.
        self.observe()
        witness = self.state.ledger[-1]
        self.assertNotIn(BINS - 1, self.state.uncovered())
        self.state.ledger = (witness,) * BINS
        self.state.covered = (1 << BINS) - 1
        self.assertFalse(self.controller.can_continue(self.state))
        self.controller.world.mass = np.array([.04, .96])
        self.observe()
        self.assertIs(self.state.ledger[-1], witness)
        self.assertIn(BINS - 1, self.state.uncovered())
        self.assertTrue(self.controller.can_continue(self.state))

    def test_historical_mode_reproduces_first_image_coverage(self):
        self.controller.evidence_mode = LEGACY_EVIDENCE
        self.observe()
        self.state.x = -6.
        self.observe()
        self.assertEqual(self.state.ledger[0].frame, 0)
        self.assertNotIn(0, self.state.uncovered())
        self.assertEqual(self.record.retractions, [])

    def enable_visible(self, initial_prior):
        self.controller.evidence_mode = VISIBLE_EVIDENCE
        self.state.belief = np.array(initial_prior)
        self.state.image_candidates = (((), np.empty((0, 2))),) * BINS
        self.state.capture_support = (np.zeros(2),) * BINS

    def test_later_evidence_confirms_an_off_center_image_from_an_earlier_interval(self):
        self.enable_visible([.5, .5])
        self.controller.world.responsibility[:] = [1., 0.]
        self.observe()
        self.assertEqual(self.state.slot, -1)
        self.assertIsNone(self.state.ledger[0])
        self.state.s = 8.
        self.controller.world.mass = np.array([.999, .001])
        self.observe()
        self.assertEqual(self.state.ledger[0].frame, 0)
        event = self.record.certificates[0]
        self.assertEqual((event["frame"], event["accepted_at"]), (0, 1))
        self.assertFalse(event["center_ok"])
        self.assertNotIn(0, self.state.uncovered())
        self.controller.world.mass = np.array([.00001, 1.])
        self.observe()
        self.assertIn(0, self.state.uncovered())
        self.assertEqual(len(self.state.image_candidates[0][0]), 1)

    def test_virtual_rollout_keeps_prior_candidates_without_evaluator_labels(self):
        self.enable_visible([.5, .5])
        self.controller.world.responsibility[:] = [1., 0.]
        self.observe()
        fork = replace(self.state)
        record = self.controller.virtual_record(fork)
        self.controller.world.mass = np.array([.999, .001])
        self.controller.observe(None, fork, record)
        event = record.certificates[0]
        self.assertNotIn("identity_ok", event)
        np.testing.assert_equal(event["predicted_identity"], [1., 0.])
        self.assertIsNone(self.state.ledger[0])
        self.assertEqual(final_metrics(fork, record, sampled_latent=0)["new_identity_mistake"], 0.)

    def test_centering_is_recorded_separately_from_visible_completion(self):
        self.enable_visible([.99, .01])
        for s in (EDGES[:-1] + EDGES[1:]) / 2.:
            self.state.s = s
            self.observe()
        self.state.s = END
        metrics = final_metrics(self.state, self.record, query_bin=BINS - 1)
        self.assertEqual(metrics["corrected_complete"], 1.)
        self.assertEqual(metrics["centered_complete"], 0.)
        self.assertEqual(metrics["abs_center_error_sum_mm"], 6. * BINS)


if __name__ == "__main__":
    unittest.main()
