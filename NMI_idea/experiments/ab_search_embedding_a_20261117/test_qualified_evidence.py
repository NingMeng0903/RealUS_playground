"""Check coverage, replacement and revocation using the real online update."""

import unittest

import numpy as np

from adaptive_controller import BINS, END, LEGACY_EVIDENCE, QUALIFIED_EVIDENCE, AdaptiveController, ScanState


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


if __name__ == "__main__":
    unittest.main()
