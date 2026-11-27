import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np

import streaming_a as s


class StreamingAContract(unittest.TestCase):
    def test_public_five_frame_LA_and_independent_identity(self):
        frames = [dict(visible=True, contact=True, identity=.9, length=.017,
                       u=.001, angle=.05, evidence_id=f"f{i}", time_s=i*.125)
                  for i in range(5)]
        self.assertTrue(s.public_verifier(frames))
        self.assertFalse(s.public_verifier(frames[:4]))
        wrong = [dict(f) for f in frames]
        wrong[2]["length"] = .0149
        self.assertFalse(s.public_verifier(wrong))
        private = [dict(source="target", contact=True, length=.017,
                        u=.001, angle=.05) for _ in frames]
        self.assertTrue(s.independent_audit(list(zip(frames, private))))
        private[1]["source"] = "false"
        self.assertFalse(s.independent_audit(list(zip(frames, private))))

    def test_unique_evidence_and_normalized_posterior(self):
        b = s.Belief()
        f = dict(evidence_id="one", visible=False)
        b.update(f, s.START, np.eye(2))
        self.assertAlmostEqual(float(np.sum(b.weights)), 1.0)
        with self.assertRaisesRegex(ValueError, "duplicate evidence"):
            b.update(f, s.START, np.eye(2))
        self.assertEqual(len(b.evidence_ids), 1)

    def test_generic_prior_changes_action_without_hidden_world(self):
        b1, b2 = s.Belief(), s.Belief()
        b1.weights[:] = 0
        b1.weights[:16] = 1/16
        b2.weights[:] = 0
        b2.weights[64:] = 1/16
        c = np.array([.4, -.2, .7])
        a1 = int(np.argmax(s.rank_actions(b1, c, s.START, set(), np.eye(2), False)))
        a2 = int(np.argmax(s.rank_actions(b2, c, s.START, set(), np.eye(2), False)))
        self.assertNotEqual(a1, a2)
        # All policy inputs are public. Two different hidden worlds with this
        # same public prefix must therefore get the same selected action.
        self.assertEqual(a1, int(np.argmax(s.rank_actions(
            b1, c.copy(), s.START, set(), np.eye(2), False))))

    def test_DMP_peak_speed_and_actual_endpoint(self):
        bank = s.MotionBank()
        actual, duration = bank.reach(s.START, 1)
        self.assertGreater(duration, 0)
        self.assertLessEqual(bank.max_translation_speed, .020000001)
        self.assertLessEqual(bank.max_yaw_speed, .500000001)
        self.assertLess(bank.max_translation_endpoint_error, .002)
        self.assertLess(bank.max_yaw_endpoint_error, .02)
        self.assertNotEqual(actual, s.POSES[1])

    def test_response_zero_action_and_exact_control(self):
        c = np.array([.7, -.3, .2])
        W = np.zeros((8, 2))
        self.assertTrue(np.allclose(s.feature(c, [0, 0]) @ W, 0))
        self.assertTrue(np.allclose(s.learned_j(c, W), np.eye(2)))
        world = s.sample_worlds(931105, 1, exact=True)[0]
        self.assertTrue(np.allclose(s.response_j(world.public_c,
            world.response_E, exact=True), np.eye(2)))

    def test_counterfactual_hidden_world_and_audit_noninterference(self):
        first = s.sample_worlds(931102, 1)[0]
        second = replace(first, mode=(first.mode+1) % 4,
                         response_E=first.response_E + .2)
        def fixed_public_stream(world, pose, seed, world_id, ordinal, action_id,
                                frame_id, exact=False):
            public = dict(visible=True, contact=True, identity=.9,
                          length=.014, u=0.0, angle=0.0, depth=.04)
            private = dict(source="target" if world is first else "false",
                           contact=True, length=.017, u=0.0, angle=0.0)
            return public, private
        model = dict(W=np.zeros((8, 2)).tolist(),
                     W_constant=np.zeros((2, 2)).tolist())
        with patch.object(s, "render_frame", fixed_public_stream):
            a = s.run_episode(first, "analytic_geometry_A", model,
                              s.MotionBank(), "counterfactual", 0, 1)
            b = s.run_episode(second, "analytic_geometry_A", model,
                              s.MotionBank(), "counterfactual", 0, 1)
        self.assertEqual([t["action_id"] for t in a["trace"]],
                         [t["action_id"] for t in b["trace"]])
        self.assertEqual(a["success_ever_independent_LA"], True)
        self.assertEqual(b["success_ever_independent_LA"], False)
        self.assertFalse(a["public_stop"])
        self.assertFalse(b["public_stop"])


if __name__ == "__main__":
    unittest.main()
