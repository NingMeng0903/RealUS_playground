"""Development-only contract tests for the continuous observation API.

These tests intentionally use only the development seed in PROTOCOL.json.  In
particular they never instantiate the locked test cohort or use audit values as
policy inputs.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from observation.environment import (  # noqa: E402
    FIELDS,
    Belief,
    FRAME_OFFSETS,
    _geometry_batch,
    log_likelihood_from_prediction,
    log_likelihood_batch,
    particle_axis_plane_angle,
    posterior_angle_probability,
    public_stop,
    predict_batch,
    render,
    render_model,
    sample_model_observation,
    sample_prior,
    sample_worlds,
)
from observation.servo import candidate_actions  # noqa: E402


DEV_SEED = 720281


def _visible_case(world):
    """Find a deterministic visible development pose from a private world.

    This helper is test-only.  A controller cannot call it because controller
    APIs receive public history and model particles, never ``world.points``.
    """

    for source in range(2):
        for knot in (25, 40, 50, 60, 75):
            point = world.points[source, knot]
            pose = np.asarray([point[0], point[1], 0.0], dtype=float)
            if float(render_model(world, pose)["p_visible"]) > 0.55:
                return pose
    raise AssertionError("development world did not expose a visible source")


class ObservationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 32 is enough to exercise mixture/particle contracts while keeping the
        # development test fast.  No test split is touched here.
        cls.worlds = sample_worlds(DEV_SEED, 8)
        cls.particles = sample_prior(DEV_SEED + 1, 32)

    def test_exact_slab_clip_and_longest_connected_component(self):
        """Every segment is clipped to x/y/depth bounds; disconnected pieces
        contribute only the longest connected component.
        """

        # A straight 100-mm line at y=0, depth=30 mm clips exactly to the
        # 50-mm image width [-25,25] mm.
        straight = np.asarray(
            [[[[ -0.050, 0.0, -0.030], [0.050, 0.0, -0.030]]]], dtype=float
        )
        got = _geometry_batch(straight, [[0.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(got["geometric_length"][0, 0, 0]), 0.050, places=8)
        self.assertAlmostEqual(float(got["length"][0, 0, 0]), 0.050, places=8)
        self.assertTrue(bool(got["exists"][0, 0, 0]))
        self.assertAlmostEqual(float(got["depth"][0, 0, 0]), 0.030, places=8)

        # The middle excursion leaves the +/-2-mm slab.  The two visible
        # returns are disconnected, so the longest connected return is kept.
        disconnected = np.asarray(
            [[[
                [-0.050, 0.000, -0.030],
                [-0.020, 0.000, -0.030],
                [-0.019, 0.010, -0.030],
                [ 0.019, 0.010, -0.030],
                [ 0.020, 0.000, -0.030],
                [ 0.050, 0.000, -0.030],
            ]]],
            dtype=float,
        )
        got = _geometry_batch(disconnected, [[0.0, 0.0, 0.0]])
        # The first return contains the x-clip (5 mm) plus its short boundary
        # segment (~0.2 mm), while the right return is a separate component.
        self.assertGreater(float(got["geometric_length"][0, 0, 0]), 0.005)
        self.assertLess(float(got["geometric_length"][0, 0, 0]), 0.009)
        self.assertLess(float(got["geometric_length"][0, 0, 0]), 0.012)

    def test_miss_has_no_hidden_geometry_fields(self):
        """A slab miss carries only explicit missing values."""

        for index, world in enumerate(self.worlds):
            public, audit = render(world, [0.0, 0.040, 0.0], ("dev-miss", index))
            self.assertFalse(public["visible"])
            self.assertIsNone(public["image_u_m"])
            self.assertIsNone(public["image_depth_m"])
            self.assertIsNone(public["visible_length_m"])
            self.assertIsNone(public["image_axis_rad"])
            self.assertIsNone(public["identity_score"])
            # An audit may still explain the physical miss, but no such value
            # is copied into the public record.
            self.assertIn("source_is_target", audit)

    def test_render_model_and_public_renderer_contract(self):
        """The model exposes only source mixtures/means; rendered frames add
        noise and stable frame IDs while preserving the public field schema.
        """

        world = self.worlds[0]
        pose = _visible_case(world)
        model = render_model(world, pose)
        # ``render_model`` is the scalar-pose convenience view: source x
        # feature, while ``predict_batch`` retains particle x action x source.
        self.assertEqual(model["means"].shape, (2, 5))
        self.assertEqual(model["p_source"].shape, (2,))
        self.assertGreater(float(model["p_visible"]), 0.55)
        self.assertLessEqual(float(np.sum(model["p_source"])), 1.0 + 1e-12)

        public, audit = render(world, pose, ("dev-render", 0), frame_id="dev-render/0", time_s=0.125)
        self.assertEqual(public["frame_id"], "dev-render/0")
        self.assertAlmostEqual(public["time_s"], 0.125, places=12)
        self.assertEqual(set(public), {"frame_id", "time_s", "visible", "contact", *FIELDS})
        if public["visible"]:
            self.assertTrue(np.isfinite([public[name] for name in FIELDS]).all())
            self.assertIn("out_of_plane_angle_rad", audit)

    def test_finite_catalogue_bayes_update_and_duplicate_guard(self):
        """Finite support can be updated exactly without silent resampling."""

        particles = self.particles
        pose = np.asarray([0.0, 0.0, 0.0])
        public = sample_model_observation(particles[3], pose, ("dev-bayes", 0), frame_id="dev-bayes/0")
        ll = log_likelihood_batch(public, particles, pose)
        self.assertTrue(np.isfinite(logsumexp(ll)))

        belief = Belief(particles, ess_fraction=0.0, mh_steps=0, seed=DEV_SEED)
        before = belief.weights.copy()
        diagnostic = belief.update(public, pose, rejuvenate=False)
        expected = np.exp(ll - logsumexp(ll))
        np.testing.assert_allclose(belief.weights, expected, rtol=1e-10, atol=1e-12)
        self.assertFalse(diagnostic["duplicate"])
        self.assertEqual(len(belief.history), 1)
        self.assertAlmostEqual(float(np.sum(belief.weights)), 1.0, places=12)
        self.assertGreaterEqual(float(diagnostic["raw_ess"]), 1.0)
        self.assertFalse(np.allclose(before, belief.weights))

        duplicate = belief.update(public, pose, rejuvenate=False)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(len(belief.history), 1)
        np.testing.assert_allclose(belief.weights, expected, rtol=1e-12, atol=1e-12)

    def test_precomputed_prediction_likelihood_matches_direct(self):
        """B lookahead may reuse P x A geometry without changing likelihoods."""

        pose = np.asarray([0.0, 0.0, 0.0])
        prediction = predict_batch(self.particles, [pose])
        public = sample_model_observation(self.particles[5], pose, ("dev-precomputed", 0),
                                          frame_id="dev-precomputed/0")
        direct = log_likelihood_batch(public, self.particles, pose)
        reused = log_likelihood_from_prediction(public, prediction, action_index=0)
        np.testing.assert_allclose(reused, direct, rtol=1e-12, atol=1e-12,
                                   equal_nan=False)

    def test_zero_source_angle_posterior_and_incompatible_rollback(self):
        """Impossible visible evidence remains finite and failed updates roll back."""

        pose = np.asarray([0.0, 0.040, 0.0])
        visible_public = {
            "frame_id": "dev-zero-source/0",
            "time_s": 0.0,
            "visible": True,
            "contact": True,
            "image_u_m": 0.0,
            "image_depth_m": 0.030,
            "visible_length_m": 0.020,
            "image_axis_rad": 0.0,
            "identity_score": 0.90,
        }
        weights = np.ones(len(self.particles)) / len(self.particles)
        posterior = posterior_angle_probability(
            self.particles, weights, pose, public=visible_public
        )
        self.assertTrue(np.isfinite(posterior["probability"]))
        self.assertEqual(posterior["probability"], 0.0)
        self.assertTrue(np.isfinite(posterior["source_mass"]).all())
        self.assertEqual(float(np.sum(posterior["source_mass"])), 0.0)

        # The pose is inside the public workspace, so contact=False is
        # deliberately incompatible with every particle.  The failed update
        # must not consume its frame ID or alter the prior weights.
        incompatible = dict(visible_public, frame_id="dev-incompatible/0",
                            visible=False, contact=False)
        incompatible.update({field: None for field in FIELDS})
        belief = Belief(self.particles, support="finite", seed=DEV_SEED)
        before = belief.weights.copy()
        with self.assertRaises(ValueError):
            belief.update(incompatible, pose, rejuvenate=False)
        self.assertEqual(belief.history, [])
        self.assertEqual(belief.seen, set())
        np.testing.assert_allclose(belief.weights, before, rtol=0.0, atol=0.0)

    def test_public_five_frame_gate_rejects_duplicate_or_bad_timestamp(self):
        """The public LA proxy needs five unique, correctly spaced new frames."""

        def good_frame(index, time_s):
            return {
                "frame_id": f"dev-gate/{index}",
                "time_s": float(time_s),
                "visible": True,
                "contact": True,
                "image_u_m": 0.0,
                "image_depth_m": 0.030,
                "visible_length_m": 0.020,
                "image_axis_rad": 0.0,
                "identity_score": 0.90,
            }

        valid = [good_frame(i, float(t)) for i, t in enumerate(FRAME_OFFSETS)]
        self.assertTrue(public_stop(valid))

        duplicated = [dict(frame) for frame in valid]
        duplicated[-1]["frame_id"] = duplicated[-2]["frame_id"]
        self.assertFalse(public_stop(duplicated))

        bad_time = [dict(frame) for frame in valid]
        bad_time[-1]["time_s"] += 1.0e-5
        self.assertFalse(public_stop(bad_time))

        # A valid block may start at a nonzero time, but still has the exact
        # declared 0,.125,.25,.375,.5-second spacing.
        shifted = [good_frame(i, 3.0 + float(t)) for i, t in enumerate(FRAME_OFFSETS)]
        self.assertTrue(public_stop(shifted))

    def test_servo_is_public_history_only_and_continuous(self):
        """Changing an unpassed hidden world cannot affect common candidates."""

        world = self.worlds[0]
        pose = _visible_case(world)
        public, _audit = render(world, pose, ("dev-servo", 0), frame_id="dev-servo/0")
        history = [(public, pose)]
        weights = np.ones(len(self.particles)) / len(self.particles)
        actions_a = candidate_actions(pose, history, self.particles, weights, max_candidates=12)

        # The alternative world and audit are deliberately not passed to the
        # servo.  A second call with exactly the same public inputs must match.
        _other_public, other_audit = render(self.worlds[1], pose, ("dev-servo-other", 0), frame_id="dev-servo-other/0")
        actions_b = candidate_actions(pose, history, self.particles, weights, max_candidates=12)
        np.testing.assert_allclose(actions_a, actions_b, rtol=0.0, atol=0.0)

        # At least one candidate is the measured-u centering correction; this
        # proves the action family is continuous rather than a fixed 10-mm table.
        if public["visible"]:
            centered = pose.copy()
            centered[:2] += public["image_u_m"] * np.asarray([np.cos(pose[2]), np.sin(pose[2])])
            self.assertTrue(any(np.linalg.norm((a - centered) * [1.0, 1.0, 0.03]) < 1e-8 for a in actions_a))
            self.assertTrue(np.isfinite(actions_a).all())
        self.assertIn("source_is_target", other_audit)

    def test_particle_axis_helper_is_hypothesis_only(self):
        """Axis posterior helper returns legal particle geometry, never audit truth."""

        pose = np.asarray([0.0, 0.0, 0.0])
        angles = particle_axis_plane_angle(self.particles, pose)
        self.assertEqual(angles.shape, (len(self.particles), 2))
        finite = angles[np.isfinite(angles)]
        self.assertTrue(np.all(finite >= 0.0))
        self.assertTrue(np.all(finite <= np.pi / 2 + 1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
