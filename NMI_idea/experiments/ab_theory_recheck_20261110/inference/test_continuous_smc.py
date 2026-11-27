"""Numerical and API contracts for :mod:`continuous_smc`.

These are deliberately small self-contained tests.  The Gaussian tests use
the same generic tempered resample--move engine as the geometric wrapper;
they are not a substitute for the later held-out physical-object evaluation.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from inference.continuous_smc import (
    ContinuousBelief,
    _cess,
    adaptive_next_beta,
    adaptive_tempered_resample_move,
    raw_ess,
)
from observation import environment as env


def test_multi_pose_numpy_batch_matches_per_pose_predictions():
    particles = env.sample_prior(720901, 32)
    poses = np.array([[0.0, -0.02, 0.0], [0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    batched = ContinuousBelief._prediction_for_poses(particles, poses)
    for action_index, pose in enumerate(poses):
        single = env.predict_batch(particles, [pose])
        for key in batched:
            if np.asarray(batched[key]).dtype == bool:
                assert np.array_equal(batched[key][:, action_index], single[key][:, 0])
            else:
                assert np.allclose(
                    batched[key][:, action_index], single[key][:, 0], rtol=0.0, atol=0.0
                )


def test_joint_history_block_uses_one_multi_pose_prediction_call():
    particles = env.sample_prior(720901, 8)
    history = [(_outside_contact_frames("joint-h")[0], np.array([0.0, -0.02, 0.0]))]
    block = [(_outside_contact_frames("joint-b")[0], np.array([0.0, 0.02, 0.0]))]
    with patch.object(env, "predict_batch", wraps=env.predict_batch) as predict_call:
        history_ll, block_ll = ContinuousBelief._joint_loglikes(history, block, particles)
    assert history_ll.shape == (8,)
    assert block_ll.shape == (8,)
    assert predict_call.call_count == 1
    assert np.asarray(predict_call.call_args.args[1]).shape == (2, 3)


def test_cuda_predictor_parity_if_available():
    from inference.gpu_predictor import available, predict_batch

    if not available():
        pytest.skip("CUDA unavailable")
    particles = env.sample_prior(720901, 32)
    poses = np.array([[0.0, -0.02, 0.0], [0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    cpu = env.predict_batch(particles, poses)
    cuda = predict_batch(particles, poses)
    # Position/depth/length geometry is measured in metres; this is the
    # pre-registered 1e-10 m gate.  Angle is dimensionless and gets a looser
    # diagnostic bound for CUDA transcendental reduction order.
    assert np.max(np.abs(cpu["means"][..., :3] - cuda["means"][..., :3])) <= 1e-10
    assert np.max(np.abs(cpu["means"][..., 3] - cuda["means"][..., 3])) <= 2e-6
    assert np.max(np.abs(cpu["means"][..., 4] - cuda["means"][..., 4])) <= 1e-10
    for key in ("p_source", "p_visible", "geometric_length", "angle_audit"):
        assert np.max(np.abs(cpu[key] - cuda[key])) <= 1e-10

    world = env.sample_worlds(720902, 1)[0]
    frames = []
    for j, pose in enumerate(poses):
        frames.extend(env.render_block(world, pose, f"cuda-parity-{j}")[0])
    frames.extend(
        [
            {"frame_id": "cuda-missing", "time_s": 0.0, "visible": False, "contact": False},
            {
                "frame_id": "cuda-identity0",
                "time_s": 0.0,
                "visible": True,
                "contact": True,
                "image_u_m": 0.0,
                "image_depth_m": 0.03,
                "visible_length_m": 0.0,
                "image_axis_rad": 0.0,
                "identity_score": 0.0,
            },
            {
                "frame_id": "cuda-identity1",
                "time_s": 0.0,
                "visible": True,
                "contact": True,
                "image_u_m": 0.001,
                "image_depth_m": 0.04,
                "visible_length_m": 0.02,
                "image_axis_rad": 0.2,
                "identity_score": 1.0,
            },
        ]
    )
    ll_errors = []
    for frame in frames:
        cpu_ll = env.log_likelihood_from_prediction(frame, cpu, 0)
        cuda_ll = env.log_likelihood_from_prediction(frame, cuda, 0)
        finite = np.isfinite(cpu_ll) & np.isfinite(cuda_ll)
        ll_errors.extend(np.abs(cpu_ll[finite] - cuda_ll[finite]))
    assert max(ll_errors, default=0.0) <= 1e-6


def _gaussian_update(seed: int, n: int = 1024):
    rng = np.random.default_rng(seed)
    particles = list(rng.normal(size=n))
    weights = np.full(n, 1.0 / n)
    y, sigma = 1.2, 0.4

    def block_ll(xs):
        x = np.asarray(xs, dtype=float)
        return -0.5 * ((y - x) / sigma) ** 2

    def history_ll(xs):
        return np.zeros(len(xs), dtype=float)

    def prior(x):
        return -0.5 * float(x) ** 2

    def propose(x, proposal_rng, scale):
        return float(x + proposal_rng.normal(0.0, 0.30 * scale))

    return adaptive_tempered_resample_move(
        particles,
        weights,
        block_ll,
        history_ll,
        prior,
        propose,
        rng,
        max_temper_steps=20,
        mh_sweeps=8,
    )


def test_linear_gaussian_posterior_matches_analytic_multiple_seeds():
    # Prior N(0,1), y|x N(x,.4^2): posterior N(1.03448,.13793).
    expected_mean = 1.2 / (1.0 + 0.4**2)
    expected_var = 0.4**2 / (1.0 + 0.4**2)
    means, variances = [], []
    for seed in range(5):
        update = _gaussian_update(720500 + seed)
        assert update.committed, update.diagnostics
        x = np.asarray(update.particles, dtype=float)
        mean = float(update.weights @ x)
        var = float(update.weights @ (x - mean) ** 2)
        means.append(mean)
        variances.append(var)
        assert abs(mean - expected_mean) < 0.10
        assert abs(var - expected_var) < 0.045
        assert update.diagnostics["temperature_path"][-1] == 1.0
    assert abs(float(np.mean(means)) - expected_mean) < 0.04
    assert abs(float(np.mean(variances)) - expected_var) < 0.025


def test_constant_likelihood_is_a_negative_control_and_reaches_endpoint():
    n = 128
    rng = np.random.default_rng(720600)
    particles = list(rng.normal(size=n))
    weights = np.full(n, 1.0 / n)
    old_particles = list(particles)

    update = adaptive_tempered_resample_move(
        particles,
        weights,
        lambda xs: np.full(len(xs), -3.5),
        lambda xs: np.zeros(len(xs)),
        lambda x: -0.5 * float(x) ** 2,
        lambda x, proposal_rng, scale: float(x + proposal_rng.normal(0, 0.2 * scale)),
        rng,
        max_temper_steps=20,
        mh_sweeps=8,
    )
    assert update.committed
    assert update.diagnostics["temperature_path"] == [0.0, 1.0]
    assert np.allclose(update.weights, 1.0 / n)
    assert np.isclose(update.diagnostics["attempted_logZ_increment"], -3.5)
    # The move kernel is allowed to change the finite Monte-Carlo draw, but
    # constant evidence cannot change its weights or temperature schedule.
    assert len(old_particles) == len(update.particles)
    assert adaptive_next_beta(
        0.0, weights, np.full(n, -3.5), target_cess=0.7 * n
    )[0] == 1.0


def test_conditional_ess_formula_is_n_for_nonuniform_weights():
    weights = np.array([0.70, 0.20, 0.10])
    loglike = np.array([1.0, -2.0, 0.5])
    n = len(weights)
    for delta in (0.0, 0.2, 1.0):
        expected = n * np.sum(weights * np.exp(delta * loglike)) ** 2
        expected /= np.sum(weights * np.exp(2.0 * delta * loglike))
        assert np.isclose(_cess(weights, loglike, delta), expected)
    assert np.isclose(_cess(weights, loglike, 0.0), n)
    # Constant evidence leaves ordinary ESS unchanged but CESS remains N for
    # every temperature increment; this is the key distinction from 2*log(w).
    constant = np.full(n, -3.5)
    assert np.isclose(raw_ess(weights), 1.0 / np.sum(weights**2))
    assert np.isclose(_cess(weights, constant, 0.4), n)


def test_two_continuous_blocks_do_not_stall_at_ordinary_ess_plateau():
    n = 128
    rng = np.random.default_rng(720901)
    particles = list(rng.normal(size=n))
    weights = np.full(n, 1.0 / n)
    observed_values = []

    def history_ll(xs):
        x = np.asarray(xs, dtype=float)
        out = np.zeros(len(x), dtype=float)
        for value in observed_values:
            out += -0.5 * ((value - x) / 0.4) ** 2
        return out

    for value in (1.2, -0.4):
        block_ll = lambda xs, value=value: -0.5 * (
            (value - np.asarray(xs, dtype=float)) / 0.4
        ) ** 2
        update = adaptive_tempered_resample_move(
            particles,
            weights,
            block_ll,
            history_ll,
            lambda x: -0.5 * float(x) ** 2,
            lambda x, proposal_rng, scale: float(
                x + proposal_rng.normal(0.0, 0.30 * scale)
            ),
            rng,
            max_temper_steps=20,
            mh_sweeps=8,
        )
        assert update.committed, update.diagnostics
        path = np.asarray(update.diagnostics["temperature_path"])
        assert path[-1] == 1.0
        assert np.all(np.diff(path) > 0.0)
        assert update.diagnostics["stage_count"] < 20
        particles, weights = update.particles, update.weights
        observed_values.append(value)


def _outside_contact_frames(prefix: str, count: int = 5):
    # At this pose contact is deterministically false for every model world;
    # the invisible public observation then has a constant likelihood of one.
    return [
        {
            "frame_id": f"{prefix}-{i}",
            "time_s": 0.125 * i,
            "visible": False,
            "contact": False,
        }
        for i in range(count)
    ]


def test_duplicate_frames_are_skipped_and_history_is_not_double_counted():
    belief = ContinuousBelief(
        env.sample_prior(720901, 128),
        seed=720902,
        max_temper_steps=20,
        mh_sweeps=8,
    )
    frames = _outside_contact_frames("dup")
    first = belief.update_block(frames, [0.041, 0.041, 0.0])
    assert first["posterior_valid"] is True, first
    assert len(belief.history) == 5
    before_logz = belief.logZ
    second = belief.update_block(frames + [frames[0]], [0.041, 0.041, 0.0])
    assert second["skipped_duplicates"] == 6
    assert second["skipped"] is True
    assert len(belief.history) == 5
    assert belief.logZ == before_logz


def test_same_pose_five_frames_reuse_one_prediction():
    particles = env.sample_prior(720901, 128)[:32]
    frames = _outside_contact_frames("reuse")
    original = ContinuousBelief._prediction_for_pose
    calls = {"n": 0}

    def wrapped(ps, pose):
        calls["n"] += 1
        return original(ps, pose)

    with patch.object(ContinuousBelief, "_prediction_for_pose", staticmethod(wrapped)):
        ll = ContinuousBelief._loglike_frames(
            [(f, np.array([0.041, 0.041, 0.0])) for f in frames], particles
        )
    assert ll.shape == (32,)
    assert calls["n"] == 1


def test_unfinished_temperature_path_rolls_back_without_posterior_claim():
    rng = np.random.default_rng(720604)
    particles = list(rng.normal(size=128))
    weights = np.full(128, 1.0 / 128)
    old_ids = [id(p) for p in particles]
    update = adaptive_tempered_resample_move(
        particles,
        weights,
        lambda xs: -0.5 * ((1.2 - np.asarray(xs)) / 0.4) ** 2,
        lambda xs: np.zeros(len(xs)),
        lambda x: -0.5 * float(x) ** 2,
        lambda x, proposal_rng, scale: float(x + proposal_rng.normal(0, 0.30 * scale)),
        rng,
        max_temper_steps=1,
        mh_sweeps=8,
    )
    assert update.committed is False
    assert update.diagnostics["unfinished"] is True
    assert update.diagnostics["posterior_valid"] is False
    assert update.diagnostics["temperature_path"][-1] < 1.0
    assert [id(p) for p in update.particles] == old_ids
    assert np.allclose(update.weights, weights)


def test_near_one_beta_is_not_relabelled_as_completed():
    n = 32
    rng = np.random.default_rng(720902)
    particles = list(rng.normal(size=n))
    weights = np.full(n, 1.0 / n)
    # Force the scheduler boundary that exposed the old beta>=1-1e-9 lie.
    with patch(
        "inference.continuous_smc.adaptive_next_beta",
        return_value=(1.0 - 5e-10, float(n)),
    ):
        update = adaptive_tempered_resample_move(
            particles,
            weights,
            lambda xs: np.full(len(xs), -3.5),
            lambda xs: np.zeros(len(xs)),
            lambda x: -0.5 * float(x) ** 2,
            lambda x, proposal_rng, scale: x,
            rng,
            max_temper_steps=1,
            mh_sweeps=0,
        )
    assert update.committed is False
    assert update.diagnostics["unfinished"] is True
    assert update.diagnostics["beta_final"] == 1.0 - 5e-10
    assert update.diagnostics["temperature_path"][-1] < 1.0


def test_all_particle_support_failure_returns_rollback_diagnostic():
    belief = ContinuousBelief(env.sample_prior(720901, 128), seed=720902)
    # The public contact bit contradicts the actual pose contact for every
    # particle.  No likelihood floor or dropped block is permitted.
    frames = [
        {
            "frame_id": f"support-fail-{i}",
            "time_s": 0.125 * i,
            "visible": False,
            "contact": True,
        }
        for i in range(5)
    ]
    diagnostic = belief.update_block(frames, [0.041, 0.041, 0.0])
    assert diagnostic["failure"] == "all_particles_incompatible"
    assert diagnostic["unfinished"] is True
    assert diagnostic["posterior_valid"] is False
    assert diagnostic["history_committed"] is False
    assert len(belief.history) == 0
    assert len(belief.diagnostics) == 1

