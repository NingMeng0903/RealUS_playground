"""Contracts for the public-history-conditioned importance proposal."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm, uniform
from scipy.special import logsumexp

from inference import informed_proposal as ip
from observation import environment as env


def _scipy_log_prior_continuous(x):
    x = np.asarray(x, dtype=float)
    value = norm.logpdf(x[ip.REG], scale=0.004).sum()
    cov_y = 0.003**2 * (env.CHOL @ env.CHOL.T)
    cov_z = 0.0015**2 * (env.CHOL @ env.CHOL.T)
    for source in range(2):
        value += multivariate_normal.logpdf(x[2 + 5 * source : 2 + 5 * (source + 1)], mean=np.zeros(5), cov=cov_y)
        value += multivariate_normal.logpdf(x[12 + 5 * source : 12 + 5 * (source + 1)], mean=np.zeros(5), cov=cov_z)
    value += uniform.logpdf(x[ip.DEPTH], loc=0.025, scale=0.020).sum()
    value += norm.logpdf(x[ip.IDENTITY], scale=1.0).sum()
    value += norm.logpdf(x[ip.DETECT], scale=1.0).sum()
    return float(value)


def _public_history(seed=720902, y=-0.02):
    world = env.sample_worlds(seed, 1)[0]
    pose = np.array([0.0, y, 0.0])
    frames, _ = env.render_block(world, pose, (0, "iid", "dev"))
    return [(frame, pose.copy()) for frame in frames]


def test_normalized_prior_matches_independent_scipy_and_samples():
    vectors, modes = ip.sample_prior_vectors(32, seed=720901)
    for x, mode in zip(vectors[:8], modes[:8]):
        assert np.isclose(ip.log_prior_continuous(x), _scipy_log_prior_continuous(x), atol=1e-9)
        assert np.isclose(ip.log_prior_vector(x, mode), _scipy_log_prior_continuous(x) - np.log(16.0), atol=1e-9)
    assert np.all((vectors[:, ip.DEPTH] >= 0.025) & (vectors[:, ip.DEPTH] <= 0.045))
    assert np.all(np.isfinite(ip.log_prior_continuous(vectors)))


def test_mixture_q_is_proper_and_importance_weights_are_independent_of_old_weights():
    proposal = ip.InformedProposal(_public_history(), pattern_seed=720907)
    assert len(proposal.components) == 16 * len(proposal.patterns)
    assert np.isclose(sum(component.weight for component in proposal.components), 1.0)
    vectors, modes, _worlds = proposal.sample(128, seed=720908)
    log_q = proposal.log_q(vectors, modes)
    log_prior = proposal.log_prior(vectors, modes)
    assert np.all(np.isfinite(log_q))
    # Independent single-sample mixture evaluation using scipy's normalized
    # multivariate Gaussian density, including only components with this mode.
    i = 0
    mode_tuple = tuple(modes[i].reshape(-1))
    informed_terms = []
    for component in proposal.components:
        if component.mode == mode_tuple:
            informed_terms.append(
                np.log(component.weight)
                + multivariate_normal.logpdf(vectors[i], mean=component.mean, cov=component.covariance)
            )
    manual_q = np.logaddexp(
        np.log(proposal.epsilon) + log_prior[i],
        np.log1p(-proposal.epsilon) + float(logsumexp(informed_terms)),
    )
    assert np.isclose(log_q[i], manual_q, atol=1e-8)
    old_weights = np.linspace(1.0, 2.0, 128)
    old_weights /= old_weights.sum()
    likelihood = proposal.log_likelihood_full(_worlds)
    expected = log_prior + likelihood - log_q
    result = proposal.importance_sample(128, seed=720908)
    assert np.allclose(result.log_weights, expected, equal_nan=True)
    assert result.diagnostics["posterior_valid"] is True
    assert np.isfinite(result.diagnostics["evidence_estimate"])
    # The API has no old-weight input; a caller's stale weights cannot enter q
    # or the new full-history importance ratio.
    assert not np.allclose(expected, expected + np.log(old_weights))


def test_zero_likelihood_support_is_transparent_failure():
    pose = np.array([0.041, 0.041, 0.0])
    frames = [
        {"frame_id": f"zero-{i}", "time_s": 0.125 * i, "visible": False, "contact": True}
        for i in range(5)
    ]
    proposal = ip.InformedProposal([(frame, pose) for frame in frames])
    result = proposal.importance_sample(128, seed=720908)
    assert result.diagnostics["posterior_valid"] is False
    assert result.diagnostics["unfinished"] is True
    assert result.diagnostics["failure"] == "zero_likelihood_support"
    assert np.all(result.weights == 0.0)


def test_beta_one_resample_move_is_target_preserving_and_explicit():
    proposal = ip.InformedProposal(_public_history(), pattern_seed=720907)
    result = proposal.importance_sample(128, seed=720908)
    moved = proposal.rejuvenate(result, seed=720909, sweeps=8, scales=(1.0, 0.5, 1.8))
    assert moved.diagnostics["rejuvenated"] is True
    assert moved.diagnostics["mh_sweeps"] == 8
    assert moved.diagnostics["mh_scales"] == [1.0, 0.5, 1.8]
    assert moved.diagnostics["resampled"] is True
    assert np.allclose(moved.weights, 1.0 / 128.0)
    assert moved.vectors.shape == (128, ip.CONTINUOUS_DIM)
    assert 0.0 <= moved.diagnostics["mh_acceptance"] <= 1.0


def test_unused_detect_dimensions_stay_prior_only():
    proposal = ip.InformedProposal(_public_history(), pattern_seed=720907)
    for component in proposal.components:
        assert np.allclose(component.mean[ip.DETECT], 0.0)
        assert np.allclose(component.covariance[ip.DETECT, ip.DETECT], np.eye(2))
        assert np.allclose(component.covariance[ip.DETECT, ip.IDENTITY], 0.0)
    assert "world" not in inspect.signature(ip.InformedProposal).parameters
    assert "audit" not in inspect.signature(ip.InformedProposal).parameters


def test_rejuvenation_log_weights_preserve_zero_entries():
    proposal = ip.InformedProposal(_public_history(), pattern_seed=720907)
    result = proposal.importance_sample(128, seed=720908)
    weights = np.full(128, 1.0 / 127.0)
    weights[-1] = 0.0
    sparse = ip.ImportanceResult(
        result.vectors,
        result.modes,
        result.worlds,
        result.log_weights,
        weights,
        dict(result.diagnostics),
    )
    moved = proposal.rejuvenate(sparse, seed=720909, sweeps=0)
    assert np.allclose(moved.weights[:-1], 1.0 / 127.0)
    assert moved.weights[-1] == 0.0
    assert np.all(np.isfinite(moved.log_weights[:-1]))
    assert np.isneginf(moved.log_weights[-1])


def test_duplicate_public_frame_id_is_rejected():
    history = _public_history()
    with pytest.raises(ValueError, match="duplicate public frame_id"):
        ip.InformedProposal(history + [history[0]])


def test_log_q_rejects_invalid_mode_support():
    proposal = ip.InformedProposal(_public_history(), pattern_seed=720907)
    vectors, modes, _worlds = proposal.sample(1, seed=720908)
    invalid = modes.copy()
    invalid[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match=r"mode entries must be \+/-1"):
        proposal.log_q(vectors, invalid)

