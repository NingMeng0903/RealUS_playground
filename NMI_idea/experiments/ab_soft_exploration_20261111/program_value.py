"""Public Bayes updates and paired evaluation of fixed candidate programs."""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from toy_world import kernel, latent_index


@lru_cache(maxsize=None)
def observation_tables(action, fixture):
    transition, emission, target_emission = kernel(action, fixture)
    return transition, emission, target_emission, np.cumsum(transition, axis=1), np.cumsum(emission, axis=1)


def update(belief, fixture, action, obs):
    transition, emission, target, _, _ = observation_tables(action, fixture)
    predicted = np.asarray(belief, dtype=float) @ transition
    likelihood = predicted * emission[:, obs]
    evidence = likelihood.sum()
    if evidence <= 0.0:
        raise ValueError("observation has zero evidence under the public model")
    return likelihood / evidence, float(predicted @ target[:, obs] / evidence)


def accepted(action, obs, p_target):
    return action.dest in ("TL", "TR") and obs == 4 and p_target >= .90


def decision_prior(belief, mode):
    if mode == "soft":
        return belief
    if mode != "hard":
        raise ValueError(f"unknown selection mode {mode}")
    present = np.zeros(8, dtype=bool)
    for side in ("L", "R"):
        for permanent in (False, True):
            present[latent_index(side, permanent, True)] = True
    use_present = float(belief[present].sum()) >= .5
    hard = np.where(present == use_present, belief, 0.)
    return hard / hard.sum()


def program_scores(belief, fixture, programs, *, mode, start_draws, packet_draws):
    """Estimate probability of correct public completion before wrong stop.

    The selection distribution can condition on the MAP visibility mode.
    The verifier always updates the full public belief. Latent samples affect
    the renderer and reward only. Fixed draws are common to all programs.
    """
    belief = np.asarray(belief, dtype=float)
    sample_prior = decision_prior(belief, mode)
    start_states = np.searchsorted(np.cumsum(sample_prior), start_draws, side="right")
    start_states = np.minimum(start_states, 7)
    count = len(start_states)
    scores = np.zeros(len(programs), dtype=float)
    for index, program in enumerate(programs):
        if len(program) > packet_draws.shape[1]:
            raise ValueError("program exceeds shared packet draw table")
        public_beliefs = np.tile(belief, (count, 1))
        states = start_states.copy()
        alive = np.ones(count, dtype=bool)
        correct = np.zeros(count, dtype=bool)
        for step, action in enumerate(program):
            transition, emission, target, transition_cdf, emission_cdf = observation_tables(action, fixture)
            draws = packet_draws[:, step]
            states = np.minimum(np.sum(draws[:, 0, None] >= transition_cdf[states], axis=1), 7)
            obs = np.minimum(np.sum(draws[:, 1, None] >= emission_cdf[states], axis=1), 4)
            is_target = draws[:, 2] * emission[states, obs] < target[states, obs]

            predicted = public_beliefs @ transition
            joint = predicted * emission.T[obs]
            evidence = joint.sum(axis=1)
            if np.any(evidence <= 0.):
                raise ValueError("simulated packet has zero public evidence")
            public_beliefs = joint / evidence[:, None]
            if action.dest in ("TL", "TR"):
                p_target = (predicted * target.T[obs]).sum(axis=1) / evidence
                stopped = alive & (obs == 4) & (p_target >= .90)
                correct |= stopped & is_target
                alive &= ~stopped
                if not alive.any():
                    break
        scores[index] = correct.mean()
    return scores
