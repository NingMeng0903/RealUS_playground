"""Checks of the joint sensor kernel and public option control semantics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numba import njit

import control
import run


def check_complement():
    # Exact two-view common-offset fixture. This is a software regression,
    # independent of the robot benchmark and its results.
    bias = np.array([-1, -1, 1, 1])
    displacement = np.array([-1, 1, -1, 1])
    observation = np.zeros((7, 5, 4, 4))
    observation[:, (0, 3, 4), 0] = 1.
    for h, (b, d) in enumerate(zip(bias, displacement)):
        observation[:, 1, 2 + (b + d) // 2, h] = 1.
        observation[:, 2, 2 + (b - d) // 2, h] = 1.
    site, tilt = np.array([5, 5, 5, 4, 3]), np.array([0, 1, -1, 0, 0])
    costs = np.zeros((5, 5, 4))
    costs[..., 3] = 1.
    target = np.array([4, 3])
    potential, needed = control.nominal_potential(site, costs, target, 6, .5)
    model = (observation, np.zeros_like(observation), np.arange(5), site, tilt,
             np.zeros(5, dtype=int), costs, target, potential, needed,
             displacement.astype(float), 5)
    prior = np.full(4, .25)
    planned = np.array([1, 3, control.RULE_GEOMETRY, -1, -1])
    budget = np.array([10., 10., 10., 6.])
    assert control.option_action(model, prior, planned, 0, 0, 0, 0, budget, 0, 0) == (1, control.OPEN)
    b1, _ = control.posterior(prior, observation, np.zeros_like(observation), 1, 1, 2)
    assert b1[displacement > 0].sum() == .5
    repeated, _ = control.posterior(b1, observation, np.zeros_like(observation), 2, 1, 2)
    assert repeated[displacement > 0].sum() == .5
    assert control.option_action(model, b1, planned, 1, 1, 1, 0, budget, 2, 1) == (2, control.COMPLEMENT)
    b2, _ = control.posterior(b1, observation, np.zeros_like(observation), 2, 2, 1)
    assert b2[displacement > 0].sum() == 1.
    assert control.option_action(model, b2, planned, 2, 2, 2, 0, budget, 1, 2) == (-1, control.HANDOFF)
    assert control.option_action(model, prior, planned, 1, 1, 1, 0, budget, 0, 1) == (0, control.RECOVER)


@njit
def sample(detect, cdf, address, t, h, tapes, n_obs):
    p, s = np.zeros(n_obs), np.zeros(n_obs)
    for noise in tapes:
        obs, kind = control.capture_draw(detect, cdf, t, address, h, noise)
        p[obs] += 1
        s[obs] += kind == 0
    return p / len(tapes), s / len(tapes)


def check_kernel(path, repo):
    run.initialize(repo, path, 16, 6, "joint")
    rng = np.random.default_rng(161)
    for task, data in run.DATA.items():
        scene = data["scene"]
        t, h = 2, 3
        node = next(j for j in range(len(scene.site)) if scene.site[j] == scene.targets[0] and scene.tilt[j] == 1)
        addr = scene.to_address[node]
        tapes = rng.random((40000, t + 1, 7))
        p, s = sample(scene.detect, scene.cdf, addr, t, h, tapes, scene.likelihood.shape[2])
        discrepancy = max(np.max(abs(p - scene.likelihood[t, addr, :, h])),
                          np.max(abs(s - scene.correct[t, addr, :, h])))
        if discrepancy > .012:
            raise AssertionError(f"{task} simulated and integrated image kernel differ: {discrepancy}")
        print(f"{task}: integrated and sampled emission max difference {discrepancy:.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=run.HERE.parent / "RealUS_playground")
    args = parser.parse_args()
    check_complement()
    check_kernel(args.motion, args.repo)
    print("Complementary observations, early handoff, recovery and joint emissions passed.")


if __name__ == "__main__":
    main()
