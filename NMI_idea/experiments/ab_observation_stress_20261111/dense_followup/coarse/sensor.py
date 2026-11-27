"""Categorical observation channel for the A/B stress experiments.

The channel is deliberately a finite public observation model.  A caller may
use it with model particles to predict a branch, or with a separate private
world to generate an evaluator distribution.  ``joint`` contains latent
source and detector mode axes, but those axes are not public policy inputs.

One call represents one observation block and one latent source.  Repeated
frames from a block therefore must not be counted as five independent
likelihood factors.  Temporal persistence of detector quality is represented
by the exported HMM transition matrices and is handled by the experiment
runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
from scipy.special import ndtr, expit


# Keep the existing geometric renderer as the only geometry implementation.
_ENV_ROOT = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_module_ladder_20261110")
if str(_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENV_ROOT))
from observation import environment as env


MODE_NAMES = ("good", "bad")
SOURCE_NAMES = ("target", "decoy", "clutter", "missing")

# The first column is good and the second column is bad.  Every model has the
# same stationary bad marginal 0.3; only persistence differs.
TRANSITIONS = {
    "clean": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
    "iid": np.array([[0.7, 0.3], [0.7, 0.3]], dtype=float),
    "bursty": np.array(
        [[1.0 - 0.3 * 0.2 / 0.7, 0.3 * 0.2 / 0.7], [0.2, 0.8]],
        dtype=float,
    ),
}
INITIAL_MODE = {
    "clean": np.array([1.0, 0.0], dtype=float),
    "iid": np.array([0.7, 0.3], dtype=float),
    "bursty": np.array([0.7, 0.3], dtype=float),
}


U_EDGES = np.array([-np.inf, -0.010, -0.002, 0.002, 0.010, np.inf])
DEPTH_EDGES = np.array([-np.inf, 0.032, 0.041, np.inf])
LENGTH_THRESHOLD = 0.015
IDENTITY_THRESHOLD = 0.75

# Finite display representatives only.  The category itself is an interval,
# not a precise measurement.  The outer bins are represented inside the
# renderer's public image FOV to avoid fabricating values at infinity.
U_REPRESENTATIVES = np.array([-0.0175, -0.006, 0.0, 0.006, 0.0175])
DEPTH_REPRESENTATIVES = np.array([0.0235, 0.0365, 0.0555])
LENGTH_REPRESENTATIVES = (0.008, 0.017)
IDENTITY_REPRESENTATIVES = (0.65, 0.82)


def _make_outcomes() -> List[dict]:
    outcomes: List[dict] = [
        {
            "outcome": 0,
            "visible": False,
            "u_bin": None,
            "depth_bin": None,
            "length_pass": None,
            "identity_pass": None,
            "image_u_m": None,
            "image_depth_m": None,
            "visible_length_m": None,
            "image_axis_rad": None,
            "identity_score": None,
            "contact": None,
        }
    ]
    index = 1
    for u_bin in range(5):
        for depth_bin in range(3):
            for length_pass in range(2):
                for identity_pass in range(2):
                    outcomes.append(
                        {
                            "outcome": index,
                            "visible": True,
                            "u_bin": u_bin,
                            "depth_bin": depth_bin,
                            "length_pass": bool(length_pass),
                            "identity_pass": bool(identity_pass),
                            "image_u_m": float(U_REPRESENTATIVES[u_bin]),
                            "image_depth_m": float(DEPTH_REPRESENTATIVES[depth_bin]),
                            "visible_length_m": float(
                                LENGTH_REPRESENTATIVES[length_pass]
                            ),
                            "image_axis_rad": None,
                            "identity_score": float(
                                IDENTITY_REPRESENTATIVES[identity_pass]
                            ),
                            # Contact is known from the robot/pose interface,
                            # not inferred from the categorical sensor code.
                            "contact": None,
                        }
                    )
                    index += 1
    return outcomes


OUTCOMES = _make_outcomes()
PUBLIC_GATE = np.array(
    [
        bool(
            o["visible"]
            and o["u_bin"] == 2
            and o["length_pass"]
            and o["identity_pass"]
        )
        for o in OUTCOMES
    ],
    dtype=bool,
)


@dataclass(frozen=True)
class ObservationChannel:
    """Finite observation distribution and evaluator-only correctness mass."""

    # [world, mode(good/bad), action, source(target/decoy/clutter/missing), outcome]
    joint: np.ndarray
    # [world, mode, action, outcome], source marginalized for policy evidence
    likelihood: np.ndarray
    # [world, mode, action, outcome], target mass passing true geometry and gate
    correct_joint: np.ndarray
    # [world, action], geometric target segment exists before stochastic returns
    geometric_target_visible: np.ndarray


def _interval_probability(mean: np.ndarray, std: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Gaussian probability of each interval, retaining a finite category axis."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("categorical feature standard deviation must be positive")
    values = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        upper = np.ones_like(mean) if np.isposinf(hi) else ndtr((hi - mean) / std)
        lower = np.zeros_like(mean) if np.isneginf(lo) else ndtr((lo - mean) / std)
        # Subtraction can make a tiny negative number at a far tail.  This is
        # numerical protection only, not an artificial likelihood floor.
        values.append(np.maximum(0.0, upper - lower))
    return np.stack(values, axis=-1)


def _conditional_bins(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return [world, action, 60] product probabilities for one source."""
    pu = _interval_probability(mean[..., 0], std[..., 0], U_EDGES)
    pd = _interval_probability(mean[..., 1], std[..., 1], DEPTH_EDGES)
    pl_pass = ndtr((mean[..., 2] - LENGTH_THRESHOLD) / std[..., 2])
    pl = np.stack([1.0 - pl_pass, pl_pass], axis=-1)
    pi_pass = ndtr((mean[..., 4] - IDENTITY_THRESHOLD) / std[..., 4])
    pi = np.stack([1.0 - pi_pass, pi_pass], axis=-1)
    return (
        pu[..., :, None, None, None]
        * pd[..., None, :, None, None]
        * pl[..., None, None, :, None]
        * pi[..., None, None, None, :]
    ).reshape(*mean.shape[:-1], 60)


def _clutter_bins(shape: Sequence[int]) -> np.ndarray:
    """Fixed clutter feature distribution with support in every public bin."""
    # The clutter family has substantial overlap with real observations, while
    # all tail bins retain support because its standard deviations are finite.
    mean = np.zeros(tuple(shape) + (5,), dtype=float)
    mean[..., 0] = 0.0
    mean[..., 1] = 0.035
    mean[..., 2] = 0.0155
    mean[..., 3] = 0.0
    mean[..., 4] = 0.78
    std = np.zeros_like(mean)
    std[..., 0] = 0.009
    std[..., 1] = 0.008
    std[..., 2] = 0.005
    # Axis is not encoded but keeping a finite nuisance scale makes the
    # categorical construction explicit and future-proof.
    std[..., 3] = 0.10
    std[..., 4] = 0.15
    return _conditional_bins(mean, std)


def _as_world_list(worlds) -> list:
    if isinstance(worlds, (list, tuple)):
        worlds = list(worlds)
    else:
        worlds = list(worlds)
    if not worlds:
        raise ValueError("worlds must be nonempty")
    return worlds


def channel(worlds, poses, identity_blind: bool = False) -> ObservationChannel:
    """Build the categorical error channel for public model worlds.

    ``worlds`` are model particles or separately held evaluator worlds.  The
    function returns latent audit axes for testing, but a policy must consume
    only ``likelihood`` and the public fields in :data:`OUTCOMES`.  The same
    function intentionally supports private worlds so evaluator distributions
    can be generated without passing truth or source labels to a controller.
    """
    worlds = _as_world_list(worlds)
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses must have shape [A,3]")

    pred = env.predict_batch(worlds, poses)
    means = np.asarray(pred["means"], dtype=float)  # [W,A,2,5]
    p_source = np.asarray(pred["p_source"], dtype=float)  # [W,A,2]
    W, A, S, F = means.shape
    if S != 2 or F != 5:
        raise ValueError("environment prediction must expose two sources and five fields")

    # Transpose to the public contract [W, mode, A, source, outcome].
    joint = np.zeros((W, 2, A, 4, len(OUTCOMES)), dtype=float)
    clutter_prob = np.zeros((W, A, 2, len(OUTCOMES) - 1), dtype=float)
    clutter_prob[...] = _clutter_bins((W, A, 2))

    # Original renderer's nominal feature noise and the specified degraded
    # channel.  Angle is not emitted, but its nominal column is retained only
    # to keep field indexing explicit.
    good_std = np.broadcast_to(env.SDS, (W, A, 2, 5)).copy()
    bad_std = good_std.copy()
    bad_std[..., :3] *= 2.5
    bad_std[..., 4] = 0.20

    typical_id = np.array([expit(1.75), expit(0.75)], dtype=float)
    good_means = means.copy()
    source_base_id = np.empty_like(means[..., 4])
    source_base_id[..., 0] = 0.74 + 0.3 * (means[..., 0, 4] - typical_id[0])
    source_base_id[..., 1] = 0.75 + 0.3 * (means[..., 1, 4] - typical_id[1])

    if identity_blind:
        # The blind control intentionally removes source identity and detector
        # strength differences.  Geometry can still make target/decoy masses
        # different, while swapping source labels leaves the public mixture
        # invariant.
        good_mult = np.ones((W, A, 2), dtype=float) * 1.0
        bad_mult = np.ones((W, A, 2), dtype=float) * ((0.30 + 0.48) / 2.0)
        blind_id_mean = np.full((W, A, 2), 0.76, dtype=float)
        good_means[..., 0, 4] = blind_id_mean[..., 0]
        good_means[..., 1, 4] = blind_id_mean[..., 1]
        source_base_id = blind_id_mean
    else:
        good_mult = np.ones((W, A, 2), dtype=float)
        bad_mult = np.broadcast_to(np.array([0.30, 0.48]), (W, A, 2)).copy()

    for mode in range(2):
        if mode == 0:
            returns = p_source * good_mult
            source_bins = [
                _conditional_bins(good_means[..., s, :], good_std[..., s, :])
                for s in range(2)
            ]
            clutter_rate = 0.005
        else:
            returns = p_source * bad_mult
            bad_means = means.copy()
            bad_means[..., 0, 4] = source_base_id[..., 0]
            bad_means[..., 1, 4] = source_base_id[..., 1]
            source_bins = [
                _conditional_bins(bad_means[..., s, :], bad_std[..., s, :])
                for s in range(2)
            ]
            clutter_rate = 0.25

        # The detector first chooses a real source, clutter, or no return.
        # The source masses are mutually exclusive by construction.
        real_mass = returns.sum(axis=-1)
        clutter_mass = np.maximum(0.0, 1.0 - real_mass) * clutter_rate
        miss_mass = np.maximum(0.0, 1.0 - real_mass - clutter_mass)

        for source in range(2):
            # [W,A,60] -> [W,A,4,61], with missing slot left zero.
            joint[:, mode, :, source, 1:] = returns[..., source, None] * source_bins[source]
        joint[:, mode, :, 2, 1:] = clutter_mass[..., None] * clutter_prob[..., mode, :]
        joint[:, mode, :, 3, 0] = miss_mass

    likelihood = joint.sum(axis=3)

    # True target geometry is evaluator-only.  In particular, this is not
    # derived from a sampled detector return and it remains independent of the
    # identity-blind public mixture.
    # These fields are already returned by the legacy public prediction API;
    # using them avoids a second geometry implementation in this channel.
    geometric_length = np.asarray(pred["geometric_length"], dtype=float)
    angle_audit = np.asarray(pred["angle_audit"], dtype=float)
    target_mean = means[..., 0, :]
    contact = np.asarray(pred["contact"], dtype=bool)
    target_quality = (
        contact
        & (geometric_length[..., 0] >= LENGTH_THRESHOLD)
        & (np.abs(target_mean[..., 0]) <= 0.002)
        & (angle_audit[..., 0] <= np.deg2rad(10.0))
    )
    geometric_target_visible = geometric_length[..., 0] > 0.0005

    correct_joint = joint[:, :, :, 0, :] * target_quality[:, None, :, None]
    correct_joint *= PUBLIC_GATE[None, None, None, :]

    # Explicit finite-category normalization is an invariant and catches any
    # accidental source overlap or numerical mass loss.
    total = joint.sum(axis=(3, 4))
    if not np.all(np.isfinite(joint)) or np.any(joint < -1e-12):
        raise FloatingPointError("observation channel produced invalid probability mass")
    if not np.allclose(total, 1.0, atol=2e-12, rtol=2e-12):
        raise FloatingPointError("observation channel mass does not normalize")

    return ObservationChannel(
        joint=joint,
        likelihood=likelihood,
        correct_joint=correct_joint,
        geometric_target_visible=geometric_target_visible,
    )


__all__ = [
    "ObservationChannel",
    "MODE_NAMES",
    "SOURCE_NAMES",
    "TRANSITIONS",
    "INITIAL_MODE",
    "U_EDGES",
    "DEPTH_EDGES",
    "LENGTH_THRESHOLD",
    "IDENTITY_THRESHOLD",
    "OUTCOMES",
    "PUBLIC_GATE",
    "channel",
]
