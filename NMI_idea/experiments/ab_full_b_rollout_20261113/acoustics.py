"""Shared joint acoustic model on the baked contact chart.

The synthetic detector reports the unlabeled centroid closest to the nominal
query and whether other returns were detected. Geometry and nuisance are drawn
once per episode. Per-capture
innovations are drawn separately and can be coupled across counterfactual
paths. The estimator integrates quantized centroid likelihoods over a
separate population of joint case hypotheses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr


WIDTH = 25.
DEPTH_MIN, DEPTH_MAX = 3., 52.
IMAGE_SD = np.array([1.00, .68])
OUTLIER_SD = np.array([3.00, .68])
OUTLIER = .075
X_BINS = np.array([-8., -4., -1.5, 1.5, 4., 8.])
Z_BINS = np.array([-5., -2., -.7, .7, 2., 5.])
NX, NZ = len(X_BINS) + 1, len(Z_BINS) + 1
PIXELS = NX * NZ
OBSERVATIONS = 1 + 2 * PIXELS
MAX_KINDS = 3
PRIOR_SEED = 51884473

X_REP = np.array([-10., -6., -2.75, 0., 2.75, 6., 10.])
Z_REP = np.array([-7., -3.5, -1.35, 0., 1.35, 3.5, 7.])
DISTANCE = ((X_REP[:, None] / 4.) ** 2 + (Z_REP[None, :] / 3.) ** 2).ravel()
FAR = (DISTANCE[None, :] > DISTANCE[:, None] + 1.e-12).astype(float)
TIE = (np.abs(DISTANCE[None, :] - DISTANCE[:, None]) <= 1.e-12).astype(float)


@dataclass(frozen=True)
class Scenario:
    body: object
    calibration_depth: float
    additional_loss_start: float
    additional_loss_end: float


def setup_world(repo):
    root = Path(repo) / "NMI_idea" / "experiments"
    model_dir = root / "ab_map_recheck_20261112"
    sys.path.insert(0, str(model_dir))
    from map_model import MapWorld
    import dynamic_scan
    import knee_world

    class AcousticOnlyEdges:
        def __init__(self, scan):
            self.q = scan.q

        def allow(self, start, goal):
            return True

    asset_path = root / "ab_tro_y_continuation_20261112/review_inputs/POSED_BRANCH.npz"
    with np.load(asset_path, allow_pickle=False) as asset:
        posed = (asset["dense"], asset["s"], asset["daughter"],
                 tuple(map(int, asset["sites"])), int(asset["fork"]))
    with (patch.object(dynamic_scan, "posed_branch", return_value=posed),
          patch.object(knee_world, "Edges", AcousticOnlyEdges)):
        return MapWorld(mode="stress")


def make_scenario(world, seed, mode="joint"):
    body = world.case(int(seed))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 180731]))
    calibration_depth = float(np.clip(rng.normal(0., 2.), -5., 5.))
    loss_start = float(rng.uniform(.75, 3.5))
    loss_end = loss_start + float(rng.uniform(.45, .95))
    if rng.random() >= .32:
        loss_start, loss_end = np.inf, -np.inf
    return Scenario(body, calibration_depth if mode != "zero_depth_bias" else 0., loss_start, loss_end)


def seed_for(namespace, index):
    return int(np.random.SeedSequence([int(namespace), int(index)]).generate_state(1)[0])


class Geometry:
    """Image projection at the same nominal skin pose used by the robot graph."""

    def __init__(self, world, nodes, target_label, dropout=True):
        self.world = world
        self.label = target_label
        self.dropout = dropout
        self.addresses = tuple(dict.fromkeys((int(n["site"]), int(n["view"])) for n in nodes))
        self.index = {point: i for i, point in enumerate(self.addresses)}
        self.node_to_address = np.array([self.index[int(n["site"]), int(n["view"])] for n in nodes])
        self.rows = {point: world.projection(15., *point) for point in self.addresses}
        self.tilt = {point: float(world.scan.tilts[point[1]]) if hasattr(world.scan, "tilts")
                     else (-12. if point[1] == 7 else 12. if point[1] == 8 else 0.)
                     for point in self.addresses}

    def parameters(self, scenario, clocks, mode="joint"):
        """Times and views map to physical centroid means and presence chances."""
        case = scenario.body
        clocks = np.asarray(clocks, dtype=float)
        n_time, n_address = len(clocks), len(self.addresses)
        mu = np.zeros((n_time, n_address, MAX_KINDS, 2))
        detect = np.zeros(mu.shape[:-1])
        nominal = np.zeros((n_address, 2))
        lag = np.array([case.z_mm(t) for t in clocks])
        old_clock = clocks + self.world.nominal_route_end_s
        archive_loss = (old_clock >= case._miss_start_s) & (old_clock <= case._miss_end_s)
        extra_loss = (clocks >= scenario.additional_loss_start) & (clocks <= scenario.additional_loss_end)
        for i, (site, view) in enumerate(self.addresses):
            base, design = self.rows[site, view]
            target = min(self.label, len(base) - 1)
            nominal[i] = base[target]
            params = np.column_stack((
                np.full(n_time, case.w_subject_mm[site]), np.full(n_time, case.branch_offset_mm),
                lag, np.full(n_time, case.w_local),
            ))
            centers = base[None] + np.einsum("lcf,tf->tlc", design, params)
            centers[..., 1] += scenario.calibration_depth
            pd = np.clip(self.world.detection_probability(15., site, view) + .15 - case.dropout, 0., 1.)
            mu[:, i, 0] = centers[:, target]
            detect[:, i, 0] = pd
            if self.dropout:
                detect[extra_loss | (archive_loss & (case._miss_label == target)), i, 0] = 0.
            if len(base) > 1:
                other = 1 - target
                mu[:, i, 1] = centers[:, other]
                detect[:, i, 1] = pd
                if self.dropout and case._miss_label == other:
                    detect[archive_loss, i, 1] = 0.
            ghost = self.world._persistent_clutter(case, 15., site, view, base)
            if ghost is None:
                # One weak return from a nearby structure is also possible in
                # a single-lumen image. Its material displacement persists.
                frame = self.world._blend(self.world.scan.view_frames[:, site, view], 15.)
                central = self.world._blend(self.world.scan.view_frames[:, site, 3], 15.)
                shift_world = central[:, 1] * case._ghost_lateral_mm + central[:, 2] * case._ghost_depth_mm
                ghost = base[min(case.ghost_label, len(base) - 1)] + shift_world @ frame[:, 1:3]
                ghost_prob = case.clutter if len(base) > 1 else .5 * case.clutter
            else:
                ghost_prob = .88
            mu[:, i, 2] = ghost + [0., scenario.calibration_depth]
            detect[:, i, 2] = ghost_prob
        # Gate the simulated returns by the image aperture before competition
        # for the dominant detected centroid. The distribution is conditional
        # on the same material state and geometry for every measurement.
        lower = np.array([-WIDTH, DEPTH_MIN])
        upper = np.array([WIDTH, DEPTH_MAX])
        mu_flat = mu.reshape(-1, 2)
        aperture = []
        for sd in (IMAGE_SD, OUTLIER_SD):
            aperture.append(np.prod(ndtr((upper - mu_flat) / sd) - ndtr((lower - mu_flat) / sd), axis=1))
        masses = ((1. - OUTLIER) * aperture[0] + OUTLIER * aperture[1]).reshape(detect.shape)
        return mu, detect * masses, nominal


def component_probabilities(means, nominal):
    """Distribution of the observed centroid bin conditional on its return."""
    lower, upper = np.array([-WIDTH, DEPTH_MIN]), np.array([WIDTH, DEPTH_MAX])
    # Shape: clock, acoustic view, return kind, axis bin.
    x = np.clip(nominal[:, 0, None] + X_BINS[None], lower[0], upper[0])
    z = np.clip(nominal[:, 1, None] + Z_BINS[None], lower[1], upper[1])
    x = np.concatenate((np.full((len(x), 1), lower[0]), x, np.full((len(x), 1), upper[0])), axis=1)
    z = np.concatenate((np.full((len(z), 1), lower[1]), z, np.full((len(z), 1), upper[1])), axis=1)
    combined = np.zeros((*means.shape[:-1], NX, NZ))
    for sd, fraction in ((IMAGE_SD, 1. - OUTLIER), (OUTLIER_SD, OUTLIER)):
        px = np.diff(ndtr((x[None, :, None] - means[..., 0, None]) / sd[0]), axis=-1)
        pz = np.diff(ndtr((z[None, :, None] - means[..., 1, None]) / sd[1]), axis=-1)
        combined += fraction * px[..., :, None] * pz[..., None, :]
    combined = combined.reshape(*means.shape[:-1], NX * NZ)
    normalizer = combined.sum(axis=-1, keepdims=True)
    return np.divide(combined, normalizer, out=np.full_like(combined, 1. / (NX * NZ)),
                     where=normalizer > 1.e-30)


def probabilities(geometry, scenario, clocks, mode):
    means, detect, nominal = geometry.parameters(scenario, clocks, mode)
    bins = component_probabilities(means, nominal)
    # Integrate which centroid is closest to the query, including randomized
    # ties. An observation includes whether multiple returns were available.
    far = (bins @ FAR.T) * detect[..., None]
    tie = (bins @ TIE.T) * detect[..., None]
    absent = (1. - detect)[..., None]
    p_return = bins * detect[..., None]
    single, multiple = np.empty_like(bins), np.empty_like(bins)
    for i in range(MAX_KINDS):
        j, k = [j for j in range(MAX_KINDS) if j != i]
        f, g, a, b = far[..., j, :], far[..., k, :], tie[..., j, :], tie[..., k, :]
        x, y = absent[..., j, :], absent[..., k, :]
        single[..., i, :] = p_return[..., i, :] * x * y
        multiple[..., i, :] = p_return[..., i, :] * (
            (f + .5 * a) * y + (g + .5 * b) * x + f * g + .5 * (a * g + b * f) + a * b / 3.
        )
    joints = np.concatenate((single, multiple), axis=-1)
    total = np.concatenate((np.prod(1. - detect, axis=-1)[..., None], joints.sum(axis=-2)), axis=-1)
    correct = np.concatenate((np.zeros_like(total[..., :1]), joints[..., 0, :]), axis=-1)
    return total, correct, detect, np.cumsum(bins, axis=-1)


def model_tables(geometry, clocks, size, mode):
    total, correct, detection, images = [], [], [], []
    for i in range(size):
        scenario = make_scenario(geometry.world, seed_for(PRIOR_SEED, i), mode)
        p, s, d, bins = probabilities(geometry, scenario, clocks, mode)
        total.append(p)
        correct.append(s)
        detection.append(d)
        images.append(bins)
    likelihood = np.ascontiguousarray(np.moveaxis(np.array(total), 0, -1))
    success = np.ascontiguousarray(np.moveaxis(np.array(correct), 0, -1))
    detect = np.ascontiguousarray(np.moveaxis(np.array(detection), 0, 2))
    cdf = np.ascontiguousarray(np.moveaxis(np.array(images), 0, 2))
    if not np.allclose(likelihood.sum(axis=-2), 1., atol=2.e-12):
        raise ValueError("image kernel does not normalize")
    return likelihood, success, detect, cdf


def tape(seed, n_times):
    """Common uniforms per physical clock, per return and image channel."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), 68401])).random((n_times, 7))
