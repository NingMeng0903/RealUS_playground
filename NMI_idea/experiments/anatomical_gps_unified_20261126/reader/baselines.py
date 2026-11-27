"""Reference and ablated readers for paired Gate 3 evaluation."""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy.special import logsumexp, softmax

from .belief_ipe import ROLE_COUNT
from .dynamic_factor_graph import (N_OUTPUT, DynamicFactorGraphReader, FrameEstimate, _IDENTITY_MARGINAL,
                                   _NeutralSkin, _identity_marginals, _score_associations, joint_pullback_covariance)
from .station_classifier import topological_station_probabilities


_SINGLETON_MASK_COLUMNS = [*(1 << role for role in range(ROLE_COUNT)), 0]


def _template(prior, h):
    graph = prior.graph
    out = np.full((ROLE_COUNT, 3), np.nan)
    for role in range(ROLE_COUNT):
        bracket = graph._station_query(h, role)
        if bracket is not None:
            a, b, t = bracket
            out[role] = (1. - t) * graph.knot_xyz_mm[a] + t * graph.knot_xyz_mm[b]
    return out


class B0_Rigid27:
    """Legacy uniform 27-particle rigid translation grid, fixed 2.35 mm noise.

    Anonymous nearest-component likelihood with a uniform clutter component;
    no tracker, nonrigid motion, physical cues, graph update, or discount.
    """

    name = 'B0_Rigid27'
    sigma_mm = 2.35
    offsets_mm = np.array(list(product((-10., 0., 10.), repeat=3)))

    def __init__(self, prior):
        self.prior = prior
        self.reset()

    def reset(self):
        self._log_weights = np.full(27, -np.log(27.))
        self._history = []

    def _estimate(self, obs, probe, update):
        h = float(np.clip((probe.origin_mm + 20. * probe.rotation[:, 2])[1],
                          self.prior.graph.h_mm.min(), self.prior.graph.h_mm.max()))
        height = np.clip(h - self.offsets_mm[:, 1], self.prior.graph.h_mm.min(), self.prior.graph.h_mm.max())
        templates = {k: _template(self.prior, k) for k in np.unique(height)}
        centers = np.stack([templates[k] for k in height]) + self.offsets_mm[:, None]
        plane = (centers - probe.origin_mm) @ probe.rotation
        image = plane[..., [0, 2]]
        visible = (np.isfinite(plane).all(axis=-1) & (np.abs(plane[..., 0]) < 25.)
                   & (np.abs(plane[..., 1]) < 5.) & (plane[..., 2] > 3.) & (plane[..., 2] < 80.))
        residual = obs.echoes[None, :, None, :2] - image[:, None, :, :]
        variance = self.sigma_mm**2
        geometry = -.5 * np.sum(residual**2, axis=-1) / variance - np.log(2. * np.pi * variance)
        visible_count = np.maximum(visible.sum(axis=-1), 1)
        score = np.full((27, len(obs.echoes), N_OUTPUT), np.log(.13 / (50. * 77.)))
        score[..., :-1] = np.where(visible[:, None, :], geometry + np.log(.87 / visible_count)[:, None, None], -np.inf)
        norm = logsumexp(score, axis=-1, keepdims=True)
        if update:
            self._log_weights += norm[:, :, 0].sum(axis=-1)
            self._log_weights -= logsumexp(self._log_weights)
        probs = np.einsum('p,pir->ir', np.exp(self._log_weights), np.exp(score - norm))
        best = int(np.argmax(self._log_weights))
        identities = np.zeros((len(obs.echoes), 1 << ROLE_COUNT))
        identities[:, _SINGLETON_MASK_COLUMNS] = probs
        return FrameEstimate(probs, centers[best], image[best], np.broadcast_to(variance * np.eye(2), (6, 2, 2)),
                             topological_station_probabilities(self.prior, h, probs), h, np.empty(0),
                             np.broadcast_to(variance * np.eye(3), (6, 3, 3)), identities)

    def step(self, obs, probe_pose, f_contact, q_hat, Sigma_q, dt):
        self._history.append((obs, probe_pose))
        return self._estimate(obs, probe_pose, update=True)

    def relabel_history(self):
        return tuple(self._estimate(o, p, update=False) for o, p in self._history)


class B1_NoTopology_AbsXYZ:
    """Independent absolute XYZ filters, with tracker pullback for parity.

    Offset and covariance are separate for each role. No material chart,
    common field, branching Gaussian, or junction precision enters this update.
    Station scoring uses the same surrogate windows as the other readers.
    """

    name = 'B1_NoTopology_AbsXYZ'

    def __init__(self, prior, kinematics, *, caliber_mean_mm, caliber_std_mm, rho_corr, rho_novel):
        self.prior, self.kinematics = prior, kinematics
        self._skin = _NeutralSkin(kinematics)
        self.caliber_mean_mm, self.caliber_std_mm = caliber_mean_mm, caliber_std_mm
        self.rho_corr, self.rho_novel = rho_corr, rho_novel
        self._h_bounds = float(prior.graph.h_mm.min()), float(prior.graph.h_mm.max())
        self.reset()

    def reset(self):
        self._mean = np.zeros((ROLE_COUNT, 3))
        self._cov = np.repeat((18.**2 * np.eye(3))[None], ROLE_COUNT, axis=0)
        self._history = []
        self._last_view = None
        self._latest = None

    def _height(self, probe, q):
        return DynamicFactorGraphReader._height(self, probe, q)

    def _predict(self, h, probe, q, pose_cov):
        center = _template(self.prior, h)
        roles = np.flatnonzero(np.isfinite(center[:, 0]))
        world, rot, jq = self.kinematics.forward_pose_canonical(q, center[roles] + self._mean[roles])
        p = probe.rotation[:, [0, 2]].T
        tangent = rot[..., 1]
        plane = (world - probe.origin_mm) @ probe.rotation
        slope = tangent @ probe.rotation[:, 1]
        step = np.divide(-plane[:, 1], slope, out=np.zeros_like(slope), where=abs(slope) > .45)
        world += tangent * np.clip(step, -30., 30.)[:, None]
        image = (world - probe.origin_mm) @ p.T
        take = np.divide(tangent, slope[:, None], out=np.zeros_like(tangent), where=abs(slope[:, None]) > .45)
        transform = np.eye(3) - take[..., :, None] * probe.rotation[:, 1]
        h_world = transform @ rot
        H = p @ h_world
        jq = transform @ jq
        pose = p @ jq
        r = np.broadcast_to(np.diag([.8**2 + (probe.speed_mm_s * .06)**2 / 12., 1.]), (len(roles), 2, 2))
        effective = joint_pullback_covariance(r, pose, pose_cov)
        V = effective.copy()
        WC = h_world @ self._cov[roles] @ h_world.transpose(0, 2, 1) + jq @ pose_cov @ jq.transpose(0, 2, 1)
        for i, role in enumerate(roles):
            V[2*i:2*i+2, 2*i:2*i+2] += H[i] @ self._cov[role] @ H[i].T
        visible = np.where((abs(image[:, 0]) < 28.) & (image[:, 1] > 1.) & (image[:, 1] < 83.), .8, .03)
        return roles, world, image, H, effective, V, WC, visible

    def _condition_centroids(self, roles, H, effective, association, rho):
        for j, slot in enumerate(association.echoes):
            rows = slice(2*j, 2*j+2)
            transform = association.operator[rows]
            contributors = np.asarray(association.groups[slot], dtype=np.intp)
            local = np.searchsorted(roles, contributors)
            observation = transform.reshape(2, len(roles), 2).transpose(1, 0, 2)[local] @ H[local]
            prior = self._cov[contributors]
            pht = prior @ observation.transpose(0, 2, 1)
            # Every role sees the same innovation for this weighted centroid.
            # Discount tracker/sensor noise once; sum all independent PRE-update
            # role contributions before writing any of their marginal blocks.
            innovation = transform @ effective @ transform.T / rho + (observation @ pht).sum(axis=0)
            gain = np.linalg.solve(innovation, pht.transpose(2, 0, 1).reshape(2, -1)).T.reshape(-1, 3, 2)
            self._mean[contributors] += gain @ association.residual[rows]
            covariance = prior - gain @ pht.transpose(0, 2, 1)
            self._cov[contributors] = .5 * (covariance + covariance.transpose(0, 2, 1))

    def _estimate(self, obs, probe, force, q, cov, update):
        h = self._height(probe, q)
        roles, world, image, H, effective, V, WC, visible = self._predict(h, probe, q, cov)
        assoc = _score_associations(obs.echoes, roles, image, V, visible, 4.7 + probe.speed_mm_s * .06,
                                    force, probe.dwell_s, self.caliber_mean_mm, self.caliber_std_mm)
        identities = _identity_marginals([(a.score, a) for a in assoc], len(obs.echoes))
        probs = identities @ _IDENTITY_MARGINAL
        if update and len(assoc[0].echoes):
            novel = (self._last_view is None or probe.rotation[:, 2] @ self._last_view[0] <= np.cos(np.deg2rad(25.))
                     or abs(h - self._last_view[1]) >= 15. or abs(force - self._last_view[2]) >= .5
                     or probe.dwell_s >= .35 > self._last_view[3])
            rho = self.rho_novel if novel else self.rho_corr
            self._condition_centroids(roles, H, effective, assoc[0], rho)
            roles, world, image, _, _, V, WC, _ = self._predict(h, probe, q, cov)
        if update:
            self._last_view = (probe.rotation[:, 2], h, force, probe.dwell_s)
        points, scan = np.full((ROLE_COUNT, 3), np.nan), np.full((ROLE_COUNT, 2), np.nan)
        matrices, world_cov = np.full((ROLE_COUNT, 2, 2), np.nan), np.full((ROLE_COUNT, 3, 3), np.nan)
        points[roles], scan[roles], world_cov[roles] = world, image, WC
        for i, role in enumerate(roles):
            matrices[role] = V[2*i:2*i+2, 2*i:2*i+2]
        station = topological_station_probabilities(self.prior, h, probs, identity_probabilities=identities)
        return FrameEstimate(probs, points, scan, matrices, station,
                             h, np.empty(0), world_cov, identities)

    def step(self, obs, probe_pose, f_contact, q_hat, Sigma_q, dt):
        args = obs, probe_pose, f_contact, q_hat, Sigma_q
        self._history.append(args)
        self._latest = self._estimate(*args, update=True)
        return self._latest

    def relabel_history(self):
        return tuple(self._estimate(*args, update=False) for args in self._history)

    def get_belief_summary(self):
        if self._latest is None:
            return None
        f = self._latest
        prob = np.maximum(f.role_probabilities, np.finfo(float).tiny)
        entropy = float(-np.sum(prob * np.log(prob)))
        return dict(posterior_entropy=entropy, top2_branch_margin=1., branch_probabilities=np.ones(1),
                    centers_mm=f.centers_mm, centers_covariance_mm2=f.world_covariance_mm2, belief_ipe=f.ipe,
                    station_probabilities=f.station_probabilities, spatial_disagreement_mm2=np.zeros((6, 3, 3)))


class Full_AGPS_Reader(DynamicFactorGraphReader):
    name = 'Full_AGPS_Reader'


class B2_NoPullback(Full_AGPS_Reader):
    name = 'B2_NoPullback'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable_pullback=True, **kwargs)


class B3_NoDiscount(Full_AGPS_Reader):
    name = 'B3_NoDiscount'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, disable_discount=True, **kwargs)


class B4_NoBeliefIPE(DynamicFactorGraphReader):
    name = 'B4_NoBeliefIPE'


def make_readers(prior, kinematics, *, scorer=None, caliber_mean_mm=6., caliber_std_mm=3., rho_corr=.25, rho_novel=.85):
    params = dict(caliber_mean_mm=caliber_mean_mm, caliber_std_mm=caliber_std_mm, rho_corr=rho_corr, rho_novel=rho_novel)
    readers = (B0_Rigid27(prior), B1_NoTopology_AbsXYZ(prior, kinematics, **params),
               B2_NoPullback(prior, kinematics, scorer=scorer, **params),
               B3_NoDiscount(prior, kinematics, scorer=scorer, **params), B4_NoBeliefIPE(prior, kinematics, **params),
               Full_AGPS_Reader(prior, kinematics, scorer=scorer, **params))
    return {reader.name: reader for reader in readers}
