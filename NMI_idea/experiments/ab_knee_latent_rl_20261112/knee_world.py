"""Physical-time synthetic knee motion and public acquisition geometry."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DYNAMIC = HERE.parent / "ab_leg_dynamic_20261112"
sys.path[:0] = [str(DYNAMIC), str(HERE.parent / "ab_fisher_continuation_20261112")]
from dynamic_scan import DynamicScan
from certification import Edges


GEOMETRY = DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
WIDTH = 25.
DEPTH = (3., 52.)
TRAVEL_SPEED_MM_S = 90.
CAPTURE_S = .21
WAIT_S = .42


@dataclass(frozen=True)
class Frame:
    t: float
    q: float
    velocity: float
    site: int
    view: int
    centroids: np.ndarray
    labels: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class Candidate:
    site: int
    view: int
    duration: float
    distance_mm: float
    t: float
    kind: str


@dataclass(frozen=True)
class KneeCase:
    seed: int
    w_subject_mm: np.ndarray
    branch_offset_mm: float
    phase: float
    amplitude: float
    omega: float
    z0: float
    tau: float
    coupling: float
    drift: float
    drift_phase: float
    pose_bias: float
    dropout: float
    clutter: float
    ghost_label: int

    def true_angle_rate(self, t):
        theta = self.omega * t + self.phase
        return 15. + self.amplitude * np.sin(theta), self.amplitude * self.omega * np.cos(theta)

    def z_mm(self, t):
        # Analytic response to knee angular velocity plus an unobserved
        # oscillating disturbance. No action-count dependence.
        lam = 1. / self.tau
        om = self.omega
        second = 1.31 * om
        scale = self.coupling * self.amplitude * om / (lam**2 + om**2)
        drift_scale = self.drift / (lam**2 + second**2)

        def particular(time):
            phase = om * time + self.phase
            drift_phase = second * time + self.drift_phase
            return (scale * (lam * np.cos(phase) + om * np.sin(phase))
                    + drift_scale * (lam * np.cos(drift_phase) + second * np.sin(drift_phase)))

        return float(particular(t) + (self.z0 - particular(0.)) * np.exp(-lam * t))


class KneeWorld:
    """The repaired leg atlas, three moving knots and masked contact candidates."""

    def __init__(self, geometry=GEOMETRY, *, noisy_pose=True):
        self.geometry_path = Path(geometry)
        with np.load(self.geometry_path, allow_pickle=False) as data:
            self.scan = DynamicScan(data)
        self.angles = np.asarray(self.scan.angles)
        self.knots = tuple(int(i) for i in sorted(set(self.scan.moving)))
        self.knot_angles = self.angles[list(self.knots)]
        self.n = self.site_count = int(self.scan.n)
        self.n_views = self.scan.views.shape[2]
        self.aperture_mm = WIDTH
        self.noisy_pose = noisy_pose
        self.branch_sites = tuple(self.scan.branch_indices)
        self.fork_site = self.scan.fork
        self._site_profile = np.asarray(self.scan.feature)
        self.query_sites = tuple((site, 0) for site in range(self.fork_site - 1, self.fork_site + 6))
        self.query_sites += tuple((site, 1) for site in self.branch_sites)
        self.allowed = np.all(
            self.scan.eligible[list(self.knots)] & (self.scan.dird[list(self.knots)] >= 0.),
            axis=0,
        )
        self.edge = Edges(self.scan)
        self.edge_cache = {}
        self.base = np.stack((self.scan.trunk, self.scan.daughter), axis=2) * 1000.
        lateral = self.scan.frames[..., 1]
        self.design = np.zeros((len(self.angles), self.n, 2, 3, 3))
        self.design[:, :, :, :, 0] = lateral[:, :, None, :]
        self.design[:, :, 0, :, 2] = lateral * self._site_profile[None, :, None]
        self.design[:, :, 1, :, 2] = .68 * self.design[:, :, 0, :, 2]
        for i, site in enumerate(self.branch_sites):
            self.design[:, site, 1, :, 1] = (.48 + .26 * i) * lateral[:, site]
        self.view_bank = [self._view_order(i) for i in range(self.n)]
        self.motion_ratio = {}
        for skill in ("advance", "sweep_left", "sweep_right"):
            reference = self.scan.primitive.path(np.zeros(2), np.array([1., 0.]), skill=skill, samples=21)
            self.motion_ratio[skill] = float(np.linalg.norm(np.diff(reference, axis=0), axis=1).sum())
        self.initial_route = self._route()

    def _view_order(self, site):
        valid = np.flatnonzero(self.allowed[site])
        return tuple(sorted(map(int, valid), key=lambda v: (
            abs(self.scan.offsets[v]), v != self.scan.central, v
        )))

    def _route(self):
        result = []
        # Traverse the full material interval and finish near the junction.
        for site in range(self.n - 1, -1, -1):
            view = next((v for v in self.view_bank[site]
                         if not result or self.edge_valid(result[-1], (site, v))), None)
            if view is None:
                raise RuntimeError(f"whole-leg sweep has no legal transition at station {site}")
            result.append((site, view))
        return tuple(result)

    def edge_valid(self, previous, goal):
        key = (previous, goal)
        if key not in self.edge_cache:
            self.edge_cache[key] = all(
                self.edge.allow((pose, *previous), (pose, *goal)) for pose in self.knots
            )
        return self.edge_cache[key]

    def case(self, seed, *, difficult=True):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9173]))
        s = self.scan.s
        correlation = np.exp(-np.abs(s[:, None] - s[None, :]) / 55.)
        offsets = rng.normal(0., 2.3) + rng.multivariate_normal(np.zeros(self.n), .7**2 * correlation)
        return KneeCase(
            int(seed), np.clip(offsets, -6.5, 6.5),
            float(np.clip(rng.normal(0., 3.2), -6., 6.)),
            float(rng.uniform(0., 2 * np.pi)), float(rng.uniform(2.6, 3.85)),
            float(2 * np.pi / rng.uniform(5.4, 9.0)), float(rng.normal(0., 1.7)),
            float(rng.uniform(.5, 1.25)), float(rng.uniform(.55, 1.18)),
            float(rng.uniform(.4, 1.4)), float(rng.uniform(0., 2 * np.pi)),
            float(rng.normal(0., .24)), float(rng.uniform(.10, .30) if difficult else .08),
            float(rng.uniform(.13, .36) if difficult else .08), int(rng.integers(2)),
        )

    def _pose_bracket(self, q):
        q = float(np.clip(q, self.knot_angles[0], self.knot_angles[-1]))
        j = int(np.searchsorted(self.knot_angles, q, side="right"))
        if j == len(self.knots):
            return self.knots[-1], self.knots[-1], 0.
        lower, upper = self.knots[j - 1], self.knots[j]
        return lower, upper, (q - self.angles[lower]) / (self.angles[upper] - self.angles[lower])

    def _blend(self, a, q):
        low, high, fraction = self._pose_bracket(q)
        return (1. - fraction) * a[low] + fraction * a[high]

    def probe_world(self, q, site, view):
        return 1000. * self._blend(self.scan.views[:, site, view], q)

    def world_design(self, q, sites):
        sites = np.asarray(tuple(sites), dtype=int)
        if sites.ndim == 1:
            sites = np.column_stack((sites, np.zeros_like(sites)))
        i, label = sites.T
        return self._blend(self.base[:, i, label], q), self._blend(self.design[:, i, label], q)

    def projection(self, q, site, view):
        labels = (0, 1) if self.identity_site(site) else (0,)
        i = np.arange(len(labels))
        base = self._blend(self.base[:, site, i], q)
        design = self._blend(self.design[:, site, i], q)
        pose = self._blend(self.scan.view_frames[:, site, view], q)
        lateral = pose[:, 1] / np.linalg.norm(pose[:, 1])
        inward = pose[:, 2] - lateral * (pose[:, 2] @ lateral)
        inward /= np.linalg.norm(inward)
        project = np.stack((lateral, inward))
        return (base - self.probe_world(q, site, view)) @ project.T, np.einsum("ij,ljk->lik", project, design)

    def detection_probability(self, q, site, view):
        # Public imaging calibration. True case dropout is randomized.
        return .80 if self.identity_site(site) else .84

    def clutter_probability(self, site, view):
        return .21 if self.identity_site(site) else .12

    def identity_site(self, site):
        return site in self.branch_sites

    def true_z(self, case, t):
        return case.z_mm(t)

    def truth_world(self, case, t, sites):
        sites = tuple(sites)
        q, _ = case.true_angle_rate(t)
        base, design = self.world_design(q, sites)
        params = np.array([[case.w_subject_mm[i], case.branch_offset_mm, case.z_mm(t)] for i, _ in sites])
        return base + np.einsum("nci,ni->nc", design, params)

    def observe(self, case, action, site=None, view=None):
        if isinstance(action, Candidate):
            t, site, view = action.t, action.site, action.view
        else:
            t = float(action)
        tick = round(t * 1000)
        rng = np.random.default_rng(np.random.SeedSequence([case.seed, 141, tick, site, view]))
        pose_rng = np.random.default_rng(np.random.SeedSequence([case.seed, 147, tick]))
        q, speed = case.true_angle_rate(t)
        reading_q = q + (case.pose_bias + pose_rng.normal(0., .10) if self.noisy_pose else 0.)
        reading_v = speed + (pose_rng.normal(0., .15) if self.noisy_pose else 0.)
        base, rows = self.projection(q, site, view)
        offset = np.array([case.w_subject_mm[site], case.branch_offset_mm, case.z_mm(t)])
        expected = base + rows @ offset
        values, labels = [], []
        pd = self.detection_probability(q, site, view) + .15 - case.dropout
        for label, center in enumerate(expected):
            if rng.random() >= pd:
                continue
            scale = 3.0 if rng.random() < .075 else 1.
            observed = center + rng.normal(0., [1.0 * scale, .68])
            if abs(observed[0]) <= WIDTH and DEPTH[0] <= observed[1] <= DEPTH[1]:
                values.append(observed)
                labels.append(label)
        if rng.random() < case.clutter * (1. if self.identity_site(site) else .5):
            if self.identity_site(site) and rng.random() < .78:
                # A false acoustic structure near a plausible branch, with no
                # identity given to the policy. Tilt and site move its projection.
                center = base[case.ghost_label] + rng.normal(0., [2.9, 1.35])
                center = np.clip(center, [-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])
            else:
                center = rng.uniform([-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])
            values.append(center)
            labels.append(-1)
        if values:
            order = np.argsort(np.asarray(values)[:, 0], kind="stable")
            values = np.asarray(values)[order]
            labels = tuple(labels[i] for i in order)
        else:
            values, labels = np.zeros((0, 2)), ()
        return Frame(float(t), float(reading_q), float(reading_v), site, view, values, labels)

    def make_action(self, q, previous, goal, kind, *, wait=0.):
        site, view = goal
        distance = np.linalg.norm(self.probe_world(q, site, view)
                                  - self.probe_world(q, previous.site, previous.view))
        skill = "advance" if previous.site != site else (
            "sweep_left" if self.scan.offsets[view] < self.scan.offsets[previous.view] else "sweep_right"
        )
        distance = float(distance * self.motion_ratio[skill])
        duration = CAPTURE_S + distance / TRAVEL_SPEED_MM_S + wait
        return Candidate(site, view, duration, distance, previous.t + duration, kind)

    def candidates(self, reading, previous):
        prior = (previous.site, previous.view)
        fork = self.fork_site
        views = [view for view in self.view_bank[fork] if view != self.view_bank[fork][0]]
        alternative = min(views, key=lambda v: abs(abs(self.scan.offsets[v]) - 8.))
        goals = (
            (prior, "repeat", 0.),
            ((fork, self.view_bank[fork][0]), "fork", 0.),
            ((fork, alternative), "alternate window", 0.),
            ((self.branch_sites[1], self.view_bank[self.branch_sites[1]][0]), "follow junction", 0.),
            ((self.branch_sites[-1], self.view_bank[self.branch_sites[-1]][0]), "follow branch", 0.),
            (prior, "wait and observe", WAIT_S),
        )
        result, used = [], set()
        for goal, kind, wait in goals:
            if (goal, wait) not in used and self.edge_valid(prior, goal):
                result.append(self.make_action(reading.q, previous, goal, kind, wait=wait))
            else:
                result.append(None)
            used.add((goal, wait))
        return tuple(result)

    def initial(self, case):
        frames = []
        for goal in self.initial_route:
            if not frames:
                action = Candidate(*goal, CAPTURE_S, 0., CAPTURE_S, "initial sweep")
            else:
                action = self.make_action(frames[-1].q, frames[-1], goal, "initial sweep")
            frames.append(self.observe(case, action))
        return tuple(frames)


World = KneeWorld
