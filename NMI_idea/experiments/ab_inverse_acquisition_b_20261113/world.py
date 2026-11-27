"""Continuous tilt interpolation of the pinned synthetic observation kernel.

This does not introduce angle-dependent echogenicity or a measured acoustic
model. Legacy angles retain exactly their original kernel. Off-grid views
rotate the same persistent latent geometry. Ghost visibility is interpolated
between existing endpoint regimes, never redrawn from a new view identifier.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
from scipy.spatial.transform import Rotation


LEGACY = "ab_full_b_rollout_20261113"
ANGLES = {3: 0., 7: -12., 8: 12.}


def load(repo):
    path = Path(repo).resolve() / "NMI_idea" / "experiments" / LEGACY
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import acoustics
    import control
    import run
    return acoustics, control, run


def aperture(acoustics, means):
    lower = np.array([-acoustics.WIDTH, acoustics.DEPTH_MIN])
    upper = np.array([acoustics.WIDTH, acoustics.DEPTH_MAX])
    result = np.zeros(means.shape[:-1])
    for sd, fraction in ((acoustics.IMAGE_SD, 1. - acoustics.OUTLIER),
                         (acoustics.OUTLIER_SD, acoustics.OUTLIER)):
        result += fraction * np.prod(ndtr((upper - means) / sd) - ndtr((lower - means) / sd), axis=-1)
    return result


class Geometry:
    def __init__(self, acoustics, world, nodes, target_label):
        self.acoustics, self.world, self.label = acoustics, world, target_label
        self.nodes = nodes
        self.addresses = tuple(dict.fromkeys((int(n["site"]), int(n["view"])) for n in nodes))
        self.index = {point: i for i, point in enumerate(self.addresses)}
        self.node_to_address = np.array([self.index[int(n["site"]), int(n["view"])] for n in nodes])
        self.angles = {(int(n["site"]), int(n["view"])):
                       float(n.get("tilt_deg", ANGLES.get(int(n["view"]), float("nan")))) for n in nodes}
        if not all(np.isfinite(list(self.angles.values()))):
            raise ValueError("every synthetic view needs a physical angle")
        if any(abs(v) > 12. + 1e-6 for v in self.angles.values()):
            raise ValueError("this experiment is interpolation within +/-12 degrees")
        sites = sorted({p[0] for p in self.addresses})
        anchors = [dict(site=s, view=v) for s in sites for v in (3, 7, 8)]
        self.legacy = acoustics.Geometry(world, anchors, target_label)
        self.frames = {}
        self.nominal = {}
        self.transforms = {}
        for site, view in self.addresses:
            center = world._blend(world.scan.view_frames[:, site, 3], 15.)
            if view in ANGLES:
                frame = world._blend(world.scan.view_frames[:, site, view], 15.)
            else:
                frame = center @ Rotation.from_rotvec([np.deg2rad(self.angles[site, view]), 0., 0.]).as_matrix()
            self.frames[site, view] = frame
            self.transforms[site, view] = center[:, 1:3].T @ frame[:, 1:3]

    def parameters(self, scenario, clocks, mode="joint"):
        a = self.acoustics
        old_mu, old_detect, old_nominal = self.legacy.parameters(scenario, clocks, mode)
        old_aperture = aperture(a, old_mu)
        raw = np.divide(old_detect, old_aperture, out=np.zeros_like(old_detect), where=old_aperture > 1e-250)
        mu = np.empty((len(clocks), len(self.addresses), a.MAX_KINDS, 2))
        detect = np.empty(mu.shape[:-1])
        nominal = np.empty((len(self.addresses), 2))
        bias = np.array([0., scenario.calibration_depth])
        for i, (site, view) in enumerate(self.addresses):
            if view in ANGLES:
                j = self.legacy.index[site, view]
                mu[:, i], detect[:, i], nominal[i] = old_mu[:, j], old_detect[:, j], old_nominal[j]
                continue
            theta = self.angles[site, view]
            center = self.legacy.index[site, 3]
            transform = self.transforms[site, view]
            mu[:, i] = (old_mu[:, center] - bias) @ transform + bias
            nominal[i] = old_nominal[center] @ transform
            detect[:, i] = raw[:, center]
            va, vb, ta, tb = (7, 3, -12., 0.) if theta < 0 else (3, 8, 0., 12.)
            ja, jb = self.legacy.index[site, va], self.legacy.index[site, vb]
            u = (theta - ta) / (tb - ta)
            weight = u * u * (3. - 2. * u)
            frame = self.frames[site, view]
            fa = self.world._blend(self.world.scan.view_frames[:, site, va], 15.)
            fb = self.world._blend(self.world.scan.view_frames[:, site, vb], 15.)
            # Original endpoint clipping can differ by visibility regime.
            # Smoothly transport both effective endpoint ghost locations to
            # this same view. The same case/site remains fixed at every step.
            ga = (old_mu[:, ja, 2] - bias) @ (fa[:, 1:3].T @ frame[:, 1:3]) + bias
            gb = (old_mu[:, jb, 2] - bias) @ (fb[:, 1:3].T @ frame[:, 1:3]) + bias
            mu[:, i, 2] = (1. - weight) * ga + weight * gb
            detect[:, i, 2] = (1. - weight) * raw[:, ja, 2] + weight * raw[:, jb, 2]
            detect[:, i] *= aperture(a, mu[:, i])
        return mu, detect, nominal


def parity_report(acoustics, geometry, seeds=(918001, 918002, 918003)):
    clocks = np.arange(7) * 1.5
    report = {"legacy_kernel_max_abs": 0., "view_id_invariance_max_abs": 0.,
              "new_to_nearest_legacy_mean_total_variation": [],
              "ghost_extension": "smooth transport and visibility interpolation of fixed legacy anchor regimes",
              "acoustic_validation": "synthetic interpolation only, not measured ultrasound"}
    for seed in seeds:
        scenario = acoustics.make_scenario(geometry.world, seed)
        p, c, _, _ = acoustics.probabilities(geometry, scenario, clocks, "joint")
        lp, lc, _, _ = acoustics.probabilities(geometry.legacy, scenario, clocks, "joint")
        for i, point in enumerate(geometry.addresses):
            if point[1] in ANGLES:
                j = geometry.legacy.index[point]
                report["legacy_kernel_max_abs"] = max(report["legacy_kernel_max_abs"],
                    float(np.max(np.abs(p[:, i] - lp[:, j]))), float(np.max(np.abs(c[:, i] - lc[:, j]))))
            else:
                angle = geometry.angles[point]
                v = min(ANGLES, key=lambda view: abs(ANGLES[view] - angle))
                j = geometry.legacy.index[point[0], v]
                report["new_to_nearest_legacy_mean_total_variation"].append(
                    dict(seed=seed, site=point[0], angle=angle,
                         tv=float(.5 * np.abs(p[:, i] - lp[:, j]).sum(axis=-1).mean())))
    aliases = [dict(n, view=int(n["view"]) + 1000) if int(n["view"]) not in ANGLES else dict(n)
               for n in geometry.nodes]
    alias_geometry = Geometry(acoustics, geometry.world, aliases, geometry.label)
    scenario = acoustics.make_scenario(geometry.world, seeds[0])
    first = acoustics.probabilities(geometry, scenario, clocks, "joint")[:2]
    second = acoustics.probabilities(alias_geometry, scenario, clocks, "joint")[:2]
    report["view_id_invariance_max_abs"] = max(float(np.max(np.abs(x-y))) for x,y in zip(first,second))
    return report
