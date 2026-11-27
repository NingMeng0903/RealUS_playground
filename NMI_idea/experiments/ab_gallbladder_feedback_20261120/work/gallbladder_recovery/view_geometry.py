"""Generate diverse scan views from a registered surface and supplied anatomy models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
from shapely.geometry import Point


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'frozen_gallbladder_experiment', HERE.parent / 'gallbladder_intent' / 'pilot.py'
)
prior_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior_run)

from gallbladder_geometry import BEAM_ANGLE, DEPTH_MM  # noqa: E402
from gallbladder_mpc import CONTACT  # noqa: E402
from servo import FRAME_STEPS, Motion  # noqa: E402


REGIONS = ('neck', 'body', 'fundus')
POSE_DISTANCE_SCALE = np.array([1., 2.2, .45, .55])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1.e-12)


def frame_count(motion):
    return (len(motion.x) - 1) // FRAME_STEPS


def settled_plan(motor, q, target, frames_left):
    """Admit only motions with a checked, stationary endpoint inside the budget."""
    if frames_left < 4:
        return None
    start_velocity = np.zeros(4)
    plan = motor.replan(q, start_velocity, target)
    if plan is not None and frame_count(plan) <= frames_left:
        if np.max(np.abs(plan.v[-1])) < 1.e-7:
            return plan
    # A quintic gives an exactly endpoint-matched fallback if the fitted
    # primitive overshoots a contact boundary on departure or arrival.
    for frames in (5, 7, 10, 15, 22, 33, 48, 72, 105):
        if frames > frames_left:
            break
        t = np.linspace(0., 1., frames * FRAME_STEPS + 1)[:, None]
        delta = target - q
        x = q + (t**3 * (10. - 15. * t + 6. * t**2)) * delta
        v = 30. * t**2 * (1. - t)**2 * delta / (frames * prior_run.legacy_scene.FRAME_DT)
        plan = Motion(x, v)
        if motor.feasible(plan):
            return plan
    return None


@dataclass(frozen=True)
class ViewMetrics:
    qualifies: bool
    score: float
    center: float
    angle: float
    area: float
    coverage: float
    visibility: tuple


def measure_view(body, q, anatomy, rib):
    """Measure with the unchanged v2 geometry and explicit center/axis gates."""
    view = body.view(q, rib)
    section = body.measure(view, anatomy)
    area, x = section.gall[:2]
    angle = float(np.rad2deg(np.arcsin(np.clip(abs(anatomy.axis @ view.normal), 0., 1.))))
    points = np.stack([anatomy.landmarks[name] for name in REGIONS]) - view.origin
    lateral = points @ view.right
    depth = points @ view.forward
    normal = points @ view.normal
    inside = (
        (depth >= 4.) & (np.hypot(lateral, depth) <= DEPTH_MM)
        & (np.abs(lateral) <= depth * np.tan(BEAM_ANGLE))
        & ~view.beam_blocked(lateral, depth)
    )
    visible = inside * np.exp(-.5 * (np.maximum(np.abs(normal) - 5., 0.) / 7.2)**2)
    polygon = body._section(anatomy.gallbladder, body.gallbladder_faces, view)
    if polygon is None:
        visible[:] = 0.
    else:
        polygon = polygon.intersection(body._fan_polygon())
        shadow = body._shadow_polygon(view)
        if shadow is not None:
            polygon = polygon.difference(shadow)
        if hasattr(polygon, 'geoms'):
            polygon = max(polygon.geoms, key=lambda component: component.area, default=None)
        if polygon is None or polygon.is_empty:
            visible[:] = 0.
        else:
            distance = np.array([polygon.distance(Point(px, pd)) for px, pd in zip(lateral, depth)])
            visible *= np.exp(-.5 * (np.maximum(distance - 2., 0.) / 4.5)**2)
    strict = bool(section.qualified and abs(x) <= 12. and angle <= 15.)
    center_quality = np.exp(-.5 * (x / 14.)**2) if area > 0 else 0.
    score = (
        2.4 * strict + .95 * section.qualified + 1.65 * visible.min() + .27 * visible.mean()
        + .42 * section.coverage + .75 * center_quality * min(area / 160., 1.)
        - .15 * max(angle - 12., 0.) / 20.
    )
    return ViewMetrics(strict, float(score), float(x), angle, float(area),
                       float(section.coverage), tuple(float(v) for v in visible))


class AtlasViews:
    def __init__(self, body):
        self.body = body
        y, theta = np.meshgrid(np.arange(-269., -131., 8.), np.arange(-82., 12., 5.), indexing='ij')
        self.contacts = np.column_stack((y.ravel(), theta.ravel()))
        self._base = self._surface_basis(self.contacts)

    def _surface_basis(self, xy):
        d = np.array([[0., 0.], [.5, 0.], [-.5, 0.], [0., .4], [0., -.4]])
        p = self.body.surface((xy[:, None, :] + d[None, :, :]).reshape(-1, 2))
        origin, above, below, around, opposite = np.moveaxis(p.reshape(len(xy), 5, 3), 1, 0)
        right = unit(above - below)
        forward = unit(np.cross(right, around - opposite))
        normal = np.cross(forward, right)
        return origin, forward, right, normal

    def _pose_seeds(self, anatomy, contacts, basis):
        origin, forward, right, normal = basis
        marks = np.stack([anatomy.landmarks[name] for name in REGIONS])
        rel = marks[None, :, :] - origin[:, None, :]
        center_ray = anatomy.center - origin
        # Include both the end-to-end chord and planes that accommodate the
        # bend through the body label. The actual polygon is checked later.
        gram = np.einsum('npi,npj->nij', rel, rel)
        _, eigenvectors = np.linalg.eigh(gram)
        normals = np.stack((
            np.cross(center_ray, anatomy.axis),
            np.cross(rel[:, 0], rel[:, 2]),
            eigenvectors[:, :, 0],
        ), axis=1)
        normals = unit(normals)
        normals *= np.where(np.sum(normals * normal[:, None, :], axis=-1) < 0., -1., 1.)[:, :, None]
        n_right = np.sum(normals * right[:, None, :], axis=-1)
        n_normal = np.sum(normals * normal[:, None, :], axis=-1)
        n_forward = np.sum(normals * forward[:, None, :], axis=-1)
        roll = np.arctan2(-n_right, n_normal)
        rock = np.arctan2(-n_forward, np.hypot(n_right, n_normal))
        angles = np.rad2deg(np.stack((roll, rock), axis=-1))
        offsets = np.array([
            [0., 0.], [-7., 0.], [7., 0.], [-14., 0.], [14., 0.],
            [0., -5.], [0., 5.], [0., -10.], [0., 10.],
            [-6., -5.], [6., -5.], [-6., 5.], [6., 5.]
        ])
        angles = np.clip(angles[:, :, None, :] + offsets[None, None, :, :], CONTACT[0, 2:], CONTACT[1, 2:])
        angle_pairs = angles.reshape(len(contacts), -1, 2)
        roll, rock = np.moveaxis(np.deg2rad(angle_pairs), -1, 0)
        r = np.cos(roll)[..., None] * right[:, None, :] + np.sin(roll)[..., None] * normal[:, None, :]
        f = (np.cos(rock)[..., None] * forward[:, None, :]
             + np.sin(rock)[..., None] * np.cross(forward[:, None, :], r))
        n = np.cross(f, r)
        x = np.einsum('npi,nki->nkp', rel, r)
        d = np.einsum('npi,nki->nkp', rel, f)
        z = np.einsum('npi,nki->nkp', rel, n)
        xc = np.einsum('ni,nki->nk', center_ray, r)
        xmid = .5 * (x[..., 0] + x[..., 2])
        out_of_plane = np.max(np.abs(z), axis=-1)
        outside_fan = np.maximum(np.hypot(x, d) - DEPTH_MM + 3., 0.)
        outside_fan += np.maximum(np.abs(x) - d * np.tan(BEAM_ANGLE), 0.)
        outside_fan += np.maximum(4. - d, 0.)
        axis = np.abs(n @ anatomy.axis)
        score = (
            -.065 * out_of_plane - .018 * np.abs(xc) - .007 * np.abs(xmid)
            - .045 * outside_fan.max(axis=-1) - 1.10 * axis
        )
        xy = np.broadcast_to(contacts[:, None, :], angle_pairs.shape)
        return np.concatenate((xy, angle_pairs), axis=-1), score

    def proposals(self, anatomy, *, current=None, limit=85, local=True):
        poses, scores = self._pose_seeds(anatomy, self.contacts, self._base)
        if current is not None and local:
            offsets = np.array([(dy, dt) for dy in (-12., -6., 0., 6., 12.)
                                for dt in (-5., -2.5, 0., 2.5, 5.)])
            xy = np.unique(np.clip(current[:2] + offsets, CONTACT[0, :2], CONTACT[1, :2]), axis=0)
            local_pose, local_score = self._pose_seeds(anatomy, xy, self._surface_basis(xy))
            poses = np.concatenate((poses, local_pose))
            scores = np.concatenate((scores, local_score + .10))
        poses = poses.reshape(-1, 4)
        scores = scores.ravel()
        if current is not None:
            distance = np.linalg.norm((poses - current) * POSE_DISTANCE_SCALE, axis=1)
            scores = scores - .0018 * distance
        # Non-maximum suppression covers separate windows instead of filling
        # the exact-section budget with near-identical roll/rock candidates.
        chosen = []
        scaled = poses * POSE_DISTANCE_SCALE
        remaining = np.argsort(-scores, kind='stable')
        while len(remaining) and len(chosen) < limit:
            i = remaining[0]
            chosen.append(i)
            delta = scaled[remaining] - scaled[i]
            remaining = remaining[np.sum(delta**2, axis=1) >= 5.4**2]
        return poses[chosen]


def offsets_around(q):
    """Coordinated small translations and tilts, including along the body axis."""
    offsets = [(0., 0., 0., 0.)]
    for i, step in enumerate((4., 2., 5., 4.)):
        for sign in (-1., 1.):
            delta = np.zeros(4)
            delta[i] = sign * step
            offsets.append(tuple(delta))
    offsets.extend(((4., 2., 0., 0.), (-4., -2., 0., 0.), (4., -2., 0., 0.),
                    (-4., 2., 0., 0.), (0., 0., 5., 4.), (0., 0., -5., -4.)))
    return np.unique(np.clip(q + np.asarray(offsets), *CONTACT), axis=0)
