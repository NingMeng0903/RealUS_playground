"""Synthesize and screen learned ProDMP motion in the static artery/vein atlas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import lzma
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SEED = 20261112
SLOT_S = 1.50
DWELL_S = .21
SAMPLE_DT_S = .020
POSITION_TOL_M = .0001
ROTATION_TOL_RAD = .001


@dataclass(frozen=True)
class MotionTrace:
    legal: bool
    reason: str
    q: np.ndarray
    t_s: np.ndarray
    target: np.ndarray
    fk: np.ndarray
    cost: np.ndarray
    diagnostics: dict


def _digest(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_sources(repo, robot_repo):
    sys.path[:0] = [str(repo), str(robot_repo), str(robot_repo / 'rm75_control')]
    pkg = 'NMI_idea.experiments.'
    capability = importlib.import_module(pkg + 'ab_module_ladder_20261110.capability.capability')
    primitive = importlib.import_module(pkg + 'ab_leg_bifurcation_20261112.learned_prodmp')
    # Importing the archive's bank builder creates an output directory as a side
    # effect. The local fixed-rail solver uses its same damped Jacobian steps.
    return capability, primitive


def _cartesian_ik(capability, target, rail, seed):
    q = np.array(seed, dtype=float, copy=True)
    q[0] = rail
    for _ in range(36):
        actual = capability.fk(q)
        translation = target[:3, 3] - actual[:3, 3]
        rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        if np.linalg.norm(translation) < 4.e-6 and np.linalg.norm(rotation) < 4.e-5:
            return q
        jac = capability.kin.jacobian(q)[:, 1:]
        delta = np.linalg.solve(jac @ jac.T + 2.e-7 * np.eye(6), np.r_[translation, rotation])
        q[1:] += np.clip(jac.T @ delta, -.13, .13)
    return q  # The explicit FK gate records the residual and rejects failures.


class ProDMPSynthesizer:
    """Generate, check and integrate actual learned joint paths on demand."""

    def __init__(self, repo, robot_repo, geometry, *, primitive_mode='learned'):
        if primitive_mode not in {'learned', 'endpoint_only'}:
            raise ValueError('primitive_mode must be learned or endpoint_only')
        self.primitive_mode = primitive_mode
        repo, robot_repo = Path(repo).resolve(), Path(robot_repo).resolve()
        capability, primitive = _load_sources(repo, robot_repo)
        self.cap = capability.CapabilityEvaluator(device='cpu')
        self.primitive = primitive.LearnedProDMP.fit_default(seed=SEED)
        exp = repo / 'NMI_idea' / 'experiments'
        bank_path = exp / 'ab_full_b_rollout_20261113' / 'motion' / 'BANK_MAIN.json.xz'
        atlas_path = exp / 'ab_leg_dynamic_20261112' / 'DYNAMIC_GEOMETRY_FROZEN.npz'
        with lzma.open(bank_path, 'rt', encoding='utf-8') as stream:
            nodes = {row['id']: row for row in json.load(stream)['nodes']}
        self.poses = geometry['poses']
        for pose in self.poses:
            node = nodes[pose['id']]
            if any(node[key] != pose[key] for key in ('site', 'view', 'realization')):
                raise ValueError('geometry pose does not match its archived dIRD endpoint')
        self.endpoint_q = np.asarray([nodes[pose['id']]['q'] for pose in self.poses])
        self.endpoint_q.setflags(write=False)
        sites = np.asarray([p['site'] for p in self.poses])
        self.allowed = np.abs(sites[:, None] - sites) <= 1
        self.allowed.setflags(write=False)
        with np.load(atlas_path, allow_pickle=False) as data:
            static_pose = int(np.flatnonzero(data['pose_knee_deg'] == geometry['pose_knee_deg']).item())
            self._station = data['s_mm']
            self._offsets = data['view_offsets_mm']
            self._views = data['posed_views_m'][static_pose]
            self._frames = data['posed_frames'][static_pose]
            self._robot_from_atlas = data['robot_from_atlas']
        self._surface_order = np.unique(self._offsets, return_index=True)[1]
        self._surface_offsets = self._offsets[self._surface_order]
        self._atlas_endpoints = np.stack([self._atlas_pose(p['site'], p['view']) for p in self.poses])
        self._duration = SLOT_S - DWELL_S
        self._transit_t = np.linspace(0., self._duration, int(np.ceil(self._duration / SAMPLE_DT_S)) + 1)
        self._dwell_t = self._duration + np.linspace(0., DWELL_S, int(np.ceil(DWELL_S / SAMPLE_DT_S)) + 1)[1:]
        self._t_s = np.r_[self._transit_t, self._dwell_t]
        self._endpoints_check = [self.cap.check_q(q, t) for q, t in zip(
            self.endpoint_q, self._atlas_endpoints, strict=True)]
        if not all(check['valid'] for check in self._endpoints_check):
            raise ValueError('archived dIRD endpoints fail the current robot model')
        self._manifest = self.cap.manifest()
        self._sources = {str(p.name): _digest(p) for p in (
            bank_path, atlas_path, Path(capability.__file__), Path(primitive.__file__))}
        if primitive.BASIS_SOURCE == 'reference':
            self._sources['reference_basis'] = _digest(primitive.REFERENCE_MOTION)
        self._weights_hash = hashlib.sha256(np.asarray(self.primitive.weight_mean['advance'],
                                                       dtype='<f8').tobytes()).hexdigest()
        self._basis_source = primitive.BASIS_SOURCE

    def _atlas_pose(self, site, view):
        local = np.eye(4)
        local[:3, 3] = self._views[site, view]
        local[:3, :3] = self._frames[site, view]
        return self._robot_from_atlas @ local

    def _surface_position(self, chart):
        station, offset = chart.T
        if (np.any(station < self._station[0]) or np.any(station > self._station[-1]) or
                np.any(offset < self._surface_offsets[0]) or np.any(offset > self._surface_offsets[-1])):
            raise ValueError('ProDMP leaves the archived surface chart')
        right = np.clip(np.searchsorted(self._station, station), 1, len(self._station) - 1)
        left = right - 1
        top = np.clip(np.searchsorted(self._surface_offsets, offset), 1, len(self._surface_offsets) - 1)
        bottom = top - 1
        along = ((station - self._station[left]) / (self._station[right] - self._station[left]))[:, None]
        across = ((offset - self._surface_offsets[bottom]) /
                  (self._surface_offsets[top] - self._surface_offsets[bottom]))[:, None]
        lower_view, upper_view = self._surface_order[bottom], self._surface_order[top]
        p0 = (1. - across) * self._views[left, lower_view] + across * self._views[left, upper_view]
        p1 = (1. - across) * self._views[right, lower_view] + across * self._views[right, upper_view]
        local = (1. - along) * p0 + along * p1
        return local @ self._robot_from_atlas[:3, :3].T + self._robot_from_atlas[:3, 3]

    def _reference(self, start, goal, initial):
        a, b = self.poses[start], self.poses[goal]
        c0 = np.array([self._station[a['site']], self._offsets[a['view']]])
        c1 = np.array([self._station[b['site']], self._offsets[b['view']]])
        samples = len(self._transit_t)
        learned = self.primitive_mode == 'learned'
        goal_pose = self._atlas_endpoints[goal]
        if start == goal:
            target = np.repeat(goal_pose[None], samples, axis=0)
            return target, np.zeros(samples), np.repeat(c0[None], samples, axis=0), 'hold'
        delta = c1 - c0
        if np.linalg.norm(delta) > 0.:
            chart = self.primitive.path(c0, c1, skill='advance', samples=samples, learned=learned)
            progress = (chart - c0) @ delta / (delta @ delta)
            xyz = self._surface_position(chart)
        else:
            unit = self.primitive.path([0., 0.], [1., 0.], skill='advance', samples=samples, learned=learned)
            progress = unit[:, 0]
            chart = np.repeat(c0[None], samples, axis=0)
            xyz = np.repeat(goal_pose[:3, 3][None], samples, axis=0)
        progress = np.clip(progress, 0., 1.)
        # Preserve the learned chart excursion. Blend only the small initial FK
        # alignment residual so each new reference begins at the realized pose.
        start_nominal = self._atlas_endpoints[start]
        xyz += ((1. - progress)[:, None] * (initial[:3, 3] - start_nominal[:3, 3]) +
                progress[:, None] * (goal_pose[:3, 3] - xyz[-1]))
        rotations = Rotation.from_matrix(np.stack((initial[:3, :3], goal_pose[:3, :3])))
        target = np.repeat(np.eye(4)[None], samples, axis=0)
        target[:, :3, :3] = Slerp([0., 1.], rotations)(progress).as_matrix()
        target[:, :3, 3] = xyz
        target[0], target[-1] = initial, goal_pose
        return target, progress, chart, self.primitive_mode

    def execute(self, start, goal, current_q):
        """Regenerate the full command from the supplied exit configuration."""
        if not self.allowed[start, goal]:
            raise ValueError('transition is outside the declared action graph')
        q0 = np.asarray(current_q, dtype=float)
        if q0.shape != (8,) or not np.allclose(q0, self.endpoint_q[start], rtol=0., atol=1.e-8):
            raise ValueError('realized exit does not match the declared start configuration')
        q_goal = self.endpoint_q[goal]
        start_t = perf_counter()
        target, progress, chart, kind = self._reference(start, goal, self.cap.fk(q0))
        q = np.empty((len(self._t_s), len(q0)))
        q[0] = q0
        transit_count = len(self._transit_t)
        if kind == 'hold':
            q[1:transit_count - 1] = q0
        else:
            rail = q0[0] + progress * (q_goal[0] - q0[0])
            # The 7R arm retains a swivel degree of freedom at fixed rail.
            # Use both endpoint branches so a late exact closure cannot make
            # an otherwise correct Cartesian path jump in the null space.
            forward = np.empty((transit_count, len(q0)))
            backward = np.empty_like(forward)
            forward[0], backward[-1] = q0, q_goal
            for i in range(1, transit_count - 1):
                forward[i] = _cartesian_ik(self.cap, target[i], rail[i], forward[i - 1])
            for i in range(transit_count - 2, 0, -1):
                backward[i] = _cartesian_ik(self.cap, target[i], rail[i], backward[i + 1])
            for i in range(1, transit_count - 1):
                seed = (1. - progress[i]) * forward[i] + progress[i] * backward[i]
                q[i] = _cartesian_ik(self.cap, target[i], rail[i], seed)
        # The goal and all dwell samples use the archived endpoint. The final
        # transition is screened under the same rate/FK gates as every other.
        q[transit_count - 1:] = q_goal
        target = np.concatenate((target, np.repeat(target[-1:], len(self._dwell_t), axis=0)))
        actual = np.stack([self.cap.fk(row) for row in q])
        checks = [self.cap.check_q(row, pose) for row, pose in zip(q, target, strict=True)]
        steps = np.diff(q, axis=0)
        fractions = np.abs(steps) / np.diff(self._t_s)[:, None] / np.asarray(self.cap.kin.v_max)
        rate = float(fractions.max())
        reasons = []
        if not all(c['limits_pass'] for c in checks):
            reasons.append('sampled joint limits')
        if not all(c['self_collision_pass'] for c in checks):
            reasons.append('sampled robot/probe self-collision')
        if not all(c['fk_pass'] for c in checks):
            reasons.append('sampled FK tracking error')
        if rate > 1. + 1.e-10:
            reasons.append('sampled joint velocity')
        if start != goal and not np.array_equal(q[-1], q_goal):
            reasons.append('goal configuration closure')
        orientation = Rotation.from_matrix(actual[:, :3, :3])
        travel_mm = 1.e3 * np.linalg.norm(np.diff(actual[:, :3, 3], axis=0), axis=1).sum()
        turn_deg = np.rad2deg((orientation[:-1].inv() * orientation[1:]).magnitude()).sum()
        rail_mm = 1.e3 * np.abs(steps[:, 0]).sum()
        cost = np.array([travel_mm, turn_deg, rail_mm, self._t_s[-1]])
        clearance = [c['min_self_collision_distance_m'] for c in checks]
        diagnostics = dict(kind=kind, sampled_clearance_m=min(clearance) if all(
            x is not None for x in clearance) else None,
            min_joint_margin=float(min(min(c['joint_limit_margin_by_coordinate']) for c in checks)),
            max_position_error_mm=float(1.e3 * max(c['fk_position_error_m'] for c in checks)),
            max_rotation_error_deg=float(np.rad2deg(max(c['fk_rotation_error_rad'] for c in checks))),
            max_rate_fraction=rate, final_transit_step_rate_fraction=float(fractions[transit_count - 2].max()),
            max_transit_step_s=float(np.diff(self._transit_t).max()), synthesis_wall_s=perf_counter() - start_t,
            chart_mm=chart.tolist())
        return MotionTrace(not reasons, '; '.join(reasons) if reasons else 'passed', q, self._t_s.copy(),
                           target, actual, cost, diagnostics)

    def compile(self):
        """Construct the common action-cost surface from new learned traces."""
        costs = np.full((*self.allowed.shape, 4), np.inf)
        rows = []
        for a, b in np.argwhere(self.allowed):
            a, b = int(a), int(b)
            try:
                trace = self.execute(a, b, self.endpoint_q[a])
            except (ValueError, np.linalg.LinAlgError) as error:
                rows.append(dict(start=a, goal=b, legal=False, reason=str(error)))
                continue
            if trace.legal:
                costs[a, b] = trace.cost
            rows.append(dict(start=a, goal=b, legal=trace.legal, reason=trace.reason,
                             cost=trace.cost.tolist(), **trace.diagnostics,
                             t_s=trace.t_s.tolist(), q=trace.q.tolist(), target=trace.target.tolist(),
                             fk=trace.fk.tolist()))
        return costs, dict(attempted=len(rows), accepted=sum(row['legal'] for row in rows),
                           mode=self.primitive_mode, fitted_seed=SEED, basis_source=self._basis_source,
                           advance_weight_sha256=self._weights_hash, source_sha256=self._sources,
                           slot_s=SLOT_S, capture_dwell_s=DWELL_S, max_sample_dt_s=SAMPLE_DT_S,
                           position_tolerance_mm=1.e3 * POSITION_TOL_M,
                           rotation_tolerance_deg=float(np.rad2deg(ROTATION_TOL_RAD)),
                           poses=self.poses, endpoint_q=self.endpoint_q.tolist(),
                           endpoint_atlas_checks=self._endpoints_check, robot=self._manifest,
                           edges=rows)
