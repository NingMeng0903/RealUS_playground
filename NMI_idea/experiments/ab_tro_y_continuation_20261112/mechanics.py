"""Timed ideal actuator reference on the registered moving-Y contact chart."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


POSE_GRID = np.arange(11., 20.)
MAX_KNEE_RATE = 3.85 * 2. * np.pi / 5.4
CHECK_PROGRESS = np.linspace(0., 1., 33)
SAMPLE_DT = .020
CAPTURE_DWELL = .21
POSITION_TOL_M = .00025
ROTATION_TOL_RAD = np.deg2rad(.15)


@dataclass(frozen=True)
class Path:
    start: tuple[int, int]
    goal: tuple[int, int]
    duration: float
    stroke_mm: float
    legal: bool
    reason: str
    cost_mm: float
    max_reference_speed_fraction: float
    min_self_collision_m: float | None
    min_joint_margin: float | None
    max_ref_fk_error_mm: float | None

    @property
    def id(self):
        return f'{self.start[0]}:{self.start[1]}>{self.goal[0]}:{self.goal[1]}'


class MotionModel:
    """Simulated ideal joint tracking with common URDF rate and position limits."""

    def __init__(self, world):
        self.world = world
        self.robot = world.edge.robot
        self.kin = self.robot.kin
        self.departure = world.initial_route[-1]
        self.seed = np.copy(world.scan.q[(world.knots[1], *self.departure)])
        self.rail_m = float(self.seed[0])
        self.limits = np.asarray(self.kin.v_max, dtype=float)
        with np.load(world.geometry_path, allow_pickle=False) as geometry:
            self.robot_from_atlas = geometry['robot_from_atlas']
        self.atlas_from_robot = np.linalg.inv(self.robot_from_atlas)
        self.order = world.scan.surface_order
        self.s = world.scan.s
        self.offsets = world.scan.offsets
        self.path_meta = {}
        self.q_reference = {}
        self.endpoint_q = {}

    @lru_cache(maxsize=128)
    def pose_geometry(self, angle):
        return (self.world._blend(self.world.scan.views, angle),
                Rotation.from_matrix(self.world._blend(self.world.scan.view_frames, angle).reshape(-1, 3, 3))
                .as_matrix().reshape(self.world.n, self.world.n_views, 3, 3))

    def chart_reference(self, start, goal, progress, angle):
        views, frames = self.pose_geometry(float(angle))
        p = np.asarray(progress, dtype=float)
        a, va = start
        b, vb = goal
        station = self.s[a] + p * (self.s[b] - self.s[a])
        offset = self.offsets[va] + p * (self.offsets[vb] - self.offsets[va])
        right = np.clip(np.searchsorted(self.s, station), 1, len(self.s) - 1)
        left = right - 1
        along = (station - self.s[left]) / (self.s[right] - self.s[left])
        x_grid = self.offsets[self.order]
        upper = np.clip(np.searchsorted(x_grid, offset), 1, len(x_grid) - 1)
        lower = upper - 1
        across = (offset - x_grid[lower]) / (x_grid[upper] - x_grid[lower])
        xyz = ((1. - along[:, None]) * ((1. - across[:, None]) * views[left, self.order[lower]]
                                       + across[:, None] * views[left, self.order[upper]])
               + along[:, None] * ((1. - across[:, None]) * views[right, self.order[lower]]
                                   + across[:, None] * views[right, self.order[upper]]))
        start_rotation = Rotation.from_matrix(frames[a, va])
        rotation_delta = (start_rotation.inv() * Rotation.from_matrix(frames[b, vb])).as_rotvec()
        orientation = (start_rotation * Rotation.from_rotvec(p[:, None] * rotation_delta)).as_matrix()
        target = np.broadcast_to(np.eye(4), (len(p), 4, 4)).copy()
        target[:, :3, 3] = xyz
        target[:, :3, :3] = orientation
        return self.robot_from_atlas @ target

    def ik(self, target, seed):
        q = np.array(seed, dtype=float, copy=True)
        for _ in range(18):
            actual = self.robot.fk(q)
            translation = target[:3, 3] - actual[:3, 3]
            rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            if np.linalg.norm(translation) < 4.e-6 and np.linalg.norm(rotation) < 4.e-5:
                return q
            jac = self.kin.jacobian(q)[:, 1:]
            delta = np.linalg.solve(jac @ jac.T + 2.e-7 * np.eye(6), np.r_[translation, rotation])
            q[1:] += np.clip(jac.T @ delta, -.13, .13)
        raise ValueError('local fixed-rail IK did not converge')

    @staticmethod
    def blend(p):
        return p**3 * (10. + p * (-15. + 6. * p))

    @lru_cache(maxsize=256)
    def profile(self, duration):
        time = np.linspace(0., duration, int(np.ceil(duration / SAMPLE_DT)) + 1)
        p = self.blend(np.minimum(time / (duration - CAPTURE_DWELL), 1.))
        return time, p

    def _cost_at_pose(self, path, angle):
        _, p = self.profile(path.duration)
        points = self.chart_reference(path.start, path.goal, p, angle)[:, :3, 3]
        return float(1000. * np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    def register(self, start, goal, duration):
        key = (start, goal)
        if key in self.path_meta:
            if abs(self.path_meta[key].duration - duration) > 1.e-12:
                raise RuntimeError('registered path duration changed')
            return self.path_meta[key]
        stroke = float(abs(self.offsets[goal[1]] - self.offsets[start[1]]))
        if not (self.world.allowed[start] and self.world.allowed[goal] and self.world.edge_valid(start, goal)):
            path = Path(start, goal, duration, stroke, False, 'baked edge or endpoint gate', 0., 0., None, None, None)
            self.path_meta[key] = path
            return path
        reference, all_checks = [], []
        try:
            for angle in POSE_GRID:
                targets = self.chart_reference(start, goal, CHECK_PROGRESS, angle)
                seeds = self.endpoint_q.get(start)
                q = self.ik(targets[0], self.seed if seeds is None else seeds[int(angle - POSE_GRID[0])])
                qs = []
                for target in targets:
                    q = self.ik(target, q)
                    qs.append(q)
                    all_checks.append(self.robot.check_q(q, target))
                reference.append(qs)
        except ValueError as error:
            path = Path(start, goal, duration, stroke, False, str(error), 0., 0., None, None, None)
            self.path_meta[key] = path
            return path
        q = np.asarray(reference)
        slopes_angle = abs(np.diff(q, axis=0)) / np.diff(POSE_GRID)[:, None, None]
        timed_t, timed_p = self.profile(duration)
        timed_ref = np.asarray([[np.interp(timed_p, CHECK_PROGRESS, one[:, k])
                                 for k in range(8)] for one in q]).transpose(0, 2, 1)
        timed_rate = abs(np.diff(timed_ref, axis=1)) / np.diff(timed_t)[None, :, None]
        max_speed = timed_rate.max(axis=(0, 1)) + slopes_angle.max(axis=(0, 1)) * MAX_KNEE_RATE
        fraction = float(np.max(max_speed / self.limits))
        valid = bool(all(x['valid'] for x in all_checks) and fraction <= 1.)
        reason = ('passed' if valid else ('prechecked q/FK/self-collision failure'
                  if not all(x['valid'] for x in all_checks) else 'reference speed headroom'))
        path = Path(start, goal, duration, stroke, valid, reason, 0., fraction,
                    min(x['min_self_collision_distance_m'] for x in all_checks),
                    min(min(x['joint_limit_margin_by_coordinate'][1:]) for x in all_checks),
                    1000. * max(x['fk_position_error_m'] for x in all_checks))
        path = Path(**(path.__dict__ | {'cost_mm': self._cost_at_pose(path, 15.)}))
        self.path_meta[key] = path
        if valid:
            self.q_reference[key] = q
            self.endpoint_q.setdefault(goal, q[:, -1].copy())
        return path

    def command(self, path, progress, angle):
        p, a = float(progress), float(np.clip(angle, POSE_GRID[0], POSE_GRID[-1]))
        qa = self.q_reference[(path.start, path.goal)]
        ai = int(np.clip(np.searchsorted(POSE_GRID, a), 1, len(POSE_GRID) - 1))
        pi = int(np.clip(np.searchsorted(CHECK_PROGRESS, p), 1, len(CHECK_PROGRESS) - 1))
        u = (a - POSE_GRID[ai-1]) / (POSE_GRID[ai] - POSE_GRID[ai-1])
        v = (p - CHECK_PROGRESS[pi-1]) / (CHECK_PROGRESS[pi] - CHECK_PROGRESS[pi-1])
        q = ((1. - u) * ((1. - v) * qa[ai-1, pi-1] + v * qa[ai-1, pi])
             + u * ((1. - v) * qa[ai, pi-1] + v * qa[ai, pi]))
        target = self.chart_reference(path.start, path.goal, np.array([p]), a)[0]
        return self.ik(target, q), target

    def execute(self, path, start_t, pose_sensor, previous_q=None):
        if not path.legal:
            raise ValueError('execution requires a prechecked path')
        relative_t, progress = self.profile(path.duration)
        qs, tcp, atlas, angles, checks = [], [], [], [], []
        passed, reason = True, 'passed'
        max_rate = 0.
        for index, (dt, p) in enumerate(zip(relative_t, progress)):
            t = start_t + dt
            # The ideal simulator exports a synchronized low-level pose signal.
            # High-level decisions use only the noisy capture packets.
            angle = float(pose_sensor(t))
            desired, target = self.command(path, p, angle)
            old = (np.asarray(previous_q) if index == 0 and previous_q is not None
                   else (qs[-1] if index > 0 else desired))
            if index == 0:
                actual = old
            else:
                step = float(relative_t[index] - relative_t[index-1])
                velocity = (desired - old) / step
                max_rate = max(max_rate, float(np.max(abs(velocity) / self.limits)))
                actual = old + step * np.clip(velocity, -self.limits, self.limits)
            fk = self.robot.fk(actual)
            check = self.robot.check_q(actual, target)
            checks.append(check)
            qs.append(actual)
            tcp.append(fk)
            atlas.append((self.atlas_from_robot @ fk)[:3, 3] * 1000.)
            angles.append(angle)
            if (not check['limits_pass'] or not check['self_collision_pass']
                    or check['fk_position_error_m'] > POSITION_TOL_M
                    or check['fk_rotation_error_rad'] > ROTATION_TOL_RAD or max_rate > 1. + 1.e-8):
                passed, reason = False, 'sampled actuator, FK or self-collision limit'
                break
        path_xyz = np.asarray(atlas)
        rotation = Rotation.from_matrix(np.asarray(tcp)[:, :3, :3])
        turn = float(np.rad2deg((rotation[:-1].inv() * rotation[1:]).magnitude()).sum()) if len(qs)>1 else 0.
        times = relative_t[:len(qs)]
        trace = dict(path_id=path.id, start=list(path.start), goal=list(path.goal),
                     legal_completed=passed, reason=reason, t_start=float(start_t),
                     t_end=float(start_t + times[-1]), duration_s=float(times[-1]),
                     travel_mm=float(np.linalg.norm(np.diff(path_xyz, axis=0), axis=1).sum()),
                     turn_deg=turn, stroke_mm=float(path.stroke_mm * progress[len(qs)-1]),
                     commanded_stroke_mm=path.stroke_mm,
                     commanded_duration_s=path.duration,
                     max_commanded_joint_rate_fraction=max_rate,
                     min_self_collision_m=min(c['min_self_collision_distance_m'] for c in checks),
                     max_reference_fk_error_mm=1000. * max(c['fk_position_error_m'] for c in checks),
                     max_reference_fk_error_deg=float(np.rad2deg(max(c['fk_rotation_error_rad'] for c in checks))),
                     samples=dict(t_s=(times+start_t).tolist(), knee_deg=angles,
                                  atlas_xyz_mm=path_xyz.tolist(), q=np.asarray(qs).tolist(),
                                  robot_tcp_rotation=np.asarray(tcp)[:, :3, :3].tolist()))
        return trace, qs[-1]
