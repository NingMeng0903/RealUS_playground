"""Offline physical-probe45 capability bridge. No hardware/network API imports.

Inputs are *actual generated path* TCP poses in rail_base, metres, and raw
URDF rail_y. The proxy and sampled exact kinematic checks stay separate.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

SOURCE = Path('/media/camp/EXT_DRIVE/RealUS_playground/ird_playground')
ROBOT = SOURCE.parent / 'rm75_control'
for p in (SOURCE, ROBOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ird_playground.ird.gt_common import reachability_modules
from ird_playground.ird.robot_model import RobotModelSpec
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.optimization.srs_trajectory_dp import (
    SrsTrajectoryDpConfig, solve_srs_trajectory_dp,
)
from ird_playground.region.operator import base_from_rail_torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, torch.Tensor):
        return clean_json(x.detach().cpu().numpy())
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


class CapabilityBridge:
    def __init__(self, device='cpu', dtype=torch.float64):
        started = time.perf_counter()
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.dtype = dtype
        self.spec = RobotModelSpec.from_yaml(SOURCE / 'configs/robot_probe45.yaml')
        self.spec.validate()
        self.checkpoint = SOURCE / 'data/checkpoints/rm4d_signed_current_robot_20261108/selected.pt'
        self.calibration_path = self.checkpoint.parent / 'conformal.json'
        self.calibration = json.loads(self.calibration_path.read_text())
        if sha256(self.checkpoint) != self.calibration['checkpoint_sha256']:
            raise ValueError('checkpoint/calibration SHA mismatch')
        self.field = ReachabilitySDF.load(self.checkpoint, device=device, expected_robot=self.spec)
        self.field.model.to(dtype=dtype)
        for p in self.field.model.parameters():
            p.requires_grad_(False)
        reachability_modules()  # official offline namespace, avoids vendor SDK
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
        from rm75_control.control.joint_admittance_8dof.collision_model import CollisionModel
        self.kin = RobotKinematics(self.spec.kinematics_urdf, tcp_frame=self.spec.tcp_frame)
        self.collision = CollisionModel(self.kin.model, collision_urdf=self.spec.collision_urdf,
                                        pair_config=self.spec.collision_pairs)
        self.axis0 = torch.as_tensor(self.spec.root_to_j1_axis(), dtype=dtype, device=self.device)
        self.load_seconds = time.perf_counter() - started

    def score_tensor(self, tcp, rail, T_world_rail=None):
        eye = torch.eye(4, dtype=tcp.dtype, device=tcp.device)
        world = eye if T_world_rail is None else T_world_rail
        axis = base_from_rail_torch(rail, world, self.axis0.to(tcp), axis=1)
        return self.field.score_world(tcp, axis)

    def manifest(self):
        return clean_json({
            'schema': 'ab_dird_capability_v1', 'source': str(SOURCE),
            'checkpoint': str(self.checkpoint), 'checkpoint_sha256': sha256(self.checkpoint),
            'calibration_path': str(self.calibration_path),
            'calibration_sha256': sha256(self.calibration_path),
            'checkpoint_calibration_hash_match': True,
            'robot_contract': self.spec.to_manifest(),
            'checkpoint_meta': self.field.meta,
            'load_seconds': self.load_seconds,
            'pose_frame': 'rail_base', 'length_unit': 'm', 'angle_unit': 'rad',
            'q_order': ['rail_y'] + [f'joint_{i}' for i in range(1, 8)],
            'T_rail_j1_axis_at_rail0': self.axis0.detach().cpu().numpy(),
            'score_is_metric_distance': False, 'score_is_safety_certificate': False,
            'exact_scope': 'SRS fixed seed branch; FK and limits at task samples; full configured robot/probe self-collision pairs on sampled joint interpolation including departure; no body/environment/contact/dynamics checks',
        })

    def query_path(self, T_tcp_railbase, rail_m, q_seed, exact=True,
                   provenance='caller supplied actual path', max_joint_step_deg=12.,
                   collision_sample_deg=1., collision_sample_rail_m=.005,
                   psi_lower_deg=-180., psi_upper_deg=180.):
        start = time.perf_counter()
        tcp = np.asarray(T_tcp_railbase, dtype=np.float64)
        rail = np.asarray(rail_m, dtype=np.float64)
        seed = np.asarray(q_seed, dtype=np.float64)
        if tcp.ndim != 3 or tcp.shape[1:] != (4, 4) or not len(tcp):
            raise ValueError('TCP path must be nonempty (W,4,4)')
        if rail.shape != (len(tcp),) or seed.shape != (8,):
            raise ValueError('rail must be (W,), q_seed (8,)')
        if not all(np.isfinite(v).all() for v in (tcp, rail, seed)):
            raise ValueError('nonfinite query')
        if not np.allclose(tcp[:, 3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError('invalid homogeneous pose bottom row')
        R = tcp[:, :3, :3]
        if (not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-6)
                or not np.allclose(np.linalg.det(R), 1., atol=1e-6)):
            raise ValueError('invalid rotation')
        with torch.no_grad():
            score = self.score_tensor(torch.as_tensor(tcp, device=self.device, dtype=self.dtype),
                                      torch.as_tensor(rail, device=self.device, dtype=self.dtype)).cpu().numpy()
        proxy_seconds = time.perf_counter() - start
        result = {'schema': 'ab_dird_path_query_v1', 'provenance': provenance,
                  'n_waypoints': len(tcp), 'raw_score_per_waypoint': score,
                  'min_raw_score': float(score.min()),
                  # Calibration fits (raw_score-zero_bias); compare raw
                  # scores with m_safe, not the centered safety_threshold.
                  'proxy_threshold': self.calibration['m_safe'],
                  'proxy_threshold_formula': 'raw_score >= zero_bias + safety_threshold',
                  'calibration_zero_bias': self.calibration['zero_bias'],
                  'calibration_centered_safety_threshold': self.calibration['safety_threshold'],
                  'proxy_pass': bool(np.all(score >= self.calibration['m_safe'])),
                  'score_is_safety_certificate': False, 'q_seed': seed,
                  'rail_m': rail, 'exact': None,
                  'seconds': {'proxy': proxy_seconds}}
        if exact:
            cfg = SrsTrajectoryDpConfig(maximum_joint_step_deg=max_joint_step_deg,
                    psi_lower_deg=psi_lower_deg, psi_upper_deg=psi_upper_deg)
            lift = solve_srs_trajectory_dp(tcp, rail, seed, kin=self.kin, config=cfg)
            ex = {'lift_valid': lift.lift_valid, 'branch': lift.branch,
                  'reason': lift.failure, 'candidate_count_min': lift.candidate_count_min,
                  'candidate_count_max': lift.candidate_count_max,
                  'candidate_seconds': lift.candidate_seconds, 'solve_seconds': lift.solve_seconds,
                  'maximum_joint_step_rad': lift.maximum_joint_step_rad,
                  'maximum_joint_second_difference_rad': lift.maximum_joint_second_difference_rad,
                  'q_ref': lift.q_ref, 'psi_rad': lift.psi_rad,
                  'psi_interval_deg': [psi_lower_deg, psi_upper_deg],
                  'validated': False, 'scope': 'sampled kinematics/self-collision; no external obstacles or contact'}
            if lift.lift_valid:
                q = np.asarray(lift.q_ref, dtype=np.float64)
                poserr, roterr = [], []
                for qi, Ti in zip(q, tcp):
                    fk = self.kin.fk_placement(qi).homogeneous.copy()
                    poserr.append(float(np.linalg.norm(fk[:3, 3] - Ti[:3, 3])))
                    roterr.append(float(np.linalg.norm(Rotation.from_matrix(fk[:3, :3].T @ Ti[:3, :3]).as_rotvec())))
                chain = np.vstack((seed, q))
                dense = [chain[0]]
                for qa, qb in zip(chain[:-1], chain[1:]):
                    n = max(1, int(np.ceil(np.max(np.abs(qb[1:] - qa[1:])) / np.deg2rad(collision_sample_deg))),
                            int(np.ceil(abs(qb[0] - qa[0]) / collision_sample_rail_m)))
                    dense.extend(qa + u * (qb - qa) for u in np.linspace(0, 1, n + 1)[1:])
                dense = np.asarray(dense)
                lo, hi = self.kin.q_lower, self.kin.q_upper
                joint_margin = np.minimum(dense - lo, hi - dense)
                distances = []
                bad_distance = False
                for qi in dense:
                    self.collision.update(qi)  # full narrow phase, every configured pair
                    vals = [float(x.min_distance) for x in self.collision.geom_data.distanceResults]
                    if not vals or not np.isfinite(vals).all():
                        bad_distance = True
                    distances.append(min(vals) if vals else float('-inf'))
                minimum = min(distances)
                arm_step = np.max(np.abs(np.diff(chain[:, 1:], axis=0)), axis=1)
                rail_step = np.abs(np.diff(chain[:, 0]))
                maximum_arm_step = float(arm_step.max())
                maximum_rail_step = float(rail_step.max())
                arm_step_bound = np.deg2rad(float(cfg.maximum_joint_step_deg))
                rail_step_bound = float(cfg.maximum_rail_step_m)
                continuity_pass = (maximum_arm_step <= arm_step_bound + 1e-12
                                   and maximum_rail_step <= rail_step_bound + 1e-12)
                continuity_failures = []
                for i, (arm_delta, rail_delta) in enumerate(zip(arm_step, rail_step)):
                    segment = 'departure' if i == 0 else f'path segment {i - 1}->{i}'
                    if arm_delta > arm_step_bound + 1e-12:
                        continuity_failures.append(f'{segment} arm step {np.rad2deg(arm_delta):.6f} deg exceeds {cfg.maximum_joint_step_deg:.6f} deg')
                    if rail_delta > rail_step_bound + 1e-12:
                        continuity_failures.append(f'{segment} rail step {rail_delta:.6f} m exceeds {rail_step_bound:.6f} m')
                ex.update({'q_seed_to_first_max_joint_delta_rad': float(arm_step[0]),
                           'q_seed_to_first_rail_delta_m': float(rail_step[0]),
                           'maximum_joint_step_rad_including_departure': maximum_arm_step,
                           'maximum_rail_step_m_including_departure': maximum_rail_step,
                           'maximum_allowed_joint_step_deg': float(cfg.maximum_joint_step_deg),
                           'maximum_allowed_rail_step_m': rail_step_bound,
                           'continuity_failure_reasons': continuity_failures,
                           'q_exit': q[-1],
                           'rail_travel_including_departure_m': float(np.abs(np.diff(chain[:, 0])).sum()),
                           'arm_joint_l1_travel_including_departure_rad': float(np.abs(np.diff(chain[:, 1:], axis=0)).sum()),
                           'joint_limit_min_margin_by_coordinate': joint_margin.min(axis=0),
                           'limits_pass': bool(np.min(joint_margin) >= -1e-8),
                           'collision_pass': bool(not bad_distance and minimum >= self.spec.collision_security_margin_m),
                           'min_self_collision_distance_m': minimum,
                           'self_collision_required_m': self.spec.collision_security_margin_m,
                           'self_collision_margin_m': minimum - self.spec.collision_security_margin_m,
                           'collision_sample_count': len(dense),
                           'collision_max_sample_step_deg': collision_sample_deg,
                           'collision_max_sample_rail_step_m': collision_sample_rail_m,
                           'collision_pairs_per_sample': len(self.collision.geom_model.collisionPairs),
                           'fk_position_max_m': max(poserr), 'fk_rotation_max_rad': max(roterr),
                           'fk_pass': bool(max(poserr) < 1e-4 and max(roterr) < 1e-3),
                           'continuity_pass': bool(continuity_pass)})
                ex['validated'] = all(ex[k] for k in ('limits_pass', 'collision_pass', 'fk_pass', 'continuity_pass'))
                if not ex['validated']:
                    failures = [k + ' failed' for k in ('limits_pass', 'collision_pass', 'fk_pass') if not ex[k]]
                    failures.extend(continuity_failures)
                    ex['reason'] = ', '.join(failures)
            result['exact'] = ex
        result['seconds']['total'] = time.perf_counter() - start
        result['seconds']['exact'] = result['seconds']['total'] - proxy_seconds
        return clean_json(result)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='NPZ: T_tcp_railbase, rail_m, q_seed')
    p.add_argument('--output', required=True)
    p.add_argument('--proxy-only', action='store_true')
    a = p.parse_args()
    data = np.load(a.input)
    b = CapabilityBridge()
    out = {'manifest': b.manifest(), 'result': b.query_path(data['T_tcp_railbase'], data['rail_m'], data['q_seed'],
                    exact=not a.proxy_only, provenance=str(Path(a.input).resolve()))}
    Path(a.output).write_text(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
