"""Public-pose capability checks using the original dIRD field and robot model.

No hardware API is instantiated. Static finite IK search is evidence of a found
configuration, not a proof that an unfound configuration is impossible. Native
q_out, when requested, is the actual controller's offline ideal-plant exit state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[1] / 'ab_validation_20261110'
ADAPTER = BASELINE / 'dird' / 'adapter.py'
_spec = importlib.util.spec_from_file_location('ab_original_capability_adapter', ADAPTER)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
CapabilityBridge, clean_json, sha256 = _module.CapabilityBridge, _module.clean_json, _module.sha256


def validate_poses(value):
    x = np.asarray(value, dtype=np.float64)
    if x.shape == (4, 4):
        x = x[None]
    if x.ndim != 3 or x.shape[1:] != (4, 4) or not len(x) or not np.isfinite(x).all():
        raise ValueError('pose must be finite (4,4) or nonempty (N,4,4)')
    if not np.allclose(x[:, 3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('invalid homogeneous bottom row')
    r = x[:, :3, :3]
    if not (np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-7)
            and np.allclose(np.linalg.det(r), 1, atol=1e-7)):
        raise ValueError('rotation must be proper orthonormal SO(3)')
    return x


def relative_path(T_start, T_end, n=5):
    """Linear translation and shortest SO(3) interpolation, both typed in SE(3)."""
    a, b = validate_poses(np.stack((T_start, T_end)))
    u = np.linspace(0., 1., max(2, int(n)))
    out = np.repeat(np.eye(4)[None], len(u), axis=0)
    out[:, :3, 3] = a[:3, 3] + u[:, None] * (b[:3, 3] - a[:3, 3])
    out[:, :3, :3] = Slerp([0., 1.], Rotation.from_matrix(np.stack((a[:3, :3], b[:3, :3]))))(u).as_matrix()
    return out


@dataclass(frozen=True)
class SearchConfig:
    psi_step_deg: float = 15.
    branches: tuple = tuple(range(8))
    include_seed_swivel: bool = True
    collision_margin_m: float = .01
    fk_position_tolerance_m: float = 1e-4
    fk_rotation_tolerance_rad: float = 1e-3
    collision_segment_step_deg: float = 2.
    collision_segment_rail_m: float = .005


class CapabilityEvaluator:
    """Reusable robot capability evaluator, with separate proxy/static/native data."""
    def __init__(self, device='cpu', search_config=None):
        self.bridge = CapabilityBridge(device=device, dtype=torch.float64)
        self.search_config = search_config or SearchConfig()
        self._native = None

    @property
    def kin(self):
        return self.bridge.kin

    def fk(self, q, T_world_rail=None):
        out = self.kin.fk_placement(np.asarray(q, dtype=float)).homogeneous.copy()
        return out if T_world_rail is None else validate_poses(T_world_rail)[0] @ out

    def manifest(self):
        from rm75_control.kinematics import srs_ik
        from rm75_control.control.joint_admittance_8dof import model, collision_model
        sources = [Path(__file__), ADAPTER, Path(srs_ik.__file__), Path(model.__file__),
                   Path(collision_model.__file__), Path(self.bridge.spec.kinematics_urdf),
                   Path(self.bridge.spec.collision_urdf), Path(self.bridge.spec.collision_pairs)]
        return clean_json(dict(schema='ab_capability_evaluator_v1', original_dird=self.bridge.manifest(),
            source_sha256={str(p): sha256(p) for p in sources}, search_config=asdict(self.search_config),
            pose_convention='T_world_tcp; T_world_rail defaults to identity; q is rail_base URDF coordinates',
            twist_convention='gradient [world translation metres, left world rotation radians, raw rail metres]',
            static_statuses=['found_valid_configuration', 'search_not_found'],
            exact_scope='FK/URDF limits/configured robot-probe self-collision at endpoint; finite IK search; no body or environment collision, contact, dynamics, or exhaustive infeasibility certificate',
            q_solution='kinematic configuration found by finite static search, never executed state',
            q_out='only returned for native replay; actual q_send integrated by offline ideal plant',
            score_is_safety_certificate=False, runtime_threads=1))

    def score_tensor(self, poses, rail, T_world_rail=None):
        """Differentiable batched original dIRD score; no task reward/cost."""
        return self.bridge.score_tensor(poses, rail, T_world_rail)

    def score(self, poses, rail, T_world_rail=None):
        started = time.perf_counter()
        x = validate_poses(poses)
        rail = np.broadcast_to(np.asarray(rail, dtype=float), (len(x),)).copy()
        if not np.isfinite(rail).all():
            raise ValueError('nonfinite rail')
        b = self.bridge
        with torch.no_grad():
            raw = b.score_tensor(torch.as_tensor(x, dtype=b.dtype, device=b.device),
                torch.as_tensor(rail, dtype=b.dtype, device=b.device),
                None if T_world_rail is None else torch.as_tensor(validate_poses(T_world_rail)[0], dtype=b.dtype, device=b.device)).cpu().numpy()
        threshold = b.calibration['m_safe']
        return clean_json(dict(raw_score=raw, raw_calibrated_margin=raw-threshold,
            raw_threshold=threshold, proxy_pass=raw >= threshold,
            threshold_formula='raw_score >= m_safe = zero_bias + safety_threshold',
            score_is_safety_certificate=False, seconds=time.perf_counter()-started))

    def gradient(self, pose, rail, T_world_rail=None):
        """Autograd tangent derivative, left-multiplicative SO(3), no Euler rates."""
        x = validate_poses(pose)
        if len(x) != 1:
            raise ValueError('gradient expects one pose')
        b = self.bridge
        T = torch.tensor(x, dtype=b.dtype, device=b.device, requires_grad=True)
        r = torch.tensor([rail], dtype=b.dtype, device=b.device, requires_grad=True)
        world = None if T_world_rail is None else torch.tensor(validate_poses(T_world_rail)[0], dtype=b.dtype, device=b.device)
        value = self.score_tensor(T, r, world).sum()
        gt, gr = torch.autograd.grad(value, (T, r))
        mat, R = gt[0, :3, :3], T[0, :3, :3]
        generators = T.new_tensor([[[0,0,0],[0,0,-1],[0,1,0]],
                                  [[0,0,1],[0,0,0],[-1,0,0]],
                                  [[0,-1,0],[1,0,0],[0,0,0]]])
        rotation = torch.stack([(mat * (k @ R)).sum() for k in generators])
        return clean_json(dict(raw_score=value, gradient=np.r_[gt[0,:3,3].detach().cpu().numpy(),
            rotation.detach().cpu().numpy(), gr.detach().cpu().numpy()],
            coordinates=['tx_m','ty_m','tz_m','left_rx_rad','left_ry_rad','left_rz_rad','rail_m']))

    def check_q(self, q, target=None):
        q = np.asarray(q, dtype=float)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError('q must be finite (8,)')
        cfg = self.search_config
        margin = np.minimum(q-self.kin.q_lower, self.kin.q_upper-q)
        actual = self.fk(q)
        p_err = r_err = 0.
        if target is not None:
            target = validate_poses(target)[0]
            p_err = float(np.linalg.norm(actual[:3,3]-target[:3,3]))
            r_err = float(Rotation.from_matrix(actual[:3,:3].T @ target[:3,:3]).magnitude())
        self.bridge.collision.update(q)
        distances = np.array([x.min_distance for x in self.bridge.collision.geom_data.distanceResults])
        finite = bool(len(distances) and np.isfinite(distances).all())
        minimum = float(distances.min()) if finite else None
        limits_pass = bool(margin.min() >= -1e-8)
        fk_pass = bool(p_err <= cfg.fk_position_tolerance_m and r_err <= cfg.fk_rotation_tolerance_rad)
        collision_pass = bool(finite and minimum >= cfg.collision_margin_m)
        return clean_json(dict(valid=limits_pass and fk_pass and collision_pass,
            limits_pass=limits_pass, fk_pass=fk_pass, self_collision_pass=collision_pass,
            joint_limit_margin_by_coordinate=margin, fk_position_error_m=p_err,
            fk_rotation_error_rad=r_err, min_self_collision_distance_m=minimum,
            self_collision_required_m=cfg.collision_margin_m, pair_count=len(distances)))

    def _static_pose(self, pose, rail, q_seed):
        from rm75_control.kinematics.srs_ik import (srs_ik, flange_tcp_from_kin,
            psi_from_q, shoulder_y_from_q_rail)
        started = time.perf_counter()
        cfg = self.search_config
        target = validate_poses(pose)[0]
        seed = np.asarray(q_seed, dtype=float)
        if seed.shape != (8,) or not np.isfinite(seed).all():
            raise ValueError('q_seed must be finite (8,)')
        pose6 = np.r_[target[:3,3], Rotation.from_matrix(target[:3,:3]).as_euler('xyz')]
        grid = np.deg2rad(np.arange(-180., 180., cfg.psi_step_deg))
        psi = np.unique(np.r_[grid, psi_from_q(seed)] if cfg.include_seed_swivel else grid)
        Rft, tft = flange_tcp_from_kin(self.kin)
        attempted, returned, inspected = 0, [], []
        # All finite search candidates are collected; exact checks are ordered
        # by continuous joint distance to the public seed, not dIRD score.
        for branch in cfg.branches:
            for swivel in psi:
                attempted += 1
                arm = srs_ik(pose6, float(swivel), int(branch),
                    y_rail=shoulder_y_from_q_rail(float(rail)),
                    R_flange_tcp=Rft, t_flange_tcp=tft)
                if arm is not None:
                    returned.append((np.r_[rail, arm], int(branch), float(swivel)))
        scale = np.maximum(self.kin.q_upper-self.kin.q_lower, 1e-6)
        returned.sort(key=lambda row: float(np.linalg.norm((row[0]-seed)/scale)))
        selected = None
        for q, branch, swivel in returned:
            check = self.check_q(q, target)
            inspected.append(dict(branch=branch, psi_rad=swivel, check=check))
            if check['valid']:
                selected = dict(q_solution=q, branch=branch, psi_rad=swivel, check=check)
                break
        return clean_json(dict(status='found_valid_configuration' if selected else 'search_not_found',
            found=selected is not None, **(selected or {}), search=dict(
                branches=list(cfg.branches), psi_values_rad=psi, rail_values_m=[float(rail)],
                ik_calls=attempted, ik_candidates_returned=len(returned),
                robot_configurations_checked=len(inspected), inspected=inspected,
                stop_rule='first exactly checked valid configuration in normalized seed-distance order',
                exhaustive_infeasibility=False, no_path_claim=False),
            seconds=time.perf_counter()-started))

    def evaluate_pose(self, pose, q_seed, rail=None, *, exact=True, native=False,
                      T_world_rail=None, native_options=None):
        return self.evaluate_path(validate_poses(pose), q_seed, rail, exact=exact,
            native=native, T_world_rail=T_world_rail, native_options=native_options)

    def evaluate_path(self, poses, q_seed, rail=None, *, exact=True, native=False,
                      T_world_rail=None, native_options=None):
        started = time.perf_counter()
        x = validate_poses(poses)
        seed = np.asarray(q_seed, dtype=float)
        if seed.shape != (8,) or not np.isfinite(seed).all():
            raise ValueError('q_seed must be finite (8,)')
        r = np.broadcast_to(seed[0] if rail is None else rail, (len(x),)).copy()
        root = np.eye(4) if T_world_rail is None else validate_poses(T_world_rail)[0]
        local = np.linalg.inv(root)[None] @ x
        proxy = self.score(x, r, root)
        result = dict(schema='ab_capability_query_v1', proxy=proxy, static=None, native=None,
            q_seed=seed, q_out=None, q_out_source=None, pose_count=len(x),
            scope='public robot-only capability; no task reward or anatomy data')
        if exact:
            checks, solutions, q = [], [], seed.copy()
            for target, ri in zip(local, r):
                row = self._static_pose(target, float(ri), q)
                checks.append(row)
                if row['found']:
                    q = np.array(row['q_solution']); solutions.append(q.copy())
                else:
                    break
            segment_checks = []
            if len(solutions) == len(x):
                chain = np.vstack((seed, solutions))
                cfg = self.search_config
                for a, b in zip(chain[:-1], chain[1:]):
                    count = max(1, int(np.ceil(np.max(np.abs(b[1:]-a[1:]))/np.deg2rad(cfg.collision_segment_step_deg))),
                        int(np.ceil(abs(b[0]-a[0])/cfg.collision_segment_rail_m)))
                    checks_here = [self.check_q(a+u*(b-a)) for u in np.linspace(0,1,count+1)]
                    segment_checks.append(dict(sample_count=len(checks_here),
                        passed=all(z['valid'] for z in checks_here),
                        min_self_collision_distance_m=min(z['min_self_collision_distance_m'] for z in checks_here)))
            result['static'] = dict(endpoint_found=len(solutions)==len(x), waypoint_checks=checks,
                q_solution=None if len(solutions)!=len(x) else solutions[-1],
                q_path_solutions=solutions, sampled_joint_segment_checks=segment_checks,
                sampled_joint_connection_pass=bool(segment_checks and all(z['passed'] for z in segment_checks)),
                failed_search_is_infeasibility_certificate=False,
                note='joint interpolation is only a sampled connection diagnostic; it need not track the Cartesian path')
        if native:
            if self._native is None:
                sys.path.insert(0, str(BASELINE/'repair_20261110'/'qp_cross_track_fix'))
                from scan_replay import ScanReplayBridge
                self._native = ScanReplayBridge()
            options = dict(speed_m_s=.02, settle_s=.5, return_trace=False, return_state=True)
            options.update(native_options or {})
            native_path = local
            if len(local) == 1:
                native_path = relative_path(self.fk(seed), local[0])
            result['native'] = self._native.query_pose_path(native_path, seed,
                provenance='frozen public capability query', **options)
            result['q_out'] = result['native']['q_exit']
            result['q_out_source'] = 'actual native controller q_send integrated in offline ideal measured-q plant'
        result['seconds'] = time.perf_counter()-started
        return clean_json(result)


def evaluate_pose(pose, q_seed, rail=None, **kwargs):
    """Convenience one-shot API; reuse CapabilityEvaluator for planner loops."""
    return CapabilityEvaluator().evaluate_pose(pose, q_seed, rail, **kwargs)

