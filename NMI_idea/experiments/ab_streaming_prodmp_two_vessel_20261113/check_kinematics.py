"""Lift the first six streaming tapes to sampled fixed-rail robot kinematics."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CASE_COUNT = 6
SAMPLE_COUNT = 201
STEP_S = .02


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ik(cap, desired, previous, rail):
    """Track one SE(3) sample by damped Cartesian IK at fixed rail."""
    q = previous.copy()
    q[0] = rail
    for _ in range(36):
        actual = cap.fk(q)
        error_xyz = desired[:3, 3] - actual[:3, 3]
        error_angle = Rotation.from_matrix(desired[:3, :3] @ actual[:3, :3].T).as_rotvec()
        if np.linalg.norm(error_xyz) < 4.e-6 and np.linalg.norm(error_angle) < 4.e-5:
            break
        jac = cap.kin.jacobian(q)[:, 1:]
        step = np.linalg.solve(jac @ jac.T + 2.e-7 * np.eye(6), np.r_[error_xyz, error_angle])
        q[1:] += np.clip(jac.T @ step, -.13, .13)
    return q


def load_witness(bank):
    with lzma.open(bank, 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)['nodes']
    match = [node for node in rows if (node['site'], node['view'], node['realization']) == (3, 3, 'dird_top')]
    if len(match) != 1:
        raise ValueError('expected exactly one center-view dIRD witness at station 3')
    node = match[0]
    return node['id'], np.asarray(node['q'])


def summarize_path(cap, target, t, witness):
    q = witness.copy()
    joints, checks = [], []
    for desired in target:
        q = ik(cap, desired, q, witness[0])
        joints.append(q)
        checks.append(cap.check_q(q, desired))
    joints = np.asarray(joints)
    rate = np.abs(np.diff(joints, axis=0)) / np.diff(t)[:, None] / cap.kin.v_max
    invalid = [i for i, check in enumerate(checks) if not check['valid']]
    violating = np.flatnonzero((rate > 1. + 1.e-10).any(axis=1)).tolist()
    clearance = [check['min_self_collision_distance_m'] for check in checks]
    return {
        'passes': not (invalid or violating),
        'sample_count': len(checks),
        'invalid_kinematic_samples': invalid,
        'joint_limit_violations': [i for i, check in enumerate(checks) if not check['limits_pass']],
        'fk_violations': [i for i, check in enumerate(checks) if not check['fk_pass']],
        'self_collision_violations': [i for i, check in enumerate(checks) if not check['self_collision_pass']],
        'velocity_violation_intervals': violating,
        'minimum_self_clearance_mm': 1000. * min(clearance) if all(value is not None for value in clearance) else None,
        'minimum_joint_margin_rad_or_m': min(min(check['joint_limit_margin_by_coordinate']) for check in checks),
        'maximum_tracking_position_error_mm': 1000. * max(check['fk_position_error_m'] for check in checks),
        'maximum_tracking_rotation_error_deg': float(np.rad2deg(max(check['fk_rotation_error_rad'] for check in checks))),
        'peak_joint_rate_fraction': float(rate.max()),
        'initial_q': joints[0].tolist(),
        'final_q': joints[-1].tolist(),
        'preposition_max_arm_change_rad': float(np.abs(joints[0, 1:] - witness[1:]).max()),
    }


def check(repo, traces):
    sys.path[:0] = [str(repo), str(repo / 'rm75_control')]
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

    sections_path = HERE / 'SECTIONS.json'
    experiments = repo / 'NMI_idea' / 'experiments'
    atlas = experiments / 'ab_leg_dynamic_20261112' / 'DYNAMIC_GEOMETRY_FROZEN.npz'
    bank = experiments / 'ab_full_b_rollout_20261113' / 'motion' / 'BANK_MAIN.json.xz'
    sections = json.loads(sections_path.read_text(encoding='utf-8'))
    source_hashes = {path.name: sha256(path) for path in (traces, sections_path, atlas, bank)}
    if source_hashes[atlas.name] != sections['source_sha256']['dynamic_geometry_frozen_npz']:
        raise ValueError('dense sections and robot transform must come from the same frozen atlas')
    with np.load(atlas, allow_pickle=False) as data:
        robot_from_atlas = data['robot_from_atlas']
    with np.load(traces, allow_pickle=False) as data:
        t, s, arms, x, vx = (data[k] for k in ('t_s', 's_mm', 'arms', 'x_mm', 'vx_mm_s'))
    if (t.shape != (SAMPLE_COUNT,) or s.shape != t.shape or arms.shape != (3,) or len(set(arms)) != 3 or
            x.ndim != 3 or x.shape[0] != len(arms) or x.shape[1] < CASE_COUNT or x.shape[2] != len(t) or
            vx.shape != x.shape or not np.isfinite(x[:, :CASE_COUNT]).all() or
            not np.isfinite(vx[:, :CASE_COUNT]).all()):
        raise ValueError('expected three arms with at least six complete 201-step timed trajectories')
    np.testing.assert_allclose(t, np.arange(SAMPLE_COUNT) * STEP_S, atol=1.e-12, rtol=0.)
    grid = np.asarray(sections['s_mm'])
    if not np.isfinite(s).all() or s.min() < grid[0] - 1.e-12 or s.max() > grid[-1] + 1.e-12:
        raise ValueError('trace leaves the longitudinal section grid')
    s = np.clip(s, grid[0], grid[-1])
    origins = np.asarray(sections['origin_atlas_mm'])
    origin = np.column_stack([np.interp(s, grid, origins[:, axis]) for axis in range(3)])
    rotation = Slerp(grid, Rotation.from_matrix(sections['axes_atlas']))(s).as_matrix()
    node_id, witness = load_witness(bank)
    cap = CapabilityEvaluator(device='cpu')
    rows = []
    for arm, paths, velocities in zip(arms, x, vx, strict=True):
        for case in range(CASE_COUNT):
            target_atlas = np.repeat(np.eye(4)[None], len(t), axis=0)
            target_atlas[:, :3, :3] = rotation
            target_atlas[:, :3, 3] = (origin + paths[case, :, None] * rotation[:, :, 1]) / 1000.
            result = summarize_path(cap, robot_from_atlas @ target_atlas, t, witness)
            rows.append(dict(arm=str(arm), case=case, maximum_logged_lateral_rate_mm_s=float(np.abs(velocities[case]).max()),
                             **result))
    clearances = [row['minimum_self_clearance_mm'] for row in rows]
    return {
        'scope': 'Sampled ideal kinematic lift of unchanged saved streaming chart trajectories. The first state is independently prepositioned at each lateral start. Joint rates cover the 20 ms trajectory intervals after prepositioning. Collision checks cover configured robot and probe pairs only. This does not test body contact, acceleration/torque, controller tracking, or hardware.',
        'source_sha256': source_hashes,
        'robot_source_sha256': {Path(path).name: digest for path, digest in cap.manifest()['source_sha256'].items()},
        'case_indices': list(range(CASE_COUNT)), 'arms': arms.tolist(), 'sample_dt_s': STEP_S,
        'rail_m': float(witness[0]), 'witness_node_id': node_id, 'witness_q': witness.tolist(),
        'position_tolerance_mm': 1000. * cap.search_config.fk_position_tolerance_m,
        'rotation_tolerance_deg': float(np.rad2deg(cap.search_config.fk_rotation_tolerance_rad)),
        'clearance_required_mm': 1000. * cap.search_config.collision_margin_m,
        'paths_checked': len(rows), 'paths_passed': sum(row['passes'] for row in rows),
        'samples_checked': sum(row['sample_count'] for row in rows),
        'minimum_self_clearance_mm': min(clearances) if all(value is not None for value in clearances) else None,
        'maximum_tracking_position_error_mm': max(row['maximum_tracking_position_error_mm'] for row in rows),
        'maximum_tracking_rotation_error_deg': max(row['maximum_tracking_rotation_error_deg'] for row in rows),
        'peak_joint_rate_fraction': max(row['peak_joint_rate_fraction'] for row in rows),
        'paths': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.repo.resolve(), args.traces.resolve())
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Passed {report['paths_passed']}/{report['paths_checked']} paths over {report['samples_checked']} sampled states.")


if __name__ == '__main__':
    main()
