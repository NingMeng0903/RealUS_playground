"""Freeze public, task-free endpoint witnesses for the paired stress experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / 'ab_module_ladder_20261110'
LAYOUT = LADDER / 'planning/native_regression_20261110/public_prior_layout/SELECTED.json'
sys.path.insert(0, str(LADDER / 'capability'))
from capability import CapabilityEvaluator


def public_poses():
    poses = [(0., y, yaw) for y in np.linspace(-.024, .024, 7) for yaw in (-.34, 0., .34)]
    poses += [(x, y, yaw) for x in (-.014, .014) for y in (-.016, .016) for yaw in (-.34, .34)]
    poses += [(0., -.032, 0.), (0., .032, 0.)]
    return np.unique(np.round(np.asarray(poses), 9), axis=0)


def world_poses(local, anchor):
    transforms = np.broadcast_to(np.eye(4), (len(local), 4, 4)).copy()
    transforms[:, :2, 3] = local[:, :2]
    transforms[:, :3, :3] = Rotation.from_euler('z', local[:, 2]).as_matrix()
    return anchor @ transforms


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_boundary_context(evaluator, poses, anchor, seed, initial):
    """Select one partial-witness placement using only robot geometry."""
    candidates = []
    # Ordered world-horizontal shifts at fixed rail. No observations or
    # trial outcomes enter this finite search.
    angles = np.deg2rad([0, 45, -45, 90, -90, 135, -135, 180])
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    for distance in np.arange(.08, .401, .01):
        for direction in directions:
            T = anchor.copy()
            offset = distance * direction
            T[:3, 3] += offset
            targets = world_poses(poses, T)
            center = evaluator._static_pose(targets[initial], seed[0], seed)
            attempted = dict(offset_m=offset.tolist(), center_found=center['found'])
            candidates.append(attempted)
            if not center['found']:
                continue
            public_seed = np.asarray(center['q_solution'])
            entries = [
                center if i == initial else evaluator._static_pose(target, seed[0], public_seed)
                for i, target in enumerate(targets)
            ]
            available = np.array([entry['found'] for entry in entries], dtype=bool)
            count = int(available.sum())
            has_two_yaws = all(np.any(available & (poses[:, 2] * sign > 0)) for sign in (-1, 1))
            has_two_sides = all(np.any(available & (poses[:, 1] * sign >= .008)) for sign in (-1, 1))
            attempted.update(available=count, has_two_yaws=has_two_yaws, has_two_sides=has_two_sides)
            if 12 <= count <= 26 and has_two_yaws and has_two_sides:
                return T, targets, entries, candidates
    raise RuntimeError('no placement with valid center and 12–26 witnessed endpoints across both sides and yaws')


def pack_context(evaluator, name, T, targets, entries, seed, initial, vmax):
    available = np.array([entry['found'] for entry in entries], dtype=bool)
    public_seed = np.asarray(entries[initial].get('q_solution', seed))
    q = np.array([entry.get('q_solution', public_seed) for entry in entries])
    joint_time = np.max(np.abs(q[:, None] - q[None, :])/vmax, axis=-1)
    proxy = evaluator.score(targets, seed[0])['raw_score']
    print(name, 'initial found', bool(available[initial]), 'endpoints', int(available.sum()), '/', len(targets), flush=True)
    return dict(id=name, available=available.tolist(), initial_index=initial,
                start_found=bool(available[initial]), T_world_body=T.tolist(),
                q_kin=[entry.get('q_solution') for entry in entries], joint_time_s=joint_time.tolist(),
                endpoint_records=entries, raw_dird_score_for_diagnostics_only=proxy)


def main(output, *, include_boundary=False):
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    layout = json.loads(LAYOUT.read_text())
    anchor = np.asarray(layout['T_world_body'], dtype=float)
    seed = np.asarray(layout['q_seed'], dtype=float)
    poses = public_poses()
    initial = int(np.flatnonzero(np.all(poses == 0., axis=1))[0])
    evaluator = CapabilityEvaluator()
    vmax = np.asarray(getattr(evaluator.kin, 'q_velocity_limits', [.4]+[2.512]*7), dtype=float)
    if vmax.shape != (8,) or not np.all(vmax > 0.):
        raise ValueError('expected positive speed limits for all eight coordinates')
    contexts = []
    for name, shift in [('prior_only_layout', 0.), ('prior_only_layout_body_z_minus_0.08m', -.08)]:
        T = anchor.copy()
        T[:3, 3] += shift * T[:3, 2]
        targets = world_poses(poses, T)
        center = evaluator._static_pose(targets[initial], seed[0], seed)
        if shift == 0.:
            check = evaluator.check_q(seed, targets[initial])
            if check['valid']:
                center = dict(found=True, q_solution=seed.tolist(), check=check,
                              status='existing_verified_layout_seed', seconds=0.)
        public_seed = np.asarray(center['q_solution']) if center['found'] else seed
        entries = []
        for i, target in enumerate(targets):
            entries.append(center if i == initial else evaluator._static_pose(target, seed[0], public_seed))
        contexts.append(pack_context(evaluator, name, T, targets, entries, seed, initial, vmax))
    selection = []
    if include_boundary:
        T, targets, entries, selection = find_boundary_context(evaluator, poses, anchor, seed, initial)
        contexts.append(pack_context(evaluator, 'prior_only_layout_partial_endpoint_witnesses',
                                     T, targets, entries, seed, initial, vmax))
    out = dict(scope='Public finite IK/FK/limits/self-collision at endpoints; transition times use a joint displacement estimate',
               poses=poses.tolist(), contexts=contexts, joint_velocity_limits=vmax.tolist(),
               robot=evaluator.manifest(), seconds=time.perf_counter()-start, boundary_layout_search=selection,
               input_sha256={str(p):sha(p) for p in [LAYOUT, Path(__file__), HERE/'PROTOCOL.json']})
    output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'CAPABILITY.json')
    parser.add_argument('--include-boundary', action='store_true')
    args = parser.parse_args()
    main(args.output, include_boundary=args.include_boundary)
