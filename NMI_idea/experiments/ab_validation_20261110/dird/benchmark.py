"""Bounded, deterministic audit of installed physical-probe45 dIRD operators."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from adapter import CapabilityBridge, SOURCE, ROBOT, clean_json, sha256
from ird_playground.ird.torch_kinematics import so3_exp
from ird_playground.optimization.differentiable_energy import (
    DifferentiableTrajectoryEnergy, TrajectoryEnergyConfig, cubic_bspline_matrices,
)
from ird_playground.region.operator import base_from_rail_torch

HERE = Path(__file__).resolve().parent
SEED = 20261110


def save(name, x):
    (HERE / name).write_text(json.dumps(clean_json(x), indent=2, allow_nan=False))


def pose_delta(T0, v):
    T = T0.expand(*v.shape[:-1], 4, 4).clone()
    T[..., :3, 3] = T0[..., :3, 3] + v[..., :3]
    T[..., :3, :3] = T0[..., :3, :3] @ so3_exp(v[..., 3:6])
    return T


def audit_gradients(b, nominal, filename='gradients.json'):
    rng = np.random.default_rng(SEED)
    T0 = torch.tensor(nominal['T_tcp_railbase'], dtype=b.dtype, device=b.device)
    q0 = np.asarray(nominal['q_seed'])
    v = torch.tensor(rng.normal(size=(24, 7)) * [0.025, .025, .025, .1, .1, .1, .04],
                     dtype=b.dtype, device=b.device, requires_grad=True)
    rail0 = torch.tensor(q0[0], dtype=b.dtype, device=b.device)
    def evaluate(x):
        return b.score_tensor(pose_delta(T0, x), rail0 + x[..., 6])
    y = evaluate(v)
    grad = torch.autograd.grad(y.sum(), v)[0]
    finite = []
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        cols = []
        with torch.no_grad():
            for j in range(7):
                e = torch.zeros_like(v)
                e[:, j] = h
                cols.append((evaluate(v + e) - evaluate(v - e)) / (2 * h))
        fd = torch.stack(cols, dim=-1)
        err = (grad - fd).abs()
        rel = err / torch.maximum(grad.abs(), fd.abs()).clamp_min(1e-6)
        finite.append({'h_m_or_rad': h, 'absolute_error_max': err.max().item(),
                       'relative_error_median': rel.median().item(),
                       'relative_error_p95': torch.quantile(rel.flatten(), .95).item(),
                       'relative_error_max': rel.max().item(), 'fd': fd})
    # A common arbitrary SE(3) change of coordinates moves both robot and task.
    H = torch.eye(4, dtype=b.dtype, device=b.device)
    H[:3, :3] = so3_exp(torch.tensor([.3, -.25, .6], dtype=b.dtype, device=b.device))
    H[:3, 3] = torch.tensor([.7, -.4, .2], dtype=b.dtype, device=b.device)
    vv = v.detach().clone().requires_grad_()
    moved = b.score_tensor(H @ pose_delta(T0, vv), rail0 + vv[..., 6], H)
    moved_grad = torch.autograd.grad(moved.sum(), vv)[0]
    with torch.no_grad():
        fixed_axis = base_from_rail_torch(rail0 + v[..., 6], torch.eye(4, dtype=b.dtype, device=b.device), b.axis0)
        wrong_score = b.field.score_world(pose_delta(T0, v), torch.eye(4, dtype=b.dtype, device=b.device))
        body_alone_score = b.field.score_world(H @ pose_delta(T0, v), fixed_axis)
    out = {'n': 24, 'seed': SEED, 'variables': ['tcp_dx_m', 'tcp_dy_m', 'tcp_dz_m',
           'local_rx_rad', 'local_ry_rad', 'local_rz_rad', 'rail_delta_m'],
           'scores': y.detach(), 'autograd': grad, 'finite_difference_sweep': finite,
           'finite_gradients': bool(torch.isfinite(grad).all()),
           'common_se3_score_max_abs_error': (y - moved).abs().max().item(),
           'common_se3_gradient_max_abs_error': (grad - moved_grad).abs().max().item(),
           'wrong_identity_axis_score_max_abs_difference': (y - wrong_score).abs().max().item(),
           'task_only_se3_score_max_abs_change': (y - body_alone_score).abs().max().item()}
    save(filename, out)
    return out


def trajectory_audit(b, nominal, steps, proposals):
    # This is the production ellipse decoder and continuous cubic B-spline
    # trajectory operator; its skin/vessel scene is synthetic geometry.
    sys.path.insert(0, str(SOURCE / 'experiments'))
    from ellipse_vessel_ird_demo import DemoConfig, ellipse_surface_tcp, nearest_theta_rad
    cfg = DemoConfig()
    nway, ncontrol = 17, 5
    s = np.linspace(0, 1, nway)
    B, V, A = cubic_bspline_matrices(s, ncontrol)
    baseline_theta = torch.full((nway,), nearest_theta_rad(cfg), dtype=b.dtype)
    path_y = torch.linspace(0., .06, nway, dtype=b.dtype)
    rail = float(nominal['q_seed'][0])
    canonical_first = ellipse_surface_tcp(baseline_theta[0:1], path_y[0:1], cfg)[0]
    registration = torch.tensor(nominal['T_tcp_railbase'], dtype=b.dtype) @ torch.linalg.inv(canonical_first)
    decoder = lambda t, y: registration @ ellipse_surface_tcp(t, y, cfg)
    tcp_base = decoder(baseline_theta[None], path_y[None])[0]
    center = tcp_base[:, :3, 3].clone()
    # Bounded moving obstacle intentionally intersects middle baseline samples.
    center[:, 0] += torch.tensor(.012 + .08 * (2 * s - 1) ** 8, dtype=b.dtype)
    config = TrajectoryEnergyConfig(safe_clearance=5., clearance_output_scale=100.,
        obstacle_radius_m=.018, obstacle_safe_margin_m=.003, obstacle_planning_margin_m=.005)
    energy = DifferentiableTrajectoryEnergy(b.field.model,
        basis=torch.tensor(B), velocity_basis=torch.tensor(V), curvature_basis=torch.tensor(A),
        baseline_theta=baseline_theta, path_y=path_y,
        baseline_rail=torch.full((nway,), rail, dtype=b.dtype), obstacle_centers=center,
        T_rail_axis0=b.axis0, pose_decoder=decoder,
        config=config).to(dtype=b.dtype)
    generator = torch.Generator().manual_seed(SEED)
    initial = .35 * torch.randn(proposals, ncontrol, 5, generator=generator, dtype=b.dtype)
    initial[..., 4] += np.arctanh((rail - .4) / .4)
    # Same medically prescribed endpoints for all three methods.
    initial[:, [0, -1], :4] = 0.
    initial[:, [0, -1], 4] = np.arctanh((rail - .4) / .4)
    probe = initial[:1].clone().requires_grad_()
    out = energy(probe)
    grad = torch.autograd.grad(out.energy.sum(), probe)[0]
    fd = torch.zeros_like(probe)
    with torch.no_grad():
        for j in range(probe.numel()):
            e = torch.zeros_like(probe).flatten()
            e[j] = 1e-5
            e = e.reshape_as(probe)
            fd.flatten()[j] = ((energy(probe + e).energy - energy(probe - e).energy) / 2e-5).item()
        moved_obstacle = center.clone()
        moved_obstacle[:, 0] += .2
        changed = energy(probe, obstacle_centers=moved_obstacle)
    grad_rel = (grad - fd).abs() / torch.maximum(grad.abs(), fd.abs()).clamp_min(1e-6)
    derivative = {'finite': bool(torch.isfinite(grad).all()), 'ad': grad, 'fd': fd,
        'relative_error_p95': torch.quantile(grad_rel.flatten(), .95).item(),
        'absolute_error_max': (grad - fd).abs().max().item(),
        'raw_ird_bitwise_equal_after_obstacle_move': torch.equal(out.raw_clearance, changed.raw_clearance),
        'obstacle_distance_max_change_m': (out.obstacle_signed_distance - changed.obstacle_signed_distance).abs().max().item()}
    methods = {'initial': initial.detach()}
    timing = {'initial': 0.}
    history = {}
    # Same controls/initialization/geometry/endpoints/evaluations. The no-dIRD
    # run still computes the real field to equalize query count, but drops its
    # term from optimization. This is an offline ablation, not production SQP.
    for method, include_dird in [('no_dird', False), ('joint_guided', True)]:
        controls = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([controls], lr=.035)
        hist = []
        start = time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad()
            z = energy(controls)
            reg = z.regrets
            # Shared image/contact proxy = prescribed ellipse pose deviation.
            loss = (reg['obstacle'] + .15 * reg['rule'] + .1 * reg['orientation_rule']
                    + .1 * reg['rail_rule'] + .05 * reg['continuity'] + .02 * reg['curvature'])
            if include_dird:
                loss = loss + .15 * reg['reachability_violation']
            loss.sum().backward()
            controls.grad[:, [0, -1]] = 0.
            optimizer.step()
            with torch.no_grad():
                controls[:, [0, -1]] = initial[:, [0, -1]]
            if step == 0 or step == steps - 1:
                hist.append({'step': step + 1, 'loss': loss.detach(),
                             'minimum_ird': z.raw_clearance.min(dim=-1).values.detach()})
        timing[method] = time.perf_counter() - start
        methods[method] = controls.detach()
        history[method] = hist
    records, npz = [], {'initial_controls': initial.numpy(), 's': s}
    seed = np.asarray(nominal['q_seed'])
    for method, controls in methods.items():
        with torch.no_grad():
            z = energy(controls)
        tcp = z.decoded.tcp.cpu().numpy()
        rr = z.decoded.rail.cpu().numpy()
        npz[method + '_controls'] = controls.numpy()
        npz[method + '_T_tcp_railbase'] = tcp
        npz[method + '_rail_m'] = rr
        for i in range(proposals):
            exact = b.query_path(tcp[i], rr[i], seed, provenance='production ellipse_surface_tcp + DifferentiableTrajectoryEnergy.decode, proposal=' + str(i) + ', method=' + method)
            records.append({'method': method, 'proposal': i,
                'min_raw_ird': float(z.raw_clearance[i].min()),
                'min_obstacle_radial_sdf_m': float(z.obstacle_signed_distance[i].min()),
                'obstacle_proxy_pass': bool(z.obstacle_signed_distance[i].min() >= config.obstacle_safe_margin_m),
                'tcp_path_length_m': float(np.linalg.norm(np.diff(tcp[i, :, :3, 3], axis=0), axis=-1).sum()),
                'mean_squared_pose_rule': float(z.regrets['rule'][i]),
                'capability': exact})
    np.savez(HERE / 'trajectory_paths.npz', **npz)
    result = {'seed': SEED, 'scope': 'Synthetic ellipse geometry, actual installed physical dIRD and actual robot IK/self-collision; no ultrasound/contact evidence',
              'n_proposals': proposals, 'n_waypoints': nway, 'n_control': ncontrol,
              'ellipse_registration': registration, 'path_span_m': .06,
              'development_adjustment': 'Default global ellipse fixed endpoint failed IK in all methods (retained under default_ellipse_failed). This separate local integration run rigidly registers first production ellipse pose to the public nominal FK; 6cm span, no test-seed selection.',
              'optimizer_steps_per_optimized_method': steps,
              'field_evaluations_per_optimized_method': steps * proposals * nway,
              'endpoint_policy': 'same first/last spline controls fixed',
              'loss': 'obstacle+.15*rule+.1*orientation_rule+.1*rail_rule+.05*continuity+.02*curvature; joint adds .15*reachability_violation',
              'config': vars(config), 'gradient': derivative, 'optimization_seconds': timing,
              'history': history, 'results': records}
    save('trajectory_results.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--proposals', type=int, default=6)
    args = p.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    start = time.perf_counter()
    b = CapabilityBridge()
    nominal = json.loads((HERE / 'nominal_context.json').read_text())
    files = sorted((SOURCE / 'ird_playground').rglob('*.py'))
    files += sorted((SOURCE / 'tests').glob('test_*.py'))
    files += [SOURCE / 'configs/robot_probe45.yaml', b.checkpoint, b.calibration_path,
              SOURCE / 'data/ird/current_robot_20261108/canonical.npz',
              ROBOT / 'rm75_control/kinematics/srs_ik.py',
              ROBOT / 'rm75_control/control/joint_admittance_8dof/model.py',
              ROBOT / 'rm75_control/control/joint_admittance_8dof/collision_model.py']
    packages = {}
    for name in ['numpy', 'scipy', 'torch', 'pin', 'coal', 'pytest']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'not installed under this distribution name'
    inventory = {'python': sys.version, 'executable': sys.executable,
        'platform': platform.platform(), 'packages': packages,
        'device': 'cpu', 'dtype': str(b.dtype), 'torch_num_threads': torch.get_num_threads(),
        'source_status': subprocess.run(['git', 'status', '--short'], cwd=SOURCE, text=True, capture_output=True).stdout,
        'source_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True, capture_output=True).stdout.strip(),
        'files_sha256': {str(f): sha256(f) for f in files}, 'manifest': b.manifest()}
    inventory['installed_dataset_hash_matches_checkpoint'] = inventory['files_sha256'][str(SOURCE / 'data/ird/current_robot_20261108/canonical.npz')] == b.field.meta['dataset_sha256']
    save('inventory.json', inventory)
    g = audit_gradients(b, nominal)
    print('pose/rail gradients', clean_json({k: v for k, v in g.items() if 'error' in k or k == 'finite_gradients'}), flush=True)
    t = trajectory_audit(b, nominal, args.steps, args.proposals)
    summary = {'total_seconds': time.perf_counter() - start,
        'installed_dataset_hash_matches_checkpoint': inventory['installed_dataset_hash_matches_checkpoint'],
        'pose_gradient_p95_relative_error_at_h1e5': g['finite_difference_sweep'][2]['relative_error_p95'],
        'common_se3_score_max_abs_error': g['common_se3_score_max_abs_error'],
        'common_se3_gradient_max_abs_error': g['common_se3_gradient_max_abs_error'],
        'trajectory_gradient_p95_relative_error': t['gradient']['relative_error_p95'],
        'raw_ird_separate_from_obstacle': t['gradient']['raw_ird_bitwise_equal_after_obstacle_move'],
        'methods': {}}
    for method in ['initial', 'no_dird', 'joint_guided']:
        rows = [r for r in t['results'] if r['method'] == method]
        summary['methods'][method] = {
            'n': len(rows), 'proxy_pass': sum(r['capability']['proxy_pass'] for r in rows),
            'exact_sampled_pass': sum(r['capability']['exact']['validated'] for r in rows),
            'obstacle_proxy_pass': sum(r['obstacle_proxy_pass'] for r in rows),
            'median_min_raw_ird': float(np.median([r['min_raw_ird'] for r in rows])),
            'minimum_raw_ird': min(r['min_raw_ird'] for r in rows),
            'optimization_seconds': t['optimization_seconds'][method],
            'exact_query_seconds': sum(r['capability']['seconds']['exact'] for r in rows),
            'failures': [r['capability']['exact']['reason'] for r in rows if not r['capability']['exact']['validated']]}
    save('summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
