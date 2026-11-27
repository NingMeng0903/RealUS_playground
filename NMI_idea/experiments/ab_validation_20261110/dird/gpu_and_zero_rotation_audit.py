"""Real-checkpoint CPU/GPU parity and independent identity-exp derivative audit."""
import json
import time
import numpy as np
import torch
from adapter import CapabilityBridge, clean_json, sha256, SOURCE
from benchmark import HERE, audit_gradients, save
from ird_playground.ird.torch_kinematics import so3_exp


def zero_rotation_audit():
    records = []
    for value in (0., 1e-9, 2e-8, 1e-6):
        r = torch.tensor([value, 0., 0.], dtype=torch.float64, requires_grad=True)
        fun = lambda x: so3_exp(x)[2, 1]
        actual = torch.autograd.grad(fun(r), r)[0]
        fd = []
        for j in range(3):
            e = torch.zeros_like(r)
            e[j] = 1e-5
            fd.append(float((fun(r.detach() + e) - fun(r.detach() - e)) / 2e-5))
        # torch matrix_exp(skew(r)) is an independent differentiable reference.
        def skew(x):
            z = x[0] * 0.
            return torch.stack([z, -x[2], x[1], x[2], z, -x[0], -x[1], x[0], z]).reshape(3, 3)
        reference = torch.autograd.grad(torch.matrix_exp(skew(r))[2, 1], r)[0]
        records.append({'rotvec': r, 'production_ad': actual, 'production_fd_h1e5': fd,
                        'independent_matrix_exp_ad': reference})
    fixed = abs(records[0]['production_ad'][0].item() - 1.) < 1e-10
    save('zero_rotation_audit_postfix.json' if fixed else 'zero_rotation_defect.json', {
        'source': 'ird_playground/ird/torch_kinematics.py',
        'source_sha256': sha256(SOURCE / 'ird_playground/ird/torch_kinematics.py'),
        'zero_rotation_derivative_matches_reference': fixed,
        'finding': ('Taylor Rodrigues coefficients preserve the identity derivative.' if fixed else
            'At and near identity (<1e-8 rad), constant identity torch.where branch erases SO(3) derivative.'),
        'production_source_modified': fixed, 'records': records})


def main():
    zero_rotation_audit()
    nominal = json.loads((HERE / 'nominal_context.json').read_text())
    cpu = json.loads((HERE / 'gradients.json').read_text())
    b = CapabilityBridge(device='cuda', dtype=torch.float64)
    out = audit_gradients(b, nominal, filename='gradients_gpu.json')
    result = {'device': torch.cuda.get_device_name(), 'dtype': 'float64',
        'load_seconds': b.load_seconds,
        'cpu_gpu_score_max_abs_error': float(np.max(np.abs(np.array(cpu['scores']) - out['scores'].detach().cpu().numpy()))),
        'cpu_gpu_gradient_max_abs_error': float(np.max(np.abs(np.array(cpu['autograd']) - out['autograd'].cpu().numpy()))),
        'fd_p95_relative_error_h1e5': out['finite_difference_sweep'][2]['relative_error_p95'],
        'common_se3_score_max_abs_error': out['common_se3_score_max_abs_error'],
        'common_se3_gradient_max_abs_error': out['common_se3_gradient_max_abs_error'],
        'warm_query_timing_seconds': {}}
    paths = np.load(HERE / 'trajectory_paths.npz')
    tcp0 = torch.tensor(paths['initial_T_tcp_railbase'][0], dtype=b.dtype, device=b.device)
    rail0 = torch.tensor(paths['initial_rail_m'][0], dtype=b.dtype, device=b.device)
    for n in (1, 16, 256):
        tcp = tcp0[None].expand(n, -1, -1, -1)
        rail = rail0[None].expand(n, -1)
        with torch.no_grad():
            for _ in range(3):
                b.score_tensor(tcp, rail)
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                start = time.perf_counter()
                b.score_tensor(tcp, rail)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        result['warm_query_timing_seconds'][str(n) + '_paths_17_waypoints'] = {
            'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'n': len(times)}
    # Actual full-path score/rail gradient on a generated production spline.
    rail = rail0.clone().requires_grad_()
    value = -torch.logsumexp(-b.score_tensor(tcp0, rail), dim=-1)
    grad = torch.autograd.grad(value, rail)[0]
    fd = torch.zeros_like(grad)
    with torch.no_grad():
        for j in range(len(rail)):
            e = torch.zeros_like(rail)
            e[j] = 1e-5
            f = lambda rr: -torch.logsumexp(-b.score_tensor(tcp0, rr), dim=-1)
            fd[j] = (f(rail + e) - f(rail - e)) / 2e-5
    rel = (grad - fd).abs() / torch.maximum(grad.abs(), fd.abs()).clamp_min(1e-6)
    result['actual_spline_trajectory_rail_ad_fd'] = {'n_waypoints': len(rail),
        'absolute_error_max': (grad - fd).abs().max().item(),
        'relative_error_p95': torch.quantile(rel, .95).item(), 'ad': grad, 'fd': fd}
    save('gpu_audit.json', result)
    print(json.dumps(clean_json(result), indent=2))


if __name__ == '__main__':
    main()
