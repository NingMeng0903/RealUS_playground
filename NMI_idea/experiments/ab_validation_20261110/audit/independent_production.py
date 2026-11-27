"""Small independent fresh numerical/real-module audit. Writes only audit/."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SOURCE = Path('/media/camp/EXT_DRIVE/RealUS_playground/ird_playground')
sys.path.insert(0, str(SOURCE))
from ird_playground.ird.torch_kinematics import so3_exp


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def skew(r):
    z = r[0] * 0
    return torch.stack([z, -r[2], r[1], r[2], z, -r[0], -r[1], r[0], z]).reshape(3, 3)


def main():
    current = SOURCE / 'ird_playground/ird/torch_kinematics.py'
    before = ROOT / 'dird/pre_fix/torch_kinematics.py'
    old = module('audit_old_kinematics', before)
    vectors = [np.zeros(3), np.array([1e-11, -2e-11, 3e-11]), np.array([1e-4, 0, 0]),
               np.array([1e-4 * (1 - 1e-7), 0, 0]), np.array([1e-4 * (1 + 1e-7), 0, 0]),
               np.array([.31, -.77, .49]), np.array([0, 0, np.pi]), np.array([2 * np.pi, 0, 0])]
    so3 = []
    for v in vectors:
        r = torch.tensor(v, dtype=torch.float64, requires_grad=True)
        truth = lambda x: torch.matrix_exp(skew(x))
        value = float(torch.max(torch.abs(so3_exp(r) - truth(r))).detach())
        jac = torch.autograd.functional.jacobian(so3_exp, r)
        ref = torch.autograd.functional.jacobian(truth, r)
        jacerr = float(torch.max(torch.abs(jac - ref)))
        assert value < 1e-13 and jacerr < 1e-12
        so3.append({'rotation_vector': v.tolist(), 'matrix_value_max_error': value, 'jacobian_max_error': jacerr})
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    old_derivative = float(torch.autograd.grad(old.so3_exp(zero)[2, 1], zero)[0][0])
    new_derivative = float(torch.autograd.grad(so3_exp(zero)[2, 1], zero)[0][0])
    assert old_derivative == 0. and new_derivative == 1.
    assert torch.autograd.gradgradcheck(so3_exp, (zero,), eps=1e-6, atol=1e-7, rtol=1e-5)

    motion = module('audit_motion_adapter', ROOT / 'motion/adapter.py')
    A = json.loads((ROOT / 'core/a_results.json').read_text())
    regenerated = []; lengths = []
    for x in [-1, -.5, 0, .5, 1]:
        for y in [-1, -.5, 0, .5, 1]:
            r = motion.generate_candidates([0, 0, 0], [.04*x, .04*y, 0], n=21, normal_xyz=[0, 0, 1],
                                           rotation=np.eye(3), lateral_offsets_m=(0.,), methods=('dmp',))[0]
            regenerated.append(r['xyz_m'][-1, :2] / .04); lengths.append(r['metrics']['path_length_m'])
    acterr = float(np.max(np.abs(np.asarray(regenerated) - np.asarray(A['action_grid']))))
    lengtherr = float(np.max(np.abs(np.asarray(lengths) - A['action_path_length_m'])))
    assert acterr < 1e-12 and lengtherr < 1e-12

    cap = module('audit_capability_adapter', ROOT / 'dird/adapter.py')
    bridge = cap.CapabilityBridge()
    graph = json.loads((ROOT / 'core/graph.json').read_text())
    queries = []
    choices = [('0', 'root/inspect2'), ('1', 'root/inspect2'), ('2', 'root/inspect2'),
               ('0', 'i0/inspect1'), ('1', 'root/terminal3'), ('2', 'i1/terminal0'),
               ('1', 'i2.i0/terminal0')]
    for context, eid in choices:
        edge = graph['contexts'][context]['edges'][eid]
        tcp = np.asarray(edge['T_tcp']); rail = np.asarray(edge['capability']['rail_m']); seed = np.asarray(edge['q_seed'])
        fresh = bridge.query_path(tcp, rail, seed, exact=True, provenance='independent audit fresh query from saved actual DMP path')
        scoreerr = float(np.max(np.abs(np.asarray(fresh['raw_score_per_waypoint']) - edge['capability']['raw_score_per_waypoint'])))
        assert scoreerr < 1e-10
        assert fresh['exact']['validated'] == edge['exact_validated']
        queries.append({'context': context, 'edge': eid, 'score_max_difference': scoreerr,
                        'exact_validated': fresh['exact']['validated'], 'reason': fresh['exact']['reason'],
                        'fresh_query_seconds': fresh['seconds']['total']})

    # Independent autograd versus FD of real field, including zero rotation.
    nominal = json.loads((ROOT / 'dird/nominal_context.json').read_text())
    T = torch.tensor(nominal['T_tcp_railbase'], dtype=torch.float64)
    rail = float(nominal['q_seed'][0])
    v = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    def energy(x):
        Ti = T.clone(); Ti[:3, :3] = T[:3, :3] @ so3_exp(x[3:6]); Ti[:3, 3] = T[:3, 3] + x[:3]
        return bridge.score_tensor(Ti[None], torch.tensor([rail], dtype=torch.float64) + x[6]).sum()
    derivative = torch.autograd.grad(energy(v), v)[0]
    fd = []
    for i in range(7):
        dx = torch.zeros(7, dtype=torch.float64); dx[i] = 1e-5
        fd.append(float(((energy(v.detach()+dx) - energy(v.detach()-dx)) / 2e-5).detach()))
    derr = float(np.max(np.abs(derivative.detach().numpy() - fd)))
    assert derr < 1e-5

    # Reuse reviewed theory functions without invoking their writer/main.
    theory = module('audit_theory_contract', ROOT / 'theory/check_contracts.py')
    contract = {'cpu': theory.tensor_contract('cpu'), 'canonical': theory.canonical_contract(),
                'risk': theory.risk_contract(), 'nonrigid': theory.nonrigid_contract()}
    if torch.cuda.is_available():
        contract['cuda'] = theory.tensor_contract('cuda')
    result = {'so3_checks': so3, 'identity_derivative_before': old_derivative, 'identity_derivative_after': new_derivative,
              'identity_second_derivative_check': True,
              'source_before_sha256': hashlib.sha256(before.read_bytes()).hexdigest(),
              'source_after_sha256': hashlib.sha256(current.read_bytes()).hexdigest(),
              'actual_DMP_action_max_error': acterr, 'actual_DMP_length_max_error': lengtherr,
              'fresh_actual_capability_queries': queries, 'real_field_zero_rotvec_gradient_max_abs_FD_error': derr,
              'theory_contract_reexecution': contract,
              'capability_adapter_sha256': hashlib.sha256((ROOT / 'dird/adapter.py').read_bytes()).hexdigest(),
              'graph_sha256': hashlib.sha256((ROOT / 'core/graph.json').read_bytes()).hexdigest()}
    (OUT / 'independent_production.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
