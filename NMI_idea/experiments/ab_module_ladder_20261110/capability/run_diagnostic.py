"""Freeze task-free public robot queries, then measure original dIRD ranking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from capability import CapabilityEvaluator, HERE, BASELINE, clean_json, sha256, relative_path

PUBLIC_Q_SOURCE = BASELINE/'repair_20261110'/'audit'/'phase2_native_subset_inputs.npz'


def save(path, x):
    path.write_text(json.dumps(clean_json(x), indent=2, allow_nan=False))


def freeze(evaluator):
    """Called before querying capability or observing benchmark labels."""
    out = HERE/'frozen_public'
    out.mkdir(exist_ok=False)
    source = np.load(PUBLIC_Q_SOURCE, allow_pickle=False)
    q0 = np.array(source['branch4_path0_q_seed'], dtype=float)
    # A fixed joint-space variation of a previously public q anchor, selected
    # without checking any dIRD score, task surface, IK result, or collision.
    q1 = q0 + np.array([.15, .6, .3, 0., .3, 0., 0., 0.])
    anchors = [q0, q1]
    offsets = [('identity', np.zeros(3), np.zeros(3))]
    for axis in range(3):
        for magnitude in [.02, .08, .16, .32]:
            for sign in [-1, 1]:
                delta = np.zeros(3); delta[axis] = sign*magnitude
                offsets.append((f't{axis}_{sign*magnitude:+.2f}m', delta, np.zeros(3)))
    for axis in range(3):
        for magnitude in [15., 45., 90.]:
            for sign in [-1, 1]:
                delta = np.zeros(3); delta[axis] = np.deg2rad(sign*magnitude)
                offsets.append((f'r{axis}_{sign*magnitude:+.0f}deg', np.zeros(3), delta))
    rows = []
    for anchor_id, q in enumerate(anchors):
        start = evaluator.fk(q)
        for name, translation, rotation in offsets:
            local = np.eye(4)
            local[:3,3] = translation
            local[:3,:3] = Rotation.from_rotvec(rotation).as_matrix()
            target = start@local
            rows.append(dict(id=f'a{anchor_id}_{name}', anchor_id=anchor_id,
                q_seed=q, T_world_tcp=target, T_body_tcp=local,
                body_translation_m=translation, body_rotvec_rad=rotation,
                geometry_distance=float(np.linalg.norm(translation)/.1+np.linalg.norm(rotation)/.5)))
    save(out/'candidates.json', rows)
    protocol = dict(schema='ab_public_geometry_diagnostic_frozen_v1',
        created_before_labels=True, public_q_source=str(PUBLIC_Q_SOURCE),
        public_q_source_sha256=sha256(PUBLIC_Q_SOURCE), public_q_key='branch4_path0_q_seed',
        anchors=anchors, second_anchor_rule='q0 + [.15m,.6rad,.3rad,0,.3rad,0,0,0]',
        candidate_rule='T_world_tcp = FK(q_anchor) @ SE3(body translation, body rotation); identity; axis translations ±[.02,.08,.16,.32]m; axis rotations ±[15,45,90]deg',
        candidate_count=len(rows), candidate_manifest_sha256=sha256(out/'candidates.json'),
        world_or_anatomy_data_used=False, source_manifest=evaluator.manifest(),
        exact_label='found a configuration passing original FK, URDF limits and all configured robot/probe self-collision pairs using frozen finite SRS search',
        unresolved_label='search_not_found; never impossible or no path',
        static_label_is_execution=False,
        ranking_policies=dict(dird='descending original raw proxy score', uniform='frozen manifest order',
            random='permutation with seeds 0..31', geometry_only='ascending displacement norm/.1m + rotation angle/.5rad'),
        query_budgets=[1,2,4,8,16,32], random_seeds=list(range(32)),
        primary_metric='number of found robot-valid endpoints within a fixed exact-query budget',
        secondary_metric='exact queries to first found endpoint',
        no_retuning_after_labels=True,
        native_subset_ids=['a0_t0_+0.02m','a0_t1_+0.02m'],
        native_rule='FK(seed) to fixed 2cm positive body-x/body-y paths, actual native scan profile .02m/s, .5s settle; record tiny jerk residual without repair')
    save(out/'protocol.json', protocol)
    return rows, protocol


def tests(evaluator, rows):
    row = rows[1]
    pose = np.array(row['T_world_tcp'])
    rail = row['q_seed'][0]
    analytic = np.array(evaluator.gradient(pose, rail)['gradient'])
    eps = [1e-6]*3+[1e-6]*3+[1e-6]
    fd = []
    for axis, h in enumerate(eps):
        plus, minus = pose.copy(), pose.copy()
        rp = rm = rail
        if axis < 3:
            plus[axis,3] += h; minus[axis,3] -= h
        elif axis < 6:
            delta = np.zeros(3); delta[axis-3] = h
            plus[:3,:3] = Rotation.from_rotvec(delta).as_matrix()@pose[:3,:3]
            minus[:3,:3] = Rotation.from_rotvec(-delta).as_matrix()@pose[:3,:3]
        else:
            rp += h; rm -= h
        fd.append((evaluator.score(plus,rp)['raw_score'][0]-evaluator.score(minus,rm)['raw_score'][0])/(2*h))
    fd = np.array(fd)
    G = np.eye(4); G[:3,:3] = Rotation.from_rotvec([.31,-.22,.19]).as_matrix(); G[:3,3]=[.7,-.3,.2]
    original = evaluator.score(pose,rail)['raw_score'][0]
    relabelled = evaluator.score(G@pose,rail,G)['raw_score'][0]
    moved = pose.copy(); moved[:3,3] += pose[:3,:3]@np.array([.08,0,0])
    moved_score = evaluator.score(moved,rail)['raw_score'][0]
    invalid = pose.copy(); invalid[0,0] += .01
    rejected = False
    try:
        evaluator.score(invalid,rail)
    except ValueError:
        rejected = True
    array = np.array([r['T_world_tcp'] for r in rows])
    rails = np.array([r['q_seed'][0] for r in rows])
    batched = np.array(evaluator.score(array,rails)['raw_score'])
    single = np.array([evaluator.score(t,r)['raw_score'][0] for t,r in zip(array,rails)])
    result = dict(gradient_analytic=analytic, gradient_central_fd=fd,
        gradient_max_abs_error=float(np.max(np.abs(analytic-fd))),
        gradient_pass=bool(np.allclose(analytic,fd,rtol=2e-4,atol=2e-5)),
        rotation_parameterization='left-multiplicative SO(3) rotation vector in world radians',
        invalid_rotation_rejected=rejected, common_se3_coordinate_relabel_error=abs(original-relabelled),
        common_se3_invariance_pass=abs(original-relabelled)<1e-9,
        body_only_displacement_original_raw=original, body_only_displacement_moved_raw=moved_score,
        body_only_displacement_changes_capability=abs(original-moved_score)>1e-8,
        batch_single_max_abs_error=float(np.max(np.abs(batched-single))),
        batch_single_pass=bool(np.allclose(batched,single,rtol=1e-10,atol=1e-10)),
        cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        gpu = CapabilityEvaluator('cuda')
        gpu_score = np.array(gpu.score(array,rails)['raw_score'])
        result.update(cuda_device=torch.cuda.get_device_name(0),
            cpu_gpu_max_abs_error=float(np.max(np.abs(batched-gpu_score))),
            cpu_gpu_pass=bool(np.allclose(batched,gpu_score,rtol=1e-7,atol=1e-8)))
    return result, batched


def benchmark(evaluator, rows, protocol, scores):
    started = time.perf_counter()
    labels = []
    for index, row in enumerate(rows):
        answer = evaluator._static_pose(row['T_world_tcp'],row['q_seed'][0],row['q_seed'])
        labels.append(dict(id=row['id'], raw_score=float(scores[index]), **answer))
        # Flush raw numerical decisions during the potentially long finite search.
        save(HERE/'endpoint_checks.json', labels)
    valid = np.array([r['found'] for r in labels],dtype=bool)
    orders = dict(dird=np.argsort(-scores,kind='stable'), uniform=np.arange(len(rows)),
        geometry_only=np.argsort([r['geometry_distance'] for r in rows],kind='stable'))
    records = {}
    for name, order in orders.items():
        found = np.flatnonzero(valid[order])
        records[name] = dict(order=[rows[i]['id'] for i in order],
            queries_to_first_found=None if not len(found) else int(found[0])+1,
            found_at_budget={str(b):int(valid[order[:b]].sum()) for b in protocol['query_budgets']})
    random = []
    for seed in protocol['random_seeds']:
        order = np.random.default_rng(seed).permutation(len(rows))
        found = np.flatnonzero(valid[order])
        random.append(dict(seed=seed, queries_to_first_found=None if not len(found) else int(found[0])+1,
            found_at_budget={str(b):int(valid[order[:b]].sum()) for b in protocol['query_budgets']}))
    records['random'] = dict(runs=random, mean_found_at_budget={str(b):float(np.mean([
        r['found_at_budget'][str(b)] for r in random])) for b in protocol['query_budgets']})
    gains = {str(b):records['dird']['found_at_budget'][str(b)]-records['random']['mean_found_at_budget'][str(b)]
        for b in protocol['query_budgets']}
    return dict(schema='ab_dird_exact_budget_endpoint_benchmark_v1',
        protocol_sha256=sha256(HERE/'frozen_public'/'protocol.json'),
        total_candidates=len(rows), found_robot_valid_endpoints=int(valid.sum()),
        unresolved_finite_search=int((~valid).sum()), unresolved_is_impossible=False,
        raw_decisions_file='endpoint_checks.json', policies=records, dird_minus_random_mean_found=gains,
        claim='Descriptive frozen robot-only endpoint-query efficiency; no task performance claim and no exhaustive negative labels',
        seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--reuse-frozen',action='store_true')
    parser.add_argument('--skip-benchmark',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    started=time.perf_counter()
    evaluator=CapabilityEvaluator()
    if args.reuse_frozen:
        rows=json.loads((HERE/'frozen_public'/'candidates.json').read_text())
        protocol=json.loads((HERE/'frozen_public'/'protocol.json').read_text())
        if sha256(HERE/'frozen_public'/'candidates.json')!=protocol['candidate_manifest_sha256']:
            raise ValueError('frozen candidate manifest changed')
    else:
        rows,protocol=freeze(evaluator)
    diagnostics,scores=tests(evaluator,rows)
    save(HERE/'tests.json',diagnostics)
    save(HERE/'proxy_scores.json',dict(candidate_ids=[r['id'] for r in rows],
        **evaluator.score(np.array([r['T_world_tcp'] for r in rows]),np.array([r['q_seed'][0] for r in rows]))))
    if not args.skip_benchmark:
        save(HERE/'benchmark.json',benchmark(evaluator,rows,protocol,scores))
    if args.native:
        native=[]
        for candidate_id in protocol['native_subset_ids']:
            row=next(r for r in rows if r['id']==candidate_id)
            q=np.array(row['q_seed'])
            path=relative_path(evaluator.fk(q),row['T_world_tcp'])
            outcome=evaluator.evaluate_path(path,q,exact=False,native=True)
            native.append(dict(id=candidate_id,**outcome))
            save(HERE/'native_subset.json',native)
    save(HERE/'run_summary.json',dict(wall_seconds=time.perf_counter()-started,
        tests_pass=all(diagnostics[k] for k in diagnostics if k.endswith('_pass')),
        protocol_sha256=sha256(HERE/'frozen_public'/'protocol.json'),
        code_sha256={p.name:sha256(p) for p in [HERE/'capability.py',Path(__file__)]}))
    print(json.dumps(clean_json(diagnostics)))


if __name__=='__main__':
    main()
