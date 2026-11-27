"""Fresh, fixed-seed model audit; generated data never enters training/selection."""
import json
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata
from common import HERE, BASELINE, SOURCE, CapabilityBridge, save, sha256
from ird_playground.ird.gpu_uniform_pose_gt import UniformPoseGtConfig, build_uniform_pose_gt, save_uniform_pose_gt
from ird_playground.neural.signed_field import ReachabilitySDF
from ird_playground.optimization.srs_trajectory_dp import _candidate_lattice


def metrics(score, truth, zero_bias, raw_threshold):
    truth = np.asarray(truth, dtype=bool)
    classified = score >= zero_bias
    accepted = score >= raw_threshold
    tpr = classified[truth].mean() if truth.any() else None
    tnr = (~classified[~truth]).mean() if (~truth).any() else None
    ranks = rankdata(score)
    auc = ((ranks[truth].sum() - truth.sum() * (truth.sum() + 1) / 2) / (truth.sum() * (~truth).sum())) if truth.any() and (~truth).any() else None
    return {'n': len(truth), 'n_positive': int(truth.sum()), 'n_negative': int((~truth).sum()),
        'balanced_accuracy_at_geometric_zero': None if tpr is None or tnr is None else .5 * (tpr + tnr),
        'positive_recall_at_geometric_zero': tpr, 'specificity_at_geometric_zero': tnr,
        'false_accepts_at_calibrated_threshold': int((accepted & ~truth).sum()),
        'false_accept_rate_at_calibrated_threshold': accepted[~truth].mean() if (~truth).any() else None,
        'reachable_recall_at_calibrated_threshold': accepted[truth].mean() if truth.any() else None,
        'auroc': auc, 'raw_threshold': raw_threshold, 'zero_bias': zero_bias,
        'scores': score, 'accepted': accepted, 'classified': classified}


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    b = CapabilityBridge()
    cfg = UniformPoseGtConfig(n_queries=512, batch_size=128, seed=20261130,
        device='cpu', robot_spec=str(SOURCE / 'configs/robot_probe45.yaml'), log_every_batches=1)
    arrays, metadata = build_uniform_pose_gt(cfg)
    save_uniform_pose_gt(HERE / 'fresh_global_audit.npz', arrays, metadata)
    # Local operating-region sample: independent pose/rotation perturbations,
    # no selection on scores or controller outcomes. Full fixed-branch5deg
    # lattice + exact physical collision gives a finite point-witness oracle.
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    T0 = np.asarray(nominal['T_tcp_railbase'])
    seed = np.asarray(nominal['q_seed'])
    rng = np.random.default_rng(20261131)
    n = 256
    tcp = np.repeat(T0[None], n, axis=0)
    tcp[:, :3, 3] += rng.normal(size=(n, 3)) * [.15, .10, .12]
    tcp[:, :3, :3] = tcp[:, :3, :3] @ Rotation.from_rotvec(rng.normal(size=(n, 3)) * .5).as_matrix()
    rail = np.full(n, seed[0])
    psi = np.deg2rad(np.repeat(np.arange(-180., 180., 5.)[None], n, axis=0))
    q, valid = _candidate_lattice(tcp, rail, psi, kin=b.kin, branch=0, euler_order='xyz')
    truth = np.zeros(n, dtype=bool)
    witness = np.full((n, 8), np.nan)
    minimum = np.full(n, np.nan)
    for i in range(n):
        for j in np.flatnonzero(valid[i]):
            b.collision.update(q[i, j])
            distance = np.array([r.min_distance for r in b.collision.geom_data.distanceResults])
            if len(distance) and np.isfinite(distance).all() and distance.min() >= .01:
                truth[i], witness[i], minimum[i] = True, q[i, j], distance.min()
                break
    np.savez_compressed(HERE / 'fresh_local_audit.npz', T_tcp_railbase=tcp, rail_m=rail,
        reachable=truth, q_witness=witness, minimum_self_distance_m=minimum,
        finite_ik_candidates=valid.sum(axis=1))
    results = {'scope': 'fresh development audit fixed before model scoring; no training/selection/tuning; selected-witness global oracle and finite-grid local witness oracle are separately reported',
        'global_seed': 20261130, 'local_seed': 20261131,
        'global_oracle': metadata, 'local_oracle': 'branch0, full5deg psi grid, any finite IK with all35 mesh distances>=.01m; negative means no witness on this finite lattice, not existential impossibility',
        'global_audit_sha256': sha256(HERE / 'fresh_global_audit.npz'),
        'local_audit_sha256': sha256(HERE / 'fresh_local_audit.npz'), 'models': {}}
    for name, checkpoint, calibration in [
        ('old', b.checkpoint, b.calibration_path),
        ('new', HERE / 'candidate/selected.pt', HERE / 'candidate/conformal.json')]:
        field = ReachabilitySDF.load(checkpoint, device='cpu', expected_robot=b.spec)
        conf = json.loads(calibration.read_text())
        global_score = field.score_np(arrays['flange_canonical'])
        with torch.no_grad():
            # Same physical J1/rail transform as the production bridge.
            from ird_playground.region.operator import base_from_rail_torch
            axis = base_from_rail_torch(torch.tensor(rail, dtype=torch.float32), torch.eye(4), b.axis0.float())
            local_score = field.score_world(torch.tensor(tcp, dtype=torch.float32), axis).numpy()
        # Calibration is fit to raw_score-zero_bias. Thus RAW acceptance uses
        # m_safe=zero_bias+safety_threshold, never safety_threshold alone.
        results['models'][name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'global': metrics(global_score, arrays['reachable'] > .5, conf['zero_bias'], conf['m_safe']),
            'local': metrics(local_score, truth, conf['zero_bias'], conf['m_safe'])}
    results['elapsed_seconds'] = time.perf_counter() - started
    save('independent_model_audit.json', results)
    for name, row in results['models'].items():
        print(name, {scope: {k: v for k, v in row[scope].items() if k in ['n_positive', 'n_negative', 'balanced_accuracy_at_geometric_zero', 'false_accept_rate_at_calibrated_threshold']} for scope in ['global', 'local']}, flush=True)


if __name__ == '__main__':
    main()
