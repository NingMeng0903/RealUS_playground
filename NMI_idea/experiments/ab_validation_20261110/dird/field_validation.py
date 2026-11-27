"""Fresh collision-validated FK positives and far unreachable negatives.

Deliberately limited diagnostic; does not replace spatial/subject-held-out
production validation or claim accuracy near a reachability boundary.
"""
import json
import time
import numpy as np
import torch
from scipy.stats import rankdata
from adapter import CapabilityBridge
from benchmark import HERE, save


def main():
    b = CapabilityBridge()
    rng = np.random.default_rng(20261111)
    started = time.perf_counter()
    qs, poses, clearance = [], [], []
    attempts = 0
    while len(qs) < 64 and attempts < 2000:
        attempts += 1
        q = b.kin.q_lower + rng.random(8) * (b.kin.q_upper - b.kin.q_lower)
        b.collision.update(q)
        vals = np.array([r.min_distance for r in b.collision.geom_data.distanceResults])
        if not np.isfinite(vals).all() or min(vals) < b.spec.collision_security_margin_m:
            continue
        qs.append(q)
        poses.append(b.kin.fk_placement(q).homogeneous.copy())
        clearance.append(float(min(vals)))
    q, tcp = np.asarray(qs), np.asarray(poses)
    negative = tcp.copy()
    negative[:, 0, 3] += 3.
    rail = q[:, 0]
    with torch.no_grad():
        score = lambda T: b.score_tensor(torch.tensor(T, dtype=b.dtype), torch.tensor(rail, dtype=b.dtype)).numpy()
        pos, neg = score(tcp), score(negative)
    # Independent analytical SRS check on every far negative in its matched
    # seed branch; negatives are far outside the total link-length reach.
    exact = []
    for T, qi in zip(negative, q):
        result = b.query_path(np.repeat(T[None], 2, axis=0), np.full(2, qi[0]), qi)
        exact.append(result)
    threshold = b.calibration['safety_threshold']
    ranks = rankdata(np.r_[pos, neg])
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    save('fresh_field_validation.json', {
        'seed': 20261111, 'n_positive': len(q), 'draws_before_64_collision_valid': attempts,
        'positive_distribution': 'uniform all raw URDF joint limits, retained iff exact configured self-collision distance>=.01m; no dIRD selection',
        'negative_distribution': 'matched FK orientations and rail, TCP translated +3m in rail_base X; out-of-workspace diagnostic',
        'scope': 'fresh kinematic diagnostic, easy far negatives; not boundary calibration, not task-path feasibility and not ultrasound evidence',
        'proxy_threshold': threshold, 'positive_scores': pos, 'negative_scores': neg,
        'positive_recall_at_threshold': float(np.mean(pos >= threshold)),
        'negative_specificity_at_threshold': float(np.mean(neg < threshold)),
        'balanced_accuracy_on_this_easy_diagnostic': .5 * float(np.mean(pos >= threshold) + np.mean(neg < threshold)),
        'auroc_on_this_easy_diagnostic': float(auc),
        'negative_srs_lift_failures': sum(not r['exact']['lift_valid'] for r in exact),
        'elapsed_seconds': time.perf_counter() - started,
        'positive_q': q, 'positive_T_tcp_railbase': tcp,
        'positive_min_self_distance_m': clearance, 'negative_exact_queries': exact,
        'production_eval_report_existing_not_rerun': json.loads((b.checkpoint.parent / 'eval_report.json').read_text())})
    print('fresh n/draws', len(q), attempts)
    print('positive recall / far-negative specificity', float(np.mean(pos >= threshold)), float(np.mean(neg < threshold)))
    print('all far-negative SRS failed', sum(not r['exact']['lift_valid'] for r in exact), '/', len(exact))


if __name__ == '__main__':
    main()
