"""Attribution using real production SRS lattices, exact mesh queries and gates.

This does not change the production solver. The bounded graph is an offline
diagnostic using its actual candidate IK implementation.
"""
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from common import HERE, BASELINE, CapabilityBridge, save
from ird_playground.optimization.srs_trajectory_dp import (
    _candidate_lattice, _first_order_dp, _second_order_dp, _wrap_pi,
)
from rm75_control.kinematics.srs_ik import branch_from_q, Q_LOWER, Q_UPPER


def bounded_graph(candidates, valid, seed):
    w, k, _ = candidates.shape
    cost = np.full((w, k), np.inf)
    parent = np.full((w, k), -1, dtype=int)
    scale = np.r_[.8, Q_UPPER - Q_LOWER]
    first = np.flatnonzero(valid[0])
    if len(first):
        delta = candidates[0, first] - seed
        allowed = (np.abs(delta[:, 0]) <= .04 + 1e-12) & (np.abs(delta[:, 1:]).max(axis=1) <= np.deg2rad(12.) + 1e-12)
        first = first[allowed]
        cost[0, first] = np.mean(((candidates[0, first] - seed) / scale) ** 2, axis=-1)
    reached = [len(first)]
    for i in range(1, w):
        prev, cur = np.flatnonzero(np.isfinite(cost[i - 1])), np.flatnonzero(valid[i])
        if not len(prev) or not len(cur):
            reached.extend([0] * (w - i))
            break
        delta = candidates[i, cur][None] - candidates[i - 1, prev][:, None]
        allowed = (np.abs(delta[..., 0]) <= .04 + 1e-12) & (np.abs(delta[..., 1:]).max(axis=-1) <= np.deg2rad(12.) + 1e-12)
        edge = np.mean((delta / scale) ** 2, axis=-1)
        total = np.where(allowed, cost[i - 1, prev, None] + edge, np.inf)
        best = np.argmin(total, axis=0)
        cost[i, cur] = total[best, np.arange(len(cur))]
        parent[i, cur] = prev[best]
        reached.append(int(np.isfinite(cost[i]).sum()))
    if not np.isfinite(cost[-1]).any():
        return None, reached
    idx = np.zeros(w, dtype=int)
    idx[-1] = np.argmin(cost[-1])
    for i in range(w - 1, 0, -1):
        idx[i - 1] = parent[i, idx[i]]
    return candidates[np.arange(w), idx], reached


def collision_mask(b, q, valid):
    mask = valid.copy()
    mind = np.full(valid.shape, np.nan)
    for i, j in zip(*np.nonzero(valid)):
        b.collision.update(q[i, j])
        d = np.array([r.min_distance for r in b.collision.geom_data.distanceResults])
        mind[i, j] = min(d) if len(d) else -np.inf
        mask[i, j] = bool(len(d) and np.isfinite(d).all() and d.min() >= .01)
    return mask, mind


def validate(b, q, tcp, seed):
    chain = np.vstack((seed, q))
    delta = np.diff(chain, axis=0)
    pos, rot, dist = [], [], []
    dense = [seed]
    for a, z in zip(chain[:-1], chain[1:]):
        n = max(1, int(np.ceil(np.abs(z[1:] - a[1:]).max() / np.deg2rad(1.))), int(np.ceil(abs(z[0] - a[0]) / .005)))
        dense.extend(a + u * (z - a) for u in np.linspace(0, 1, n + 1)[1:])
    for qi, T in zip(q, tcp):
        fk = b.kin.fk_placement(qi).homogeneous.copy()
        pos.append(np.linalg.norm(fk[:3, 3] - T[:3, 3]))
        rot.append(np.linalg.norm(Rotation.from_matrix(fk[:3, :3].T @ T[:3, :3]).as_rotvec()))
    for qi in dense:
        b.collision.update(qi)
        d = np.asarray([r.min_distance for r in b.collision.geom_data.distanceResults])
        dist.append(float(d.min()) if len(d) and np.isfinite(d).all() else -np.inf)
    dense = np.asarray(dense)
    margins = np.minimum(dense - b.kin.q_lower, b.kin.q_upper - dense)
    result = {'max_arm_step_including_departure_deg': float(np.rad2deg(np.abs(delta[:, 1:]).max())),
        'max_rail_step_including_departure_m': float(np.abs(delta[:, 0]).max()),
        'minimum_self_distance_m': min(dist), 'collision_samples': len(dense),
        'fk_position_max_m': max(pos), 'fk_rotation_max_rad': max(rot),
        'min_limit_margin_by_coordinate': margins.min(axis=0), 'q_ref': q,
        'point_collision_pass': bool(min(dist) >= .01)}
    result['validated'] = bool(result['max_arm_step_including_departure_deg'] <= 12. + 1e-8 and
        result['max_rail_step_including_departure_m'] <= .04 + 1e-12 and min(dist) >= .01 and
        margins.min() >= -1e-8 and max(pos) < 1e-4 and max(rot) < 1e-3)
    return result


def main():
    b = CapabilityBridge()
    data = np.load(BASELINE / 'dird/trajectory_paths.npz')
    seed = np.asarray(json.loads((BASELINE / 'dird/nominal_context.json').read_text())['q_seed'])
    old = json.loads((BASELINE / 'dird/trajectory_results.json').read_text())
    branch = int(branch_from_q(seed[1:]))
    results = []
    for method in ['no_dird', 'joint_guided']:
        for proposal, (tcp, rail) in enumerate(zip(data[method + '_T_tcp_railbase'], data[method + '_rail_m'])):
            start = time.perf_counter()
            w = len(tcp)
            coarse_psi = np.deg2rad(np.repeat(np.arange(-180., 180., 5.)[None], w, axis=0))
            coarse_q, coarse_valid = _candidate_lattice(tcp, rail, coarse_psi, kin=b.kin, branch=branch, euler_order='xyz')
            coarse_idx = _first_order_dp(coarse_q, coarse_valid, seed)
            coarse_sel = coarse_psi[np.arange(w), coarse_idx]
            refined_psi = _wrap_pi(coarse_sel[:, None] + np.deg2rad(np.arange(-5., 6., 1.))[None])
            refined_q, refined_valid = _candidate_lattice(tcp, rail, refined_psi, kin=b.kin, branch=branch, euler_order='xyz')
            production_idx = _second_order_dp(refined_q, refined_valid, seed)
            production_q = refined_q[np.arange(w), production_idx]
            refined_safe, refined_dist = collision_mask(b, refined_q, refined_valid)
            bounded_refined, reach_refined = bounded_graph(refined_q, refined_safe, seed)
            # Full 1-degree grid removes the production refinement corridor
            # restriction. It is still a finite fixed-branch IK discretization.
            fine_psi = np.deg2rad(np.repeat(np.arange(-180., 180., 1.)[None], w, axis=0))
            fine_q, fine_valid = _candidate_lattice(tcp, rail, fine_psi, kin=b.kin, branch=branch, euler_order='xyz')
            fine_safe, fine_dist = collision_mask(b, fine_q, fine_valid)
            bounded_fine, reach_fine = bounded_graph(fine_q, fine_safe, seed)
            record = {'method': method, 'proposal': proposal,
                'old_field_min_score': next(r['min_raw_ird'] for r in old['results'] if r['method'] == method and r['proposal'] == proposal),
                'branch': branch, 'production_selected': validate(b, production_q, tcp, seed),
                'refined_valid_candidates_per_waypoint': refined_valid.sum(axis=1),
                'refined_collision_safe_candidates_per_waypoint': refined_safe.sum(axis=1),
                'refined_hard_graph_reached_candidates': reach_refined,
                'refined_bounded_witness': None if bounded_refined is None else validate(b, bounded_refined, tcp, seed),
                'full_1deg_valid_candidates_per_waypoint': fine_valid.sum(axis=1),
                'full_1deg_collision_safe_candidates_per_waypoint': fine_safe.sum(axis=1),
                'full_1deg_hard_graph_reached_candidates': reach_fine,
                'full_1deg_bounded_witness': None if bounded_fine is None else validate(b, bounded_fine, tcp, seed),
                'seconds': time.perf_counter() - start}
            results.append(record)
            save('path_attribution.json', {'scope': 'fixed frozen paths, actual IK candidates; selected versus hard-transition finite graphs, sampled exact mesh checks; no source solver change', 'results': results})
            print(method, proposal, 'pointwise', int(np.min(fine_safe.sum(axis=1))),
                'prod', record['production_selected']['validated'],
                'bounded-refined', None if bounded_refined is None else record['refined_bounded_witness']['validated'],
                'bounded-full1deg', None if bounded_fine is None else record['full_1deg_bounded_witness']['validated'],
                'seconds', record['seconds'], flush=True)


if __name__ == '__main__':
    main()
