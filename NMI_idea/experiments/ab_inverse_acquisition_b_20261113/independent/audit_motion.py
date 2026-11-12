"""Independently replay generated endpoints and a spread of full sampled paths."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from audit_paths import load_bank


def audit(args):
    sys.path[:0] = [str(args.repo.resolve()), "/media/camp/EXT_DRIVE/RealUS_playground"]
    from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

    cap = CapabilityEvaluator(device="cpu")
    full, original = json.loads(load_bank(args.motion)), json.loads(load_bank(args.base))
    nodes = {n["id"]: n for n in full["nodes"]}
    old_edges = {(e["start"], e["goal"]): e for e in original["edges"]}
    new_edges = [e for e in full["edges"] if (e["start"], e["goal"]) not in old_edges]
    geo = args.repo / "NMI_idea/experiments/ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geo, allow_pickle=False) as raw:
        target_centers = raw["target_robot"][2, :, 3]
    endpoints = [n for n in full["nodes"] if n["id"] not in {r["id"] for r in original["nodes"]}]
    endpoint_error_mm = 0.
    endpoint_error_deg = 0.
    for node in endpoints:
        target = target_centers[node["site"]].astype(float)
        desired = target.copy()
        angle = np.deg2rad(node["tilt_deg"])
        desired[:, 1] = np.cos(angle) * target[:, 1] + np.sin(angle) * target[:, 2]
        desired[:, 2] = -np.sin(angle) * target[:, 1] + np.cos(angle) * target[:, 2]
        check = cap.check_q(node["q"], desired)
        assert check["valid"]
        np.testing.assert_allclose(cap.fk(node["q"])[:3, 3] * 1000., node["tip_xyz_mm"], atol=1.e-9, rtol=0)
        result = cap.score(desired[None, :, :], [node["q"][0]])
        assert result["proxy_pass"][0]
        np.testing.assert_allclose(result["raw_calibrated_margin"][0], node["dird_score"]["calibrated_margin"], atol=1.e-7, rtol=0)
        endpoint_error_mm = max(endpoint_error_mm, 1000. * check["fk_position_error_m"])
        endpoint_error_deg = max(endpoint_error_deg, np.degrees(check["fk_rotation_error_rad"]))
    # Replay a deterministic spread of edge positions. Include every checked
    # coarse diagonal, because these edges distinguish the two graph controls.
    coarse_ids = {n["id"] for n in original["nodes"]}
    pick = set(np.linspace(0, len(new_edges) - 1, min(len(new_edges), args.edges), dtype=int))
    max_rate = 0.
    min_clearance = np.inf
    n_samples = n_edges = 0
    for i, edge in enumerate(new_edges):
        a, b = edge["start"], edge["goal"]
        trace = edge["samples"]
        q, t = np.asarray(trace["q"]), np.asarray(trace["t_s"])
        np.testing.assert_allclose(q[[0, -1]], [nodes[a]["q"], nodes[b]["q"]], atol=1.e-10, rtol=0)
        assert t[0] == 0 and t[-1] == edge["duration_s"] and np.all(np.diff(t) > 0.)
        rate = (np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None] / cap.kin.v_max).max()
        limits = np.minimum(q - cap.kin.q_lower, cap.kin.q_upper - q).min()
        assert rate <= 1. + 1.e-10 and limits >= -1.e-8
        np.testing.assert_allclose([rate, limits], [edge["max_joint_rate_fraction"], edge["min_joint_limit_margin"]],
                                   rtol=0, atol=1.e-10)
        if i not in pick and not {a, b} <= coarse_ids:
            continue
        poses = np.asarray([cap.fk(row) for row in q])
        clear = min(cap.check_q(row)["min_self_collision_distance_m"] for row in q)
        xyz = poses[:, :3, 3] * 1000.
        relative = np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
        accumulated = [np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum(),
                       np.degrees(Rotation.from_matrix(relative).magnitude().sum()),
                       np.abs(np.diff(q[:, 0])).sum() * 1000., clear]
        np.testing.assert_allclose(trace["tip_xyz_mm"], xyz, atol=1.e-9, rtol=0)
        np.testing.assert_allclose(accumulated, [edge[k] for k in ("travel_mm", "turn_deg", "rail_mm", "min_self_collision_distance_m")],
                                   rtol=0, atol=1.e-9)
        assert clear >= cap.search_config.collision_margin_m
        min_clearance = min(min_clearance, clear)
        max_rate = max(max_rate, rate)
        n_edges += 1
        n_samples += len(q)
    return dict(new_endpoints=len(endpoints), max_fk_position_mm=endpoint_error_mm, max_fk_angle_deg=endpoint_error_deg,
                new_edges_rate_and_limits_checked=len(new_edges), replayed_edges=n_edges,
                replayed_samples=n_samples, min_replayed_clearance_mm=min_clearance * 1000.,
                max_replayed_rate_fraction=max_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "motion", "base"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--edges", type=int, default=80)
    print(json.dumps(audit(parser.parse_args()), indent=2))
