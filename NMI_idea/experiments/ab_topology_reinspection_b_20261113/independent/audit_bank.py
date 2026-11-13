"""Read archived sampled paths and recheck the public topology motion bank."""
import argparse
import hashlib
from itertools import product
import json
import lzma
from pathlib import Path
import sys

import numpy as np


def read_bank(root, filename):
    path = root / filename
    with lzma.open(path, "rt") as stream:
        return path, json.load(stream)


def recheck(root, kinematics=None):
    banks = (
        "ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz",
        "ab_inverse_acquisition_b_20261113/motion/BANK.json.xz",
    )
    max_error = {"sample_endpoint": 0.0, "joint_endpoint": 0.0, "tcp_mm": 0.0, "rail_mm": 0.0}
    records = {}
    costs = {}
    traces = {}
    previous_nodes = None
    for filename in banks:
        path, bank = read_bank(root, filename)
        nodes = {row["id"]: row for row in bank["nodes"] if row["id"] < 30}
        assert set(nodes) == set(range(30))
        assert all(node["endpoint_check"]["valid"] for node in nodes.values())
        if previous_nodes is not None:
            assert previous_nodes == nodes
        previous_nodes = nodes
        current = {}
        current_traces = {}
        for edge in bank["edges"]:
            start, goal = edge["start"], edge["goal"]
            if start not in nodes or goal not in nodes:
                continue
            key = (start, goal)
            assert key not in current
            samples = edge["samples"]
            q = np.asarray(samples["q"])
            tips = np.asarray(samples["tip_xyz_mm"])
            t = np.asarray(samples["t_s"])
            assert q.shape == (76, 8) and tips.shape == (76, 3)
            assert np.allclose(t, np.linspace(0.0, 1.5, 76), rtol=0, atol=1e-12)
            assert edge["duration_s"] == 1.5 and 0 <= edge["max_joint_rate_fraction"] <= 1
            assert edge["min_joint_limit_margin"] >= 0
            assert edge["min_self_collision_distance_m"] >= 0.01
            error_tip = float(np.abs(tips[[0, -1]] - [nodes[i]["tip_xyz_mm"] for i in key]).max())
            error_q = float(np.abs(q[[0, -1]] - [nodes[i]["q"] for i in key]).max())
            tcp = np.linalg.norm(np.diff(tips, axis=0), axis=-1).sum()
            rail = 1000 * np.abs(np.diff(q[:, 0])).sum()
            turn = edge["turn_deg"]
            assert turn >= 0
            for name, value in (
                ("sample_endpoint", error_tip), ("joint_endpoint", error_q),
                ("tcp_mm", abs(tcp - edge["travel_mm"])), ("rail_mm", abs(rail - edge["rail_mm"])),
            ):
                assert value <= 1e-8, (filename, key, name, value)
                max_error[name] = max(max_error[name], value)
            current[key] = [float(tcp), turn, float(rail), edge["duration_s"]]
            current_traces[key] = (q, t, tips, edge)
        assert set(costs).issubset(current)
        assert all(np.allclose(value, current[key], rtol=0, atol=1e-10) for key, value in costs.items())
        costs = current
        traces = current_traces
        records[filename] = {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "edges": len(current)}
    report = {"max_abs_error": max_error, "inputs": records}
    if kinematics is not None:
        from scipy.spatial.transform import Rotation

        max_error.update(fk_tcp_mm=0.0, fk_turn_deg=0.0, fk_rate=0.0)
        for key, (q, t, tips, edge) in traces.items():
            poses = np.asarray([kinematics.fk(row) for row in q])
            translation_error = np.abs(poses[:, :3, 3] * 1000.0 - tips).max()
            increments = np.swapaxes(poses[:-1, :3, :3], 1, 2) @ poses[1:, :3, :3]
            turn = np.degrees(Rotation.from_matrix(increments).magnitude().sum())
            rate = (np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None] / kinematics.kin.v_max).max()
            for name, error in (
                ("fk_tcp_mm", translation_error),
                ("fk_turn_deg", abs(turn - edge["turn_deg"])),
                ("fk_rate", abs(rate - edge["max_joint_rate_fraction"])),
            ):
                assert error <= 1e-8, (key, name, error)
                max_error[name] = max(max_error[name], float(error))
        report["kinematic_replay_samples"] = sum(len(trace[0]) for trace in traces.values())
    return previous_nodes, costs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original-repo", type=Path, help="Use the original runtime's robot model for sampled FK")
    args = parser.parse_args()
    kinematics = None
    if args.original_repo is not None:
        sys.path[:0] = [str(args.archive.resolve().parents[1]), str(args.original_repo.resolve())]
        from NMI_idea.experiments.ab_module_ladder_20261110.capability.capability import CapabilityEvaluator

        kinematics = CapabilityEvaluator(device="cpu")
    nodes, costs, report = recheck(args.archive, kinematics)
    report.update(node_count=len(nodes), edge_count=len(costs), sites=sorted({node["site"] for node in nodes.values()}))
    start = next(i for i, row in nodes.items() if row["site"] == 7 and row["view"] == 3 and row["realization"] == "dird_top")
    site_options = [[i for i, row in nodes.items() if row["site"] == site] for site in (7, 6, 5, 4, 3)]
    fixed_order_costs = []
    for route in product(*site_options):
        path = list(zip((start, *route[:-1]), route, strict=True))
        if all(edge in costs for edge in path):
            fixed_order_costs.append(np.sum([costs[edge] for edge in path], axis=0).tolist())
    report["prescan_fixed_order_route_bounds"] = {
        "count": len(fixed_order_costs),
        "fields": ["tcp_mm", "turn_deg", "rail_mm", "movement_s"],
        "minimum": np.min(fixed_order_costs, axis=0).tolist(),
        "maximum": np.max(fixed_order_costs, axis=0).tolist(),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
