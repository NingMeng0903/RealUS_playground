"""Compose the tested ProDMP reference with production harmonic material frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from motion import ProDMPReference


HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "ab_theory_recheck_20261110/transport_response_bridge"
sys.path.insert(0, str(BRIDGE))
from run_bridge import ASSET, POSES, RAW_HARMONIC, cell_geometry, frame_from_transport, mats, unit


def homogeneous(p, R):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def max_abs(values):
    return float(np.max(np.abs(values)))


def check_references():
    rng = np.random.default_rng(721120)
    raw = dict(np.load(RAW_HARMONIC))
    points = 32
    tau = .9
    start, goal = np.zeros(3), np.array([.004, -.003, .12])
    times = np.linspace(0, tau, 9)
    baseline = ProDMPReference(start, goal, np.eye(4), duration_s=tau)
    states = [baseline.local_position_velocity(float(t)) for t in times]
    local_pos = np.array([np.r_[z[:2], 0.] for z, _ in states])
    local_vel = np.array([np.r_[v[:2], 0., 0., 0., v[2]] for _, v in states])
    local_R = Rotation.from_euler("z", [z[2] for z, _ in states]).as_matrix()
    selected, rows, negative = {}, [], {"frozen_frame_endpoint_mm": [], "wrong_normal_deg": []}
    inputs = [RAW_HARMONIC, BRIDGE / "run_bridge.py", HERE / "motion.py", Path(__file__)]
    for side in ("right", "left"):
        atlas, ids, bary, cells, X0, _, t0, n0 = cell_geometry(side, raw)
        inputs.append(ASSET / f"atlas_{side}.npz")
        sample = rng.choice(len(ids), points, replace=False)
        selected[side] = ids[sample].tolist()
        _, _, R0 = frame_from_transport(np.broadcast_to(np.eye(3), (len(ids), 3, 3)), t0, n0)
        for knee in POSES:
            source = ASSET / f"{side}_pose_{knee:02d}.npz"
            inputs.append(source)
            with np.load(source) as posed:
                vp = posed["vertices"][cells[sample]].astype(float)
            F = mats(vp) @ np.linalg.inv(X0[sample])
            center = np.einsum("ni,nij->nj", bary[sample], vp)
            t, n, frame = frame_from_transport(F, t0[sample], n0[sample])
            if knee > 0:
                dv = np.einsum("nij,j->ni", frame - R0[sample], local_pos[-1])
                negative["frozen_frame_endpoint_mm"].extend(1000 * np.linalg.norm(dv, axis=1))
                wrong = unit(np.einsum("nij,nj->ni", F, n0[sample]))
                negative["wrong_normal_deg"].extend(np.rad2deg(np.arccos(
                    np.clip(np.einsum("ni,ni->n", n, wrong), -1., 1.)
                )))
            errors = []
            for a, E in zip(center, frame):
                anchor = homogeneous(a, E)
                g = homogeneous(rng.uniform(-.5, .5, 3), Rotation.random(random_state=rng).as_matrix())
                ref = ProDMPReference(start, goal, anchor, duration_s=tau)
                moved = ProDMPReference(start, goal, g @ anchor, duration_s=tau)
                pose, vel = map(np.asarray, zip(*(ref.sample_world(float(t)) for t in times)))
                pose_g, vel_g = map(np.asarray, zip(*(moved.sample_world(float(t)) for t in times)))
                world_R = Rotation.from_euler("xyz", pose[:, 3:]).as_matrix()
                world_Rg = Rotation.from_euler("xyz", pose_g[:, 3:]).as_matrix()
                gR = g[:3, :3]
                errors.append([
                    max_abs((pose[:, :3] - a) @ E - local_pos),
                    max_abs(E.T @ world_R - local_R),
                    max_abs(np.c_[vel[:, :3] @ E, vel[:, 3:] @ E] - local_vel),
                    max_abs(pose_g[:, :3] - (pose[:, :3] @ gR.T + g[:3, 3])),
                    max_abs(world_Rg - gR @ world_R),
                    max_abs(vel_g - np.c_[vel[:, :3] @ gR.T, vel[:, 3:] @ gR.T]),
                ])
            rows.append({
                "side": side, "knee_deg": knee, "addresses": points,
                "minimum_F_determinant": float(np.linalg.det(F).min()),
                "max_tangent_normal_dot": max_abs(np.einsum("ni,ni->n", t, n)),
                "errors_max": np.asarray(errors).max(axis=0).tolist(),
            })
    if min(row["minimum_F_determinant"] for row in rows) <= 0:
        raise ValueError("chosen production cells have a nonpositive deformation determinant")
    if max(max(row["errors_max"]) for row in rows) > 1e-11:
        raise ValueError("ProDMP decoding/transport contract failed")
    return {
        "scope": "Actual saved harmonic cells and seven SMPL-X poses per side. Fixed local tangent frame per trajectory, not motion within a deforming frame, contact, or scan execution. Interior samples test the material interface only.",
        "seed": 721120,
        "cell_ids": selected,
        "reference_duration_s": tau,
        "sample_times_s": times.tolist(),
        "local_goal_m_m_rad": goal.tolist(),
        "error_columns": ["local_position_m", "local_rotation_matrix", "local_twist",
                          "common_SE3_position_m", "common_SE3_rotation_matrix", "common_SE3_twist"],
        "by_pose": rows,
        "negative_deformed_pose_quantiles": {
            name: dict(zip(("median", "p95", "max"), np.quantile(values, [.5, .95, 1.]).tolist()))
            for name, values in negative.items()
        },
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_references()
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
