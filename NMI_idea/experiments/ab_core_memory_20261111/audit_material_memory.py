"""Check the persistent response query at actual saved material correspondences."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "ab_theory_recheck_20261110/transport_response_bridge"
sys.path.insert(0, str(BRIDGE))
from run_bridge import ASSET, POSES, RAW_HARMONIC, cell_geometry, frame_from_transport, mats


NOISE = .035
PRIOR_STD = np.array([.065, .23])
PRIOR_PRECISION = np.diag(PRIOR_STD ** -2)
PROBE_YAW = np.array([-.025, .025])
TERMINAL_YAW = np.array([-.10, .10])


def response_design(E, world_rotation):
    """Signed in-plane tilt from orthonormal material and probe axes."""
    direction = world_rotation[..., :, 0]
    tangent = np.einsum("...i,...i->...", E[..., :, 0], direction)
    lateral = np.einsum("...i,...i->...", E[..., :, 1], direction)
    yaw = np.arctan2(lateral, tangent)
    return np.stack((np.ones_like(yaw), np.sin(yaw) / np.sin(.10)), axis=-1)


def coefficients(records):
    precision = PRIOR_PRECISION.copy()
    info = np.zeros(2)
    if records:
        X = np.array([r["design"] for r in records])
        y = np.array([r["residual"] for r in records])
        precision += X.T @ X / NOISE**2
        info += X.T @ y / NOISE**2
    covariance = np.linalg.inv(precision)
    return np.linalg.solve(precision, info), covariance


def prepare(rng):
    frames, addresses = [], []
    inputs = [RAW_HARMONIC, BRIDGE / "run_bridge.py", Path(__file__)]
    with np.load(RAW_HARMONIC) as source:
        harmonic = {key: source[key] for key in source.files}
    for side in ("left", "right"):
        _, ids, _, cells, X0, _, t0, n0 = cell_geometry(side, harmonic)
        selected = rng.choice(len(ids), 32, replace=False)
        addresses.extend((side, int(i)) for i in ids[selected])
        pose_frames = []
        inputs.append(ASSET / f"atlas_{side}.npz")
        for knee in POSES:
            path = ASSET / f"{side}_pose_{knee:02d}.npz"
            inputs.append(path)
            with np.load(path) as posed:
                vp = posed["vertices"][cells[selected]].astype(float)
            F = mats(vp) @ np.linalg.inv(X0[selected])
            _, _, E = frame_from_transport(F, t0[selected], n0[selected])
            pose_frames.append(E)
        frames.append(np.array(pose_frames))
    return addresses, np.concatenate(frames, axis=1), inputs


def run():
    rng = np.random.default_rng(721220)
    addresses, frames, inputs = prepare(rng)
    n = len(addresses)
    probe_R = Rotation.from_euler("z", PROBE_YAW).as_matrix()
    end_R = Rotation.from_euler("z", TERMINAL_YAW).as_matrix()
    probes_world = frames[0, :, None] @ probe_R
    probe_design = response_design(frames[0, :, None], probes_world)
    latent = rng.normal(size=(n, 2)) * PRIOR_STD
    observed = np.einsum("nab,nb->na", probe_design, latent) + rng.normal(0, NOISE, (n, 2))
    records = [
        {"event_id": (address, i), "address": address, "design": probe_design[k, i], "residual": observed[k, i]}
        for k, address in enumerate(addresses) for i in range(len(PROBE_YAW))
    ]
    precision = {address: PRIOR_PRECISION.copy() for address in addresses}
    info = {address: np.zeros(2) for address in addresses}
    seen, all_by_address = set(), defaultdict(list)
    # Re-present the whole set once, as a replay would. An event contributes
    # at most once to the persistent snapshot.
    for record in records + records:
        key, address = record["event_id"], record["address"]
        if key in seen:
            continue
        seen.add(key)
        row = record["design"]
        precision[address] += np.outer(row, row) / NOISE**2
        info[address] += row * record["residual"] / NOISE**2
        all_by_address[address].append(record)
    posterior = np.stack([np.linalg.solve(precision[a], info[a]) for a in addresses])
    replay = [coefficients(all_by_address[a]) for a in addresses]
    window_size = 8
    window = np.stack([
        coefficients([r for r in records[-window_size:] if r["address"] == a])[0]
        for a in addresses
    ])
    replay_error = float(np.max(np.abs(posterior - np.array([m for m, _ in replay]))))
    covariance_error = float(np.max(np.abs(np.array([np.linalg.inv(precision[a]) for a in addresses]) -
                                            np.array([C for _, C in replay]))))
    result = {"seed": 721220, "independent_synthetic_coefficients": n, "saved_poses_per_side": len(POSES),
              "accepted_unique_events": len(seen), "window_frames": window_size,
              "addresses_recoverable_from_recent_window": len({r["address"] for r in records[-window_size:]}),
              "memory_vs_full_replay_max_coefficient_difference": replay_error,
              "memory_vs_full_replay_max_covariance_difference": covariance_error}
    if max(replay_error, covariance_error) > 1e-12:
        raise ValueError("persistent and full replay estimates differ")

    by_pose = []
    for knee, E in zip(POSES, frames):
        end_world = E[:, None] @ end_R
        X = response_design(E[:, None], end_world)
        true_value = np.einsum("nab,nb->na", X, latent)
        current = np.einsum("nab,nb->na", X, posterior)
        only_window = np.einsum("nab,nb->na", X, window)
        Q = Rotation.random(n, random_state=rng).as_matrix()[:, None]
        transported = response_design(Q @ E[:, None], Q @ end_world)
        frozen = response_design(frames[0, :, None], end_world)
        row = {
            "knee_deg": knee,
            "common_rotation_design_max": float(np.max(np.abs(transported - X))),
            "old_world_frame_design_rmse": float(np.sqrt(np.mean((frozen[..., 1] - X[..., 1]) ** 2))),
            "residual_rmse_with_memory": float(np.sqrt(np.mean((current - true_value) ** 2))),
            "residual_rmse_with_window": float(np.sqrt(np.mean((only_window - true_value) ** 2))),
            "best_sign_correct_with_memory": int((current.argmax(axis=1) == true_value.argmax(axis=1)).sum()),
            "best_sign_correct_with_window": int((only_window.argmax(axis=1) == true_value.argmax(axis=1)).sum()),
        }
        by_pose.append(row)
    if max(row["common_rotation_design_max"] for row in by_pose) > 1e-12:
        raise ValueError("common rotation changed the material design")
    # Independent acoustic reset at the same address. Knowing it happened
    # leaves the same atlas coordinates but restores the current prior.
    new_coefficients = rng.normal(size=(n, 2)) * PRIOR_STD
    X = response_design(frames[-1, :, None], frames[-1, :, None] @ end_R)
    new_value = np.einsum("nab,nb->na", X, new_coefficients)
    stale = np.einsum("nab,nb->na", X, posterior)
    result["new_epoch"] = {
        "reset_prior_prediction_rmse": float(np.sqrt(np.mean(new_value**2))),
        "stale_prediction_rmse": float(np.sqrt(np.mean((new_value - stale)**2))),
    }
    result["scope"] = (
        "Real stored material frames and typed action features. Residual coefficients and measurements "
        "are synthetic and assumed stable within the first epoch. No real contact, acoustic transport, "
        "learning transfer, or acquisition benefit is established by this interface test."
    )
    result["by_pose"] = by_pose
    result["input_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as file:
        json.dump(run(), file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
