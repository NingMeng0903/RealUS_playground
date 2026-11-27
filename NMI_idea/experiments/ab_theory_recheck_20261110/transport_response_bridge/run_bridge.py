#!/usr/bin/env python3
"""Bounded atlas-geometry to frozen synthetic-response bridge.

This file intentionally writes only inside transport_response_bridge/.  It
reuses the audited harmonic RAW_RESULTS and the already-frozen checkpoints;
there is no refitting and no hidden target input to the predictors.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / "harmonic_mechanism"
RAW_HARMONIC = AUDIT / "RAW_RESULTS.npz"
ASSET = AUDIT.parent.parent / "ab_validation_20261110/repair_20261110/harmonic/shaft_roi_v2"
LEARNING = ROOT.parent / "response_learning_geometry/geometry_addendum"
VENDOR = ROOT.parent.parent / "ab_validation_20261110/theory/vendor"
sys.path.insert(0, "/media/camp/EXT_DRIVE/Among_US/src")
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(LEARNING))

from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import (  # noqa: E402
    load_leg_volume_atlas,
)
import torch  # noqa: E402
import run_corrected as corrected  # noqa: E402


SIDES = ("right", "left")
POSES = tuple(range(0, 91, 15))
CELL_COUNT = 256
CANDIDATES = 32
SEED = 720998
TARGET = np.asarray([0.4, -0.3], dtype=np.float32)
THRESHOLD = 0.2
UNIT_MM = 10.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mats(vertices: np.ndarray) -> np.ndarray:
    return np.moveaxis(vertices[:, 1:] - vertices[:, :1], 1, 2)


def cell_geometry(side: str, harmonic: dict[str, np.ndarray]):
    atlas = load_leg_volume_atlas(ASSET / f"atlas_{side}.npz")
    ids = harmonic[f"{side}_cell_ids"]
    bary = harmonic[f"{side}_barycentric"]
    cells = atlas.harmonic_tets[ids]
    v0 = atlas.harmonic_vertices[cells].astype(float)
    X0 = mats(v0)
    labels = np.stack((atlas.harmonic_theta, atlas.harmonic_h, atlas.harmonic_d), -1)[cells].astype(float)
    theta = labels[:, :, 0]
    ss = np.sum(bary * np.sin(theta), axis=1)
    cc = np.sum(bary * np.cos(theta), axis=1)
    theta_b = (
        cc[:, None] * (np.sin(theta[:, 1:]) - np.sin(theta[:, :1]))
        - ss[:, None] * (np.cos(theta[:, 1:]) - np.cos(theta[:, :1]))
    ) / (ss * ss + cc * cc)[:, None]
    Kb = np.stack(
        (theta_b, labels[:, 1:, 1] - labels[:, :1, 1], labels[:, 1:, 2] - labels[:, :1, 2]),
        axis=1,
    )
    J0 = X0 @ np.linalg.inv(Kb)
    # h=1 is the tangent coordinate; d=2 is the outward-depth covector.
    t0 = unit(J0[:, :, 1])
    n0 = -unit(np.linalg.inv(J0)[:, 2, :])
    return atlas, ids, bary, cells, X0, J0, t0, n0


def frame_from_transport(F: np.ndarray, t0: np.ndarray, n0: np.ndarray):
    t = unit(np.einsum("nij,nj->ni", F, t0))
    n = unit(np.linalg.solve(np.swapaxes(F, 1, 2), n0[..., None])[..., 0])
    Rx = t
    Rz = -n
    Ry = unit(np.cross(Rz, Rx))
    R = np.stack((Rx, Ry, Rz), axis=2)
    return t, n, R


def make_actions():
    rng = np.random.default_rng(SEED)
    s = rng.uniform(-1.0, 1.0, size=(CELL_COUNT, 4)).astype(np.float32)
    direction = unit(rng.normal(size=(CELL_COUNT, CANDIDATES, 3)).astype(np.float32))
    magnitude = rng.uniform(0.15, 0.8, size=(CELL_COUNT, CANDIDATES, 1)).astype(np.float32)
    return s, (direction * magnitude).astype(np.float32)


def true_response(s, u_local, R, n, t):
    R_T = np.swapaxes(R, 1, 2)
    n_local = np.einsum("nij,nj->ni", R_T, n)
    t_local = np.einsum("nij,nj->ni", R_T, t)
    q_local = corrected.base.plant_residual_local(
        s[:, None, :], u_local, n_local[:, None, :], t_local[:, None, :]
    )
    u_world = np.einsum("nij,nkj->nki", R, u_local)
    q_world = np.einsum("nij,nkj->nki", R, q_local)
    return u_world, q_world, u_local + q_local


def predict_model(model, s, u_world, n, t, R):
    # Frozen checkpoints were trained in float32.  Geometry is intentionally
    # accumulated in float64 for the contract checks, then cast only at this
    # model boundary so the scorer and formulas retain their native precision.
    data = {
        "s": np.asarray(s, dtype=np.float32),
        "u_world": np.asarray(u_world, dtype=np.float32),
        "n_world": np.asarray(n, dtype=np.float32),
        "t_world": np.asarray(t, dtype=np.float32),
        "R": np.asarray(R, dtype=np.float32),
    }
    return corrected.predict(model, data).astype(np.float32)


def candidate_score(pred_q, u_world, q_true, u_local, R):
    pred_local = np.einsum("nij,nkj->nki", np.swapaxes(R, 1, 2), u_world + pred_q)
    true_local = u_local + np.einsum("nij,nkj->nki", np.swapaxes(R, 1, 2), q_true)
    true_cost = np.linalg.norm(true_local[..., [0, 2]] - TARGET, axis=-1)
    pred_cost = np.linalg.norm(pred_local[..., [0, 2]] - TARGET, axis=-1)
    chosen = np.argmin(pred_cost, axis=1)
    rows = np.arange(chosen.shape[0])
    actual = true_cost[rows, chosen]
    oracle = true_cost.min(axis=1)
    return {
        "response_rmse_mm": float(np.sqrt(np.mean((pred_q - q_true) ** 2)) * UNIT_MM),
        "mean_actual_error_mm": float(actual.mean() * UNIT_MM),
        "p95_actual_error_mm": float(np.quantile(actual, 0.95) * UNIT_MM),
        "success_rate_2mm": float(np.mean(actual <= THRESHOLD)),
        "mean_oracle_error_mm": float(oracle.mean() * UNIT_MM),
        "oracle_success_rate_2mm": float(np.mean(oracle <= THRESHOLD)),
        "mean_regret_mm": float(np.mean(actual - oracle) * UNIT_MM),
        "p95_regret_mm": float(np.quantile(actual - oracle, 0.95) * UNIT_MM),
    }


def load_checkpoint(class_name: str, path: Path):
    cls = getattr(corrected, class_name)
    model = cls().to(corrected.DEVICE).float()
    ckpt = torch.load(path, map_location=corrected.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run():
    harmonic = dict(np.load(RAW_HARMONIC))
    s, u_local = make_actions()
    arrays: dict[str, np.ndarray] = {"cell_context_s": s, "cell_actions_u_local": u_local}
    hashes: dict[str, str] = {str(RAW_HARMONIC): sha(RAW_HARMONIC), str(AUDIT / "run_transport.py"): sha(AUDIT / "run_transport.py")}
    models = {
        "e3nn_correct": load_checkpoint(
            "E3NNTypedBasis", LEARNING / "weights/e3nn_typed_basis_positive_n2048_seed101.pt"
        ),
        "e3nn_frozen_world": load_checkpoint(
            "E3NNTypedBasis", LEARNING / "weights/e3nn_typed_basis_positive_n2048_seed101.pt"
        ),
        "e3nn_wrong_F_normal": load_checkpoint(
            "E3NNTypedBasis", LEARNING / "weights/e3nn_typed_basis_positive_n2048_seed101.pt"
        ),
        "local_dyadic_correct": load_checkpoint(
            "LocalDyadicMLP", LEARNING / "weights/probe_local_dyadic_mlp_positive_n2048_seed101.pt"
        ),
    }
    for p in (
        LEARNING / "weights/e3nn_typed_basis_positive_n2048_seed101.pt",
        LEARNING / "weights/probe_local_dyadic_mlp_positive_n2048_seed101.pt",
    ):
        hashes[str(p)] = sha(p)

    results = {"status": "complete", "protocol": {"seed": SEED, "unit_mm": UNIT_MM, "target_normalized": TARGET.tolist(), "threshold_normalized": THRESHOLD, "cells_per_side": CELL_COUNT, "candidates": CANDIDATES, "poses": list(POSES), "device": str(corrected.DEVICE)}, "sides": {}, "metrics": []}
    for side in SIDES:
        atlas, ids, bary, cells, X0, J0, t0, n0 = cell_geometry(side, harmonic)
        hashes[str(ASSET / f"atlas_{side}.npz")] = sha(ASSET / f"atlas_{side}.npz")
        arrays[f"{side}_J0"] = J0
        arrays[f"{side}_t0"] = t0
        arrays[f"{side}_n0"] = n0
        side_rows = []
        dots0 = np.abs(np.sum(t0 * n0, axis=1))
        for degree in POSES:
            posefile = ASSET / f"{side}_pose_{degree:02d}.npz"
            hashes[str(posefile)] = sha(posefile)
            F = harmonic[f"{side}_{degree}_deformation"].astype(np.float64)
            t, n, R = frame_from_transport(F, t0, n0)
            u_world, q_true, total_local = true_response(s, u_local, R, n, t)
            wrong_n = unit(np.einsum("nij,nj->ni", F, n0))
            arrays[f"{side}_{degree}_F"] = F
            arrays[f"{side}_{degree}_t"] = t
            arrays[f"{side}_{degree}_n"] = n
            arrays[f"{side}_{degree}_R"] = R
            arrays[f"{side}_{degree}_q_true_world"] = q_true
            arrays[f"{side}_{degree}_wrong_n"] = wrong_n

            # The label/scorer truth is evaluated with current geometry only;
            # the models see public s, action, and whichever geometry variant
            # the row names below declare.
            inputs = {
                "e3nn_correct": (n, t),
                "e3nn_frozen_world": (np.repeat(n0[:, None, :], 1, axis=1)[:, 0], t0),
                "e3nn_wrong_F_normal": (wrong_n, t),
                "local_dyadic_correct": (n, t),
            }
            row = {"side": side, "pose_deg": degree, "canonical_tn_abs_cosine_max": float(dots0.max())}
            for name, model in models.items():
                nm, tm = inputs[name]
                pred = predict_model(model, s, u_world, nm, tm, R)
                row[name] = candidate_score(pred, u_world, q_true, u_local, R)
                arrays[f"{side}_{degree}_{name}_pred_q_world"] = pred
            # Independent ordinary tet/barycentric correspondence check.
            data = np.load(posefile)
            vp = data["vertices"].astype(float)[cells]
            Xp = mats(vp)
            p_bary = np.einsum("ni,nij->nj", bary, vp)
            p_edges = vp[:, 0] + np.einsum("nij,nj->ni", Xp, bary[:, 1:])
            row["ordinary_tet_barycentric_max_error_m"] = float(np.max(np.linalg.norm(p_bary - p_edges, axis=1)))
            row["frame_max_orth_error"] = float(np.max(np.linalg.norm(np.swapaxes(R, 1, 2) @ R - np.eye(3), axis=(1, 2))))
            row["frame_min_det"] = float(np.min(np.linalg.det(R)))
            row["transport_tn_abs_cosine_max"] = float(np.max(np.abs(np.sum(t * n, axis=1))))
            results["metrics"].append(row)
            side_rows.append(row)
            print(side, degree, {k: round(v["response_rmse_mm"], 4) for k, v in row.items() if isinstance(v, dict)}, flush=True)
        results["sides"][side] = {"cell_count": int(len(ids)), "canonical_tn_abs_cosine_max": float(dots0.max()), "rows": side_rows}

    np.savez_compressed(ROOT / "RAW_RESULTS.npz", **arrays)
    results["source_sha256"] = sha(ROOT / "run_bridge.py")
    results["protocol_sha256"] = sha(ROOT / "PROTOCOL.txt")
    results["input_hashes"] = hashes
    (ROOT / "RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(results)
    write_manifest(results)
    return results


def make_report(result):
    lines = [
        "Transport-response bridge",
        "=========================",
        "Synthetic compliance bridge over the audited atlas/tet geometry; not real ultrasound and not an independent-patient test.",
        "",
        "mode | pose | response RMSE mm | chosen error mm | success <=2mm | regret mm | oracle success",
    ]
    for r in result["metrics"]:
        for name in ("e3nn_correct", "e3nn_frozen_world", "e3nn_wrong_F_normal", "local_dyadic_correct"):
            m = r[name]
            lines.append(f"{name} | {r['side']} {r['pose_deg']} | {m['response_rmse_mm']:.6g} | {m['mean_actual_error_mm']:.6g} | {m['success_rate_2mm']:.6g} | {m['mean_regret_mm']:.6g} | {m['oracle_success_rate_2mm']:.6g}")
    lines.extend(["", "Geometry checks:"])
    for r in result["metrics"]:
        lines.append(f"{r['side']} {r['pose_deg']}: frame orth max={r['frame_max_orth_error']:.3g}, det min={r['frame_min_det']:.9g}, t-n cosine max={r['transport_tn_abs_cosine_max']:.3g}, ordinary tet error m={r['ordinary_tet_barycentric_max_error_m']:.3g}")
    lines.extend(["", "All predictions use public context/action/body geometry and frozen checkpoints. Scorer truth uses the current transported frame; no held-out truth is an input."])
    (ROOT / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(result):
    local = ["PROTOCOL.txt", "run_bridge.py", "test_bridge.py", "RAW_RESULTS.npz", "RESULTS.json", "REPORT.txt"]
    manifest = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(corrected.DEVICE),
        "files_sha256": {name: sha(ROOT / name) for name in local},
        "external_input_sha256": result["input_hashes"],
        "note": "The external input hashes include both frozen geometry assets and frozen model checkpoints; no checkpoint is copied or retrained here.",
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
