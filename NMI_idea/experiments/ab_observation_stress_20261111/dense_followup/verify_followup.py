"""Verify that coarse/dense development runs share worlds, CRN, and common poses."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(arm: str, cohort: str):
    fixture = json.loads((HERE / arm / "CAPABILITY.json").read_text())
    run = HERE / arm / ("dev_721111" if cohort == "dev" else "test_721129")
    model = np.load(run / "model_inputs.npz")
    audit = np.load(run / "renderer_audit.npz")
    return fixture, model, audit


def _max_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.dtype == bool or right.dtype == bool:
        return float(np.not_equal(left, right).max()) if left.size else 0.0
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    coarse, cmodel, caudit = _load("coarse", args.cohort)
    dense, dmodel, daudit = _load("dense", args.cohort)
    cposes = np.asarray(coarse["poses"], dtype=float)
    dposes = np.asarray(dense["poses"], dtype=float)
    dmap = {tuple(np.round(pose, 9)): index for index, pose in enumerate(dposes)}
    common = [(index, dmap[tuple(np.round(pose, 9))]) for index, pose in enumerate(cposes)]
    ci = np.asarray([row[0] for row in common], dtype=int)
    di = np.asarray([row[1] for row in common], dtype=int)

    checks: dict[str, object] = {
        "coarse_pose_count": len(cposes),
        "dense_pose_count": len(dposes),
        "common_pose_count": len(common),
        "world_points_array_equal": bool(np.array_equal(caudit["world_points"], daudit["world_points"])),
        "uniforms_array_equal": bool(np.array_equal(caudit["uniforms"], daudit["uniforms"])),
        "regime_ids_array_equal": bool(np.array_equal(caudit["regime_ids"], daudit["regime_ids"])),
        "common_model_likelihood_max_abs": _max_diff(cmodel["likelihood"][:, :, ci, :], dmodel["likelihood"][:, :, di, :]),
        "common_model_correct_joint_max_abs": _max_diff(cmodel["correct_joint"][:, :, ci, :], dmodel["correct_joint"][:, :, di, :]),
        "common_model_joint_max_abs": _max_diff(cmodel["joint"][:, :, ci, :, :], dmodel["joint"][:, :, di, :, :]),
        "common_motion_time_max_abs": _max_diff(cmodel["motion_time_s"][np.ix_(ci, ci)], dmodel["motion_time_s"][np.ix_(di, di)]),
        "common_free_cost_max_abs": _max_diff(cmodel["motion_and_observation_s"][np.ix_(ci, ci)], dmodel["motion_and_observation_s"][np.ix_(di, di)]),
        "common_uniforms_array_equal": bool(np.array_equal(caudit["uniforms"], daudit["uniforms"])),
    }
    for name in ("target_visible", "target_qualified", "target_center_m", "target_geometric_length_m", "target_out_of_plane_angle_rad"):
        checks[f"common_audit_{name}_max_abs"] = _max_diff(caudit[name][:, ci], daudit[name][:, di])

    capability_checks = []
    for ccontext, dcontext in zip(coarse["contexts"], dense["contexts"]):
        crows = np.asarray(ccontext["joint_time_s"], dtype=float)
        drows = np.asarray(dcontext["joint_time_s"], dtype=float)
        mask_c = np.asarray(ccontext["available"], dtype=bool)
        mask_d = np.asarray([dcontext["available"][index] for index in di], dtype=bool)
        capability_checks.append({
            "id": ccontext["id"],
            "common_available_mask_equal": bool(np.array_equal(mask_c, mask_d)),
            "common_joint_time_max_abs": _max_diff(crows, drows[np.ix_(di, di)]),
            "coarse_initial_q_equal_dense": ccontext["q_start"] == dcontext["q_start"],
            "fixed_rail_equal": ccontext["fixed_rail_m"] == dcontext["fixed_rail_m"],
        })
    checks["capability_contexts"] = capability_checks
    checks["all_numeric_common_zero"] = all(
        value == 0.0
        for key, value in checks.items()
        if key.endswith("max_abs")
    )
    checks["all_shared_inputs_equal"] = bool(
        checks["world_points_array_equal"]
        and checks["uniforms_array_equal"]
        and checks["regime_ids_array_equal"]
        and checks["all_numeric_common_zero"]
        and all(
            row["common_available_mask_equal"]
            and row["common_joint_time_max_abs"] == 0.0
            and row["coarse_initial_q_equal_dense"]
            and row["fixed_rail_equal"]
            for row in capability_checks
        )
    )
    output = HERE / ("CROSS_ARM_DEV_CHECK.json" if args.cohort == "dev" else "CROSS_ARM_TEST_CHECK.json")
    output.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    if not checks["all_shared_inputs_equal"]:
        raise SystemExit("common coarse/dense inputs differ")


if __name__ == "__main__":
    main()
