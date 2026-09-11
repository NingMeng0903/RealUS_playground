"""Five archived states, three allocation policies, four command-model budgets.

This is a static counterfactual command matrix, never a closed-loop replay.
Run with the repository's genesis Python; no robot/transport is constructed.
"""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rm75_control")]
BASE_SCRIPT = ROOT / "analysis_artifacts/review_669beac6/single_state_policy_check.py"
spec = importlib.util.spec_from_file_location("archived_five_state_reference", BASE_SCRIPT)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.port_constraint import PortEnergyConstraint
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry

ENERGY_HOLD_S = 0.050
BUDGETS_J = (None, 0.00001, 0.001, 0.2)


def main():
    output = {
        "schema": "archived_five_state_command_matrix_v1",
        "scope": "Static counterfactual QP commands at five archived active003 states; no motion, physical energy, acoustic benefit or closed-loop force claim.",
        "reconstruction": "Reuse frozen state IDs and reconstruction from review_669beac6/single_state_policy_check.py. Log timestamp replaces unlogged solve time; measured angle is zero; preserve recorded nominal, previous outer command and actual slew interval.",
        "repair_allowance": "A fresh solver is created for every policy/budget cell. Full initial episode allowance; this does not reconstruct episode history.",
        "energy_model": {
            "wrench": "W_model = -physical_wrench_candidate_tool, full six-dimensional already-compensated TCP/tool wrench",
            "hold_s": ENERGY_HOLD_S,
            "available_j": list(BUDGETS_J),
            "uncertainty_and_tracking_bounds": "zero; nominal algebra only, not conservative physical bounds",
            "damping": "none",
            "assurance": "command_model",
            "physical_w_checked": False,
            "limitation": "Budget snapshots only. No runtime ledger, final IK publication, dwell/stop accounting, or positive physical credit is simulated.",
        },
        "source_sha256": {},
        "input_sha256": {},
        "points": [],
    }
    for path in (Path(__file__), BASE_SCRIPT, ROOT / "peirastic/contact_qp/qp.py",
                 ROOT / "peirastic/contact_qp/repair_policy.py",
                 ROOT / "peirastic/contact_qp/repair_episode.py",
                 ROOT / "peirastic/contact_qp/port_constraint.py"):
        output["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, ids in reference.SELECTED.items():
        path = reference.BASE / name / "001/contact_qp.jsonl"
        raw = path.read_bytes()
        output["input_sha256"][str(path)] = hashlib.sha256(raw).hexdigest()
        records = [json.loads(line) for line in raw.splitlines() if line]
        header = next(row for row in records if row["event"] == "study_start")
        scan = [row for row in records if row["event"] == "control_sample" and row["h_ref_s"] > 0]
        study = header["config"]
        force = header["effective_configuration"]["force"]
        geometry_settings = dict(study["geometry"])
        geometry_settings.pop("face_normal_convention")
        geometry = ProbeGeometry(**geometry_settings)
        vmax = np.array(force["max_velocity"])
        vmax[2] = min(vmax[2], force["max_vz_tool_m_s"])
        vmax[4] = min(vmax[4], .28)
        amax = np.array(force["max_acceleration"])
        amax[4] = min(amax[4], 3.)
        v7 = QpConfig(**dict(study.get("qp") or {}), c_min=study["feature"]["c_min"],
                      quality_policy_version=study["feature"]["quality_policy_version"],
                      lateral_windows=study["feature"]["config"]["lateral_windows"],
                      max_velocity=vmax, max_acceleration=amax, angle_limit_rad=np.deg2rad(150.))
        policies = {
            "legacy_v7": v7,
            "v8r2_bounded_episode": replace(v7, allocation_policy="differential_repair_v8"),
            "v8r3_confidence_balance": replace(v7, allocation_policy="differential_repair_v8",
                differential_repair=replace(v7.differential_repair, revision="v8r3_confidence_balance")),
        }
        left, right = window_rows(geometry, v7.lateral_windows)[[0, 2]]
        for cid in ids:
            row = next(row for row in scan if row["control_id"] == cid)
            nominal = np.array(row["nominal_twist_tool"])
            path_twist = nominal.copy()
            path_twist[[2, 4]] = 0.
            observation = ContactObservation.from_dict(row["feature"])
            wrench_model = -np.array(row["physical_wrench_candidate_tool"])
            point = dict(scan=name, control_id=cid,
                elapsed_s=row["record_monotonic_s"]-scan[0]["record_monotonic_s"],
                force_n=row["control_wrench_tool"][2], quality=observation.quality,
                nominal_twist_tool=nominal, wrench_command_model_tool=wrench_model, variants={})
            for label, config in policies.items():
                point["variants"][label] = {}
                for budget in BUDGETS_J:
                    energy = None if budget is None else PortEnergyConstraint(
                        wrench_environment=wrench_model, available_j=budget, hold_s=ENERGY_HOLD_S,
                        assurance="command_model", bounds_version="archived_zero_uncertainty_diagnostic")
                    inp = QpInput(geometry, nominal, path_twist, row["control_wrench_tool"][2],
                        row["command_hold_model_s"], row["record_monotonic_s"],
                        observation=observation, previous_twist=np.array(row["previous_outer_command_tool"]),
                        acceleration_dt_s=row["command_slew_dt_s"], energy=energy,
                        repair_execution_enabled=True)
                    result = ContactQp(config).solve(inp, interval_diagnostics=True)
                    diagnostics = result.diagnostics
                    cell = dict(status=result.status.value, success=result.success,
                        reason=diagnostics.get("reason"), budget_j=budget,
                        repair_episode=diagnostics.get("repair_episode"),
                        force_gate=diagnostics.get("repair_force_gate"))
                    if result.success:
                        velocity = result.qp_twist
                        power = -float(wrench_model @ velocity)
                        cell.update(twist_tool=velocity, vn_mm_s=1000*velocity[2],
                            omega_deg_s=np.rad2deg(velocity[4]),
                            added_vn_mm_s=1000*(velocity[2]-nominal[2]),
                            added_omega_deg_s=np.rad2deg(velocity[4]-nominal[4]),
                            alpha=result.alpha, visual_slack_mm_s=1000*result.slack,
                            left_local_mm_s=1000*float(left@velocity), right_local_mm_s=1000*float(right@velocity),
                            differential_request_mm_s=1000*diagnostics.get("differential_request_m_s", 0.),
                            differential_shortfall_mm_s=1000*diagnostics.get("differential_shortfall_m_s", 0.),
                            model_output_power_w=power, model_output_work_50ms_j=power*ENERGY_HOLD_S,
                            command_energy_margin_w=None if energy is None else energy.margin_power_w(velocity),
                            active_hard_rows=diagnostics.get("active_hard_rows"))
                        if energy is not None:
                            assert energy.admissible(velocity, tolerance_w=1e-8), (name, cid, label, budget)
                        if label == "legacy_v7" and budget is None:
                            error = float(np.max(abs(velocity-np.array(row["candidate_twist_tool"]))))
                            assert error < 1e-7, (name, cid, error)
                            point["v7_saved_max_abs_error"] = error
                    point["variants"][label]["off" if budget is None else f"{budget:.8g}J"] = cell
            output["points"].append(point)
    output["source_sha256_end"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                    for name in output["source_sha256"]}
    output["source_unchanged_during_run"] = output["source_sha256"] == output["source_sha256_end"]
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(reference.serial(output), indent=2, allow_nan=False)+"\n")
    print(destination)
    for point in output["points"]:
        print(point["scan"], point["control_id"], "F", round(point["force_n"], 3))
        for policy, variants in point["variants"].items():
            print(policy, {name: (round(cell["omega_deg_s"], 5), round(cell["alpha"], 5),
                round(cell["model_output_power_w"], 6)) if cell["success"] else cell["status"]
                for name, cell in variants.items()})


if __name__ == "__main__":
    main()
