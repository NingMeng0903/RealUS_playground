"""Execute frozen dIRD candidate paths in an offline measured-state joint plant."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .dird_probe import DirdProbe
from .joint_execution import JointServoBackend, JointServoConfig


DEFAULT_CHECKPOINT = Path("/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/data/checkpoints/rm4d_signed_current_robot_20261108/selected.pt")


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_trajectory(path: Path, trajectory: list[dict]):
    keys = ("q_cmd", "q_meas", "qdot", "T_tcp_base", "T_tcp_world", "T_image_world",
            "sim_time_s", "wall_elapsed_s", "self_collision_distance_m", "dense_step_clearance_m",
            "dense_step_collision_samples", "waypoint_index")
    shapes = {"q_cmd": (8,), "q_meas": (8,), "qdot": (8,),
              "T_tcp_base": (4, 4), "T_tcp_world": (4, 4), "T_image_world": (4, 4)}
    arrays = {key: np.asarray([r[key] for r in trajectory]).reshape((len(trajectory),) + shapes.get(key, ()))
              for key in keys}
    arrays["phase"] = np.asarray([r["phase"] for r in trajectory], dtype="U16")
    np.savez_compressed(path, **arrays)


def run(output: Path, audit_path: Path, *, backend="genesis_pd", entries=2):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    audit = json.loads(audit_path.read_text())
    probe = DirdProbe(checkpoint=DEFAULT_CHECKPOINT, allow_stale_audit=False)
    assert probe.metadata()["robot_contract_matches"]
    assert audit["checkpoint_metadata"]["checkpoint_sha256"] == probe.metadata()["checkpoint_sha256"]
    config = JointServoConfig(backend=backend)
    plant = JointServoBackend(config, base_world=np.asarray(audit["base_world"]),
                              T_image_tcp=np.asarray(audit["T_image_tcp"]))
    q_seed = np.asarray(audit["q_seed"])
    rng = np.random.default_rng(20261108)
    entry_states = [q_seed]
    for i in range(1, entries):
        # Public independent entry perturbation; chosen once, shared over cases.
        offset = np.r_[rng.uniform(-.025, .025), rng.uniform(-.025, .025, 7)]
        q = q_seed + offset
        if plant.clearance(q) < config.collision_margin_m:
            raise RuntimeError("predeclared random entry violates self-collision margin")
        entry_states.append(q)
    protocol = {"schema": "offline_current_rm75_joint_execution_v1", "source_audit": str(audit_path.resolve()),
                "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "strict_checkpoint": probe.metadata(), "plant": plant.manifest(),
                "entry_states": [x.tolist() for x in entry_states], "entry_seed": 20261108,
                "trial_rule": "execute all audited candidates, including preflight or IK failures; do not select only successful paths",
                "physical_execution": False, "ready_for_hardware": False,
                "limitations": ["virtual placement, not measured registration", "illustrative image-to-TCP transform",
                                "no body/bed contact or environment collision", "no B-mode/image acceptance",
                                "Genesis collision response disabled; current configured self-collision checked and stopped independently"]}
    _write(output / "frozen_protocol.json", protocol)
    rows = []
    for case in audit["cases"]:
        for rank, candidate in enumerate(case["audit"]["mode_audits"]):
            lifted = candidate["audit"]
            for entry_id, entry in enumerate(entry_states):
                trial = {"case_id": case["case_id"], "knee_deg": case["knee_deg"],
                         "shape_controls": [case["width_scale"], case["length_scale"]],
                         "reference_mode": case["reference_mode"], "candidate_rank": rank + 1,
                         "rail_m": candidate["rail_m"][0], "entry_id": entry_id,
                         "dird_min_score": candidate["dird_min_score"],
                         "nominal_scan_retiming_s": lifted.get("nominal_duration_s")}
                if not lifted.get("kinematic_audit_passed", False):
                    trial.update(success=False, status="candidate_kinematic_rejection", duration_s=0.,
                                 failure=lifted.get("failure"), recorded_ticks=0)
                else:
                    result = plant.execute_path(np.asarray(lifted["q_ref"]), entry)
                    trajectory = result.pop("trajectory")
                    stem = f"trial_{len(rows):03d}"
                    save_trajectory(output / (stem + ".npz"), trajectory)
                    trial.update(result, trajectory_file=stem + ".npz")
                    _write(output / (stem + ".json"), trial)
                rows.append(trial)
                print(json.dumps({k: trial.get(k) for k in ("case_id", "candidate_rank", "entry_id", "status", "duration_s")}), flush=True)
    summary = {"schema": protocol["schema"], "backend": backend, "physical_execution": False,
               "trials": rows, "n_trials": len(rows), "n_success": sum(r["success"] for r in rows),
               "n_entry_collision_rejection": sum(r["status"] == "entry_collision_rejection" for r in rows),
               "n_candidate_kinematic_rejection": sum(r["status"] == "candidate_kinematic_rejection" for r in rows),
               "n_dynamic_stops": sum("stop" in r["status"] or "timeout" in r["status"] for r in rows),
               "wall_seconds": time.perf_counter() - began,
               "case_with_success": sorted({r["case_id"] for r in rows if r["success"]}),
               "duration_kind": "actual simulated execution clock incl. entry transition and settling",
               "hardware_or_contact_validated": False}
    _write(output / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path, default=Path("outputs/dird_current_robot_20261108/smplx_mode_audit.json"))
    p.add_argument("--backend", choices=("genesis_pd", "pinocchio_lag"), default="genesis_pd")
    p.add_argument("--entries", type=int, default=2)
    args = p.parse_args()
    report = run(args.output, args.audit, backend=args.backend, entries=args.entries)
    print(json.dumps({k: report[k] for k in ("backend", "n_trials", "n_success", "n_entry_collision_rejection", "n_dynamic_stops", "wall_seconds")}))


if __name__ == "__main__":
    main()
