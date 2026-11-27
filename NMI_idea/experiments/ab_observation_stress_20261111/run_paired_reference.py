"""Paired ProDMP versus audited spline replay on the same native inputs.

This is a diagnostic runner only.  It uses a fresh independent replay for each
short segment, the same q seed, public anchor, native binary, profile, and QP
configuration.  It never changes production files or controller parameters.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_validation_20261110/repair_20261110/qp_cross_track_fix")
SELECTED = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_module_ladder_20261110/planning/native_regression_20261110/public_prior_layout/SELECTED.json")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from motion import ProDMPReference, duration_matrix  # noqa: E402
from run_native_motion import ProDMPReplayBridge  # noqa: E402
from scan_replay import ScanReplayBridge  # noqa: E402
from rm75_control.control.joint_admittance_8dof.scan_reference import WorldPoseSplineReference  # noqa: E402


GOALS = np.asarray([
    [0.004, 0.000, 0.000],
    [0.000, 0.004, 0.080],
    [-0.004, 0.003, -0.080],
    [0.006, -0.004, 0.120],
], dtype=float)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _tcp_pair(anchor: np.ndarray, goal: np.ndarray) -> np.ndarray:
    target = anchor.copy()
    target[:3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([0., 0., goal[2]]).as_matrix()
    target[:3, 3] = anchor[:3, 3] + anchor[:3, :3] @ np.r_[goal[:2], 0.]
    return np.stack([anchor, target])


def _trace_metrics(trace, reference, kin, result):
    if not trace:
        return {"sample_count": 0, "reference_pose_max_abs_m_or_rad": None}
    t = np.asarray([row["t_s"] for row in trace], dtype=float)
    desired = np.asarray([row["pose_desired"] for row in trace], dtype=float)
    actual_qnext = np.asarray([row["pose_actual"] for row in trace], dtype=float)
    measured_qbefore = np.asarray([kin.fk_placement(np.asarray(row["q_meas_before"], dtype=float)).homogeneous for row in trace])
    refs = [reference.sample(float(x)) for x in t]
    expected = np.asarray([reference_to_matrix(ref.pose_d, "xyz") for ref in refs])
    ff = np.asarray([ref.vel_ff for ref in refs])
    requested = np.asarray([row["twist_requested"] for row in trace])
    # The replay trace exposes the controller's requested/feasible task twist,
    # while `ff` below is the external reference's actual base/world FF.
    feedback = requested - ff
    same_time_pos = np.linalg.norm(measured_qbefore[:, :3, 3] - expected[:, :3, 3], axis=1)
    same_time_rot = Rotation.from_matrix(np.transpose(expected[:, :3, :3], (0, 2, 1)) @ measured_qbefore[:, :3, :3]).magnitude()
    qdot = np.asarray([row["qdot"] for row in trace])
    box_lo = np.asarray([row["box_lo"] for row in trace])
    box_hi = np.asarray([row["box_hi"] for row in trace])
    v_recv = np.asarray([row["v_cmd_received"] for row in trace])
    v_feas = np.asarray([row["v_cmd_feasible"] for row in trace])
    return {
        "sample_count": int(len(trace)),
        "reference_duration_s": float(reference.duration_s()),
        "trace_last_t_s": float(t[-1]),
        "trace_dt_s_median": float(np.median(np.diff(t))) if len(t) > 1 else None,
        "reference_pose_max_abs_m_or_rad": float(np.max(np.abs(desired - expected))),
        "reference_ff_max_abs_m_s_or_rad_s": float(np.max(np.abs(ff))),
        "ff_frame": "world/base geometric twist from reference; controller receives it through rail_base Cartesian track",
        "requested_minus_reference_ff_norm_p95": float(np.percentile(np.linalg.norm(feedback, axis=1), 95)),
        "requested_minus_reference_ff_norm_max": float(np.max(np.linalg.norm(feedback, axis=1))),
        "same_time_q_meas_before_position_p95_m": float(np.percentile(same_time_pos, 95)),
        "same_time_q_meas_before_position_max_m": float(np.max(same_time_pos)),
        "same_time_q_meas_before_rotation_p95_rad": float(np.percentile(same_time_rot, 95)),
        "same_time_q_meas_before_rotation_max_rad": float(np.max(same_time_rot)),
        "legacy_q_send_position_p95_m": float(result.get("position_error_p95_m", np.nan)),
        "legacy_q_send_position_max_m": float(result.get("position_error_max_m", np.nan)),
        "legacy_q_send_rotation_p95_rad": float(result.get("rotation_error_p95_rad", np.nan)),
        "qdot_box_margin_min": float(np.min(np.minimum(qdot - box_lo, box_hi - qdot))),
        "velocity_margin_min": float(result.get("velocity_margin_min", np.nan)),
        "acceleration_margin_min": float(result.get("acceleration_margin_min", np.nan)),
        "jerk_margin_min": float(result.get("jerk_margin_min", np.nan)),
        "task_paused_count": int(result.get("task_paused_count", 0)),
        "joint_limited_count": int(sum(bool(row["joint_limited"]) for row in trace)),
        "rail_limited_count": int(sum(bool(row["rail_limited"]) for row in trace)),
        "physical_saturated_count": int(sum(bool(row["physical_saturated"]) for row in trace)),
        "qp_hard_residual_max": float(np.max([row["qp_hard_residual_max"] for row in trace])),
        "v_cmd_projection_norm_p95": float(np.percentile(np.linalg.norm(v_recv - v_feas, axis=1), 95)),
        "v_cmd_projection_norm_max": float(np.max(np.linalg.norm(v_recv - v_feas, axis=1))),
    }


def reference_to_matrix(pose, order):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(order, np.asarray(pose)[3:]).as_matrix()
    T[:3, 3] = np.asarray(pose)[:3]
    return T


def _run_one(bridge, tcp, q_seed, reference, output_path, provenance, settle_s, kin):
    result = bridge.query_pose_path(tcp, q_seed, speed_m_s=.02,
                                    settle_s=settle_s, provenance=provenance,
                                    return_trace=True, return_state=False)
    trace = result.pop("trace", [])
    if trace:
        np.savez_compressed(output_path, **{k: np.asarray([r[k] for r in trace]) for k in trace[0]})
    return result, _trace_metrics(trace, reference, kin, result)


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "motion_validation" / "paired_spline_vs_prodmp")
    parser.add_argument("--settle-s", type=float, default=.1)
    args = parser.parse_args()
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    public = json.loads(SELECTED.read_text())
    q_seed = np.asarray(public["q_seed"], dtype=float)
    anchor = np.asarray(public["T_world_body"], dtype=float)
    # A kinematics instance is used only for same-time q_meas_before FK in the
    # report.  It does not create a controller session or native SHM channel.
    probe = ScanReplayBridge()
    kin, inner, cfg = probe.make_controller()
    inner.close()
    spline_bridge = ScanReplayBridge()
    prodmp_bridge = ProDMPReplayBridge()
    records = {
        "schema": "ab_observation_stress_paired_reference_v1",
        "scope": "same q/target/profile native replay; no hardware; no QP/controller edits",
        "profile": str(ROOT / "scan_tracking_profile.yaml"),
        "q_seed": q_seed,
        "T_world_body": anchor,
        "settle_s": float(args.settle_s),
        "segments": [],
    }
    try:
        for index, goal in enumerate(GOALS):
            tcp = _tcp_pair(anchor, goal)
            local_pair = np.vstack([np.zeros(3), goal])
            prodmp = ProDMPReference(np.zeros(3), goal, anchor,
                                     duration_s=float(duration_matrix(local_pair, "prodmp")[0, 1]),
                                     euler_order=cfg.euler_order)
            spline = WorldPoseSplineReference(tcp[:, :3, 3],
                rpy=Rotation.from_matrix(tcp[:, :3, :3]).as_euler(cfg.euler_order),
                speed_m_s=.02, ramp_s=.8, euler_order=cfg.euler_order)
            segment = {"index": index, "goal_local": goal,
                       "prodmp_duration_s": prodmp.duration_s(),
                       "spline_duration_s": spline.duration_s(), "methods": {}}
            for name, bridge, ref in (("prodmp", prodmp_bridge, prodmp), ("spline", spline_bridge, spline)):
                try:
                    result, metrics = _run_one(bridge, tcp, q_seed, ref,
                        outdir / f"{name}_{index}_trace.npz",
                        f"paired {name} short local segment {index}", args.settle_s, kin)
                    segment["methods"][name] = {"status": "completed", "result": result, "metrics": metrics}
                except Exception as exc:
                    segment["methods"][name] = {"status": "error", "error_type": type(exc).__name__,
                                                "error": str(exc), "traceback": traceback.format_exc()}
            records["segments"].append(segment)
            (outdir / "RESULTS.json").write_text(json.dumps(_jsonable(records), indent=2, allow_nan=False))
            print(index, {name: method.get("status") for name, method in segment["methods"].items()}, flush=True)
    finally:
        # QPReplayBridge has no public close; each query owns/closes its runtime.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
