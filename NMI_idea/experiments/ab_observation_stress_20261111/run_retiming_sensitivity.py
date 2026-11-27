"""Fixed-time scaling diagnostic for the four ProDMP native short segments.

Only the declared duration is multiplied by 2 or 4.  The local endpoint,
normalized ProDMP shape, q seed, profile, native binary, and QP settings are
unchanged.  The existing 1x trace is read as the formal baseline; no planner
timing or acceptance threshold is changed here.
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
BASE = HERE / "motion_validation"
BASELINE_JSON = BASE / "native_results.json"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from motion import ProDMPReference, duration_matrix  # noqa: E402
from run_native_motion import ProDMPReplayBridge  # noqa: E402


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


class RetimedProDMPReplayBridge(ProDMPReplayBridge):
    """The same native bridge with only a fixed reference duration multiplier."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = float(factor)
        if not np.isfinite(self.factor) or self.factor <= 0.:
            raise ValueError("retiming factor must be finite and positive")

    def make_pose_reference(self, tcp, speed_m_s, cfg):
        base = super().make_pose_reference(tcp, speed_m_s, cfg)
        # For the declared zero endpoint velocities W is independent of tau,
        # so this is a pure time scaling of the same normalized spatial curve.
        return ProDMPReference(base.start, base.goal, base.anchor,
                               duration_s=base.duration_s() * self.factor,
                               start_velocity=base.start_velocity,
                               end_velocity=base.end_velocity,
                               euler_order=base.euler_order)


def _reference_matrix(sample):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", sample.pose_d[3:]).as_matrix()
    T[:3, 3] = sample.pose_d[:3]
    return T


def _metrics(trace, reference, goal, kin, result, factor):
    if not trace:
        return {"sample_count": 0, "factor": factor}
    t = np.asarray([r["t_s"] for r in trace], dtype=float)
    desired = np.asarray([r["pose_desired"] for r in trace], dtype=float)
    q_before_pose = np.asarray([kin.fk_placement(np.asarray(r["q_meas_before"], dtype=float)).homogeneous for r in trace])
    q_send_pose = np.asarray([r["pose_actual"] for r in trace], dtype=float)
    refs = [reference.sample(float(x)) for x in t]
    expected = np.asarray([_reference_matrix(r) for r in refs])
    ff = np.asarray([r.vel_ff for r in refs])
    requested = np.asarray([r["twist_requested"] for r in trace])
    feasible = np.asarray([r["v_cmd_feasible"] for r in trace])
    received = np.asarray([r["v_cmd_received"] for r in trace])
    position_error = np.linalg.norm(q_before_pose[:, :3, 3] - expected[:, :3, 3], axis=1)
    legacy_position_error = np.linalg.norm(q_send_pose[:, :3, 3] - expected[:, :3, 3], axis=1)
    rotation_error = Rotation.from_matrix(np.transpose(expected[:, :3, :3], (0, 2, 1)) @ q_before_pose[:, :3, :3]).magnitude()
    legacy_rotation_error = Rotation.from_matrix(np.transpose(expected[:, :3, :3], (0, 2, 1)) @ q_send_pose[:, :3, :3]).magnitude()
    # Cross-track error is perpendicular to the declared world chord.  This is
    # a diagnostic direction only; it does not replace the full pose gate.
    chord = expected[-1, :3, 3] - expected[0, :3, 3]
    chord_norm = float(np.linalg.norm(chord))
    tangent = chord / chord_norm if chord_norm > 1.e-12 else np.array([1., 0., 0.])
    error_vector = q_before_pose[:, :3, 3] - expected[:, :3, 3]
    transverse = np.linalg.norm(error_vector - np.outer(error_vector @ tangent, tangent), axis=1)
    legacy_error_vector = q_send_pose[:, :3, 3] - expected[:, :3, 3]
    legacy_transverse = np.linalg.norm(legacy_error_vector - np.outer(legacy_error_vector @ tangent, tangent), axis=1)
    qdot = np.asarray([r["qdot"] for r in trace])
    box_lo = np.asarray([r["box_lo"] for r in trace])
    box_hi = np.asarray([r["box_hi"] for r in trace])
    active = t <= reference.duration_s() + 0.5 * np.median(np.diff(t))
    active = np.flatnonzero(active)
    if active.size >= 2:
        dpos = np.diff(q_before_pose[active, :3, 3], axis=0)
        dt = np.diff(t[active])
        actual_speed = np.linalg.norm(dpos, axis=1) / np.maximum(dt, 1.e-12)
        path_distance = float(np.sum(np.linalg.norm(np.diff(expected[active, :3, 3], axis=0), axis=1)))
        actual_distance = float(np.sum(np.linalg.norm(dpos, axis=1)))
        actual_effective = actual_distance / max(reference.duration_s(), 1.e-12)
        reference_effective = path_distance / max(reference.duration_s(), 1.e-12)
    else:
        actual_speed = np.zeros(1)
        actual_effective = 0.
        reference_effective = 0.
    return {
        "factor": float(factor),
        "sample_count": int(len(trace)),
        "reference_duration_s": float(reference.duration_s()),
        "trace_last_t_s": float(t[-1]),
        "trace_dt_s_median": float(np.median(np.diff(t))) if len(t) > 1 else None,
        "normalized_spatial_shape": "same endpoint ProDMP shape; duration only scaled",
        "reference_pose_reproduction_max_abs": float(np.max(np.abs(desired - expected))),
        "reference_ff_peak_linear_m_s": float(np.max(np.linalg.norm(ff[:, :3], axis=1))),
        "reference_ff_peak_angular_rad_s": float(np.max(np.linalg.norm(ff[:, 3:], axis=1))),
        "same_time_position_p95_m": float(np.percentile(position_error, 95)),
        "same_time_position_max_m": float(np.max(position_error)),
        "same_time_transverse_p95_m": float(np.percentile(transverse, 95)),
        "same_time_transverse_max_m": float(np.max(transverse)),
        "same_time_rotation_p95_rad": float(np.percentile(rotation_error, 95)),
        "same_time_rotation_max_rad": float(np.max(rotation_error)),
        "q_send_mismatch_position_p95_m": float(np.percentile(legacy_position_error, 95)),
        "q_send_mismatch_position_max_m": float(np.max(legacy_position_error)),
        "q_send_mismatch_transverse_p95_m": float(np.percentile(legacy_transverse, 95)),
        "q_send_mismatch_transverse_max_m": float(np.max(legacy_transverse)),
        "q_send_mismatch_rotation_p95_rad": float(np.percentile(legacy_rotation_error, 95)),
        "q_send_mismatch_rotation_max_rad": float(np.max(legacy_rotation_error)),
        "requested_to_feasible_projection_p95": float(np.percentile(np.linalg.norm(received - feasible, axis=1), 95)),
        "requested_to_feasible_projection_max": float(np.max(np.linalg.norm(received - feasible, axis=1))),
        "reference_ff_to_requested_residual_p95": float(np.percentile(np.linalg.norm(requested - ff, axis=1), 95)),
        "qdot_box_margin_min": float(np.min(np.minimum(qdot - box_lo, box_hi - qdot))),
        "velocity_margin_min": float(result.get("velocity_margin_min", np.nan)),
        "acceleration_margin_min": float(result.get("acceleration_margin_min", np.nan)),
        "jerk_margin_min": float(result.get("jerk_margin_min", np.nan)),
        "task_paused_count": int(result.get("task_paused_count", 0)),
        "solver_fault_count": int(sum(bool(r["solver_fault_latched"]) for r in trace)),
        "joint_limited_count": int(sum(bool(r["joint_limited"]) for r in trace)),
        "rail_limited_count": int(sum(bool(r["rail_limited"]) for r in trace)),
        "physical_saturated_count": int(sum(bool(r["physical_saturated"]) for r in trace)),
        "actual_speed_p95_m_s": float(np.percentile(actual_speed, 95)),
        "actual_speed_max_m_s": float(np.max(actual_speed)),
        "actual_effective_speed_m_s": float(actual_effective),
        "reference_effective_speed_m_s": float(reference_effective),
        "actual_to_reference_effective_speed_ratio": float(actual_effective / max(reference_effective, 1.e-12)),
        "native_motion_gate": bool(result.get("motion_tracking_pass_2mm_2deg", False)),
        "failures": result.get("failures", []),
    }


def _load_1x_metrics(index, goal, anchor, q_seed, kin, cfg, baseline):
    trace_path = BASE / f"prodmp_segment_{index}_trace.npz"
    if not trace_path.exists():
        raise FileNotFoundError(f"formal 1x trace missing: {trace_path}")
    archive = np.load(trace_path, allow_pickle=True)
    trace = [{key: archive[key][i] for key in archive.files} for i in range(len(archive["t_s"]))]
    duration = float(duration_matrix(np.vstack([np.zeros(3), goal]), "prodmp")[0, 1])
    reference = ProDMPReference(np.zeros(3), goal, anchor, duration_s=duration, euler_order=cfg.euler_order)
    result = baseline["segments"][index]["result"]
    return _metrics(trace, reference, goal, kin, result, 1.)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "retiming_sensitivity")
    parser.add_argument("--settle-s", type=float, default=.1)
    args = parser.parse_args()
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    public = json.loads(SELECTED.read_text())
    q_seed = np.asarray(public["q_seed"], dtype=float)
    anchor = np.asarray(public["T_world_body"], dtype=float)
    baseline = json.loads(BASELINE_JSON.read_text())
    # A kinematics object is used only for q_meas_before FK diagnostics.
    probe = ProDMPReplayBridge()
    kin, inner, cfg = probe.make_controller()
    inner.close()
    records = {
        "schema": "ab_observation_stress_prodmp_retiming_v1",
        "scope": "fixed-time scaling only; same spatial curve/q/profile/native binary; offline no hardware",
        "formal_1x_source": str(BASELINE_JSON),
        "profile": str(ROOT / "scan_tracking_profile.yaml"),
        "q_seed": q_seed,
        "T_world_body": anchor,
        "settle_s": float(args.settle_s),
        "factors": [1., 2., 4.],
        "segments": [],
    }
    for index, goal in enumerate(GOALS):
        segment = {"index": index, "goal_local": goal, "factors": {}}
        try:
            segment["factors"]["1.0"] = {"status": "reused_formal_baseline",
                "source_trace": str(BASE / f"prodmp_segment_{index}_trace.npz"),
                "metrics": _load_1x_metrics(index, goal, anchor, q_seed, kin, cfg, baseline)}
        except Exception as exc:
            segment["factors"]["1.0"] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        tcp = _tcp_pair(anchor, goal)
        base_duration = float(duration_matrix(np.vstack([np.zeros(3), goal]), "prodmp")[0, 1])
        for factor in (2., 4.):
            bridge = RetimedProDMPReplayBridge(factor)
            name = str(factor)
            try:
                result = bridge.query_pose_path(tcp, q_seed, speed_m_s=.02,
                                                settle_s=args.settle_s,
                                                provenance=f"ProDMP fixed time stretch {factor:g}x segment {index}",
                                                return_trace=True, return_state=False)
                trace = result.pop("trace", [])
                if trace:
                    np.savez_compressed(outdir / f"prodmp_{factor:g}x_segment_{index}_trace.npz",
                                        **{k: np.asarray([r[k] for r in trace]) for k in trace[0]})
                ref = ProDMPReference(np.zeros(3), goal, anchor,
                                      duration_s=base_duration * factor, euler_order=cfg.euler_order)
                segment["factors"][name] = {"status": "completed", "result": result,
                    "metrics": _metrics(trace, ref, goal, kin, result, factor),
                    "trace_file": str(outdir / f"prodmp_{factor:g}x_segment_{index}_trace.npz")}
            except Exception as exc:
                segment["factors"][name] = {"status": "error", "error_type": type(exc).__name__,
                                              "error": str(exc), "traceback": traceback.format_exc()}
            (outdir / "RESULTS.json").write_text(json.dumps(_jsonable(records | {"segments": records["segments"] + [segment]}), indent=2, allow_nan=False))
        records["segments"].append(segment)
        (outdir / "RESULTS.json").write_text(json.dumps(_jsonable(records), indent=2, allow_nan=False))
        print(index, {factor: item.get("status") for factor, item in segment["factors"].items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
