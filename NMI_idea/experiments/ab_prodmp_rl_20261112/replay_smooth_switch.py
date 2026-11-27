"""Offline native-QP replay around the diagnosed reversal, on the repaired reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from audit_trajectory import DynamicScan, load_rows
from smooth_switch import MotionReference, SmoothAtlas, learned_shapes
from validate_smooth_switch import GEOMETRY, HERE, OUT, SEED, ReferenceRecorder

REPLAY = HERE.parent / "ab_validation_20261110" / "repair_20261110" / "qp_cross_track_fix"
sys.path.insert(0, str(REPLAY))
from scan_replay import ScanReplayBridge  # noqa: E402


class ReferenceWindow:
    def __init__(self, segments):
        self.segments = segments
        self.times = np.r_[0., np.cumsum([s.duration for s in segments])]
        self.origin = 0.
        self.euler_order = "xyz"

    def duration_s(self):
        return float(self.times[-1])

    def set_origin(self, pose, *, t_s=None):
        if t_s is not None:
            self.origin = float(t_s)

    def sample(self, time_s):
        elapsed = float(np.clip(time_s - self.origin, 0., self.times[-1]))
        index = min(len(self.segments) - 1, int(np.searchsorted(self.times, elapsed, side="right")) - 1)
        current = self.segments[index].sample(elapsed - self.times[index], self.euler_order)
        return MotionReference(current.pose_d, current.vel_ff, float(time_s))


class ExactReferenceReplay(ScanReplayBridge):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def make_pose_reference(self, tcp, speed_m_s, cfg):
        self.reference.euler_order = cfg.euler_order
        return self.reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-s", type=float, default=6.)
    parser.add_argument("--first-capture", type=int, default=13)
    parser.add_argument("--last-capture", type=int, default=16)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if not np.isfinite(args.slot_s) or args.slot_s <= 0.:
        parser.error("--slot-s must be positive and finite")
    case = load_rows([HERE / "eval_q_24"], [])[SEED]["rl"]
    with np.load(GEOMETRY, allow_pickle=False) as geometry:
        scan, atlas = DynamicScan(geometry), SmoothAtlas(geometry)
    # The original acquisitions 13, 14, 15, 16 include one along-surface
    # advance and the same-site 15° -> 19° -> 15° turn shown in the audit.
    record = ReferenceRecorder(atlas, learned_shapes(scan.primitive), scan.moving, args.slot_s)
    first, last = args.first_capture, args.last_capture
    if not 1 <= first < last <= len(case["steps"]):
        parser.error("the replay requires at least two capture indices in the logged range")
    steps = case["steps"][first - 1:last]
    states = [(s["pose"], *s["action"]) for s in steps]
    for previous, goal, step in zip(states, states[1:], steps[1:]):
        record.append(previous, goal, step["path_skill"])
    reference = ReferenceWindow(record.segments)
    bridge = ExactReferenceReplay(reference)
    q_seed = scan.q[states[0]]
    requested = np.asarray([atlas.targets[s] for s in states])
    result = bridge.query_pose_path(
        requested, q_seed, speed_m_s=.02, settle_s=0.,
        provenance=f"offline C2 ProDMP reference, seed {SEED}, captures {first}-{last}",
        return_trace=True,
    )
    trace = result.pop("trace", [])
    report = {
        "seed": SEED,
        "capture_indices": list(range(first, last + 1)),
        "capture_states": [list(s) for s in states],
        "slot_s": args.slot_s,
        "duration_s": reference.duration_s(),
        "samples": len(trace),
        "failures": result["failures"],
        "native_profile": str(bridge.profile_path),
        "profile_and_model_fingerprint": bridge._state_fingerprint(),
        "scope": "continuous offline QP and ideal joint feedback; no physical contact or ultrasound",
    }
    if trace:
        t = np.asarray([row["t_s"] for row in trace])
        expected = np.asarray([row["pose_desired"] for row in trace])
        commanded = [reference.sample(float(time)) for time in t]
        expected_position = np.asarray([r.pose_d[:3] for r in commanded])
        expected_rotation = Rotation.from_euler(reference.euler_order, np.asarray([r.pose_d[3:] for r in commanded]))
        feedforward = np.asarray([r.vel_ff for r in commanded])
        actual = np.asarray([bridge.assets.kin.fk_placement(row["q_meas_before"]).homogeneous for row in trace])
        position = 1000. * np.linalg.norm(expected[:, :3, 3] - actual[:, :3, 3], axis=1)
        angular = np.rad2deg((Rotation.from_matrix(expected[:, :3, :3]).inv()
                              * Rotation.from_matrix(actual[:, :3, :3])).magnitude())
        reported_rotation = Rotation.from_matrix(expected[:, :3, :3])
        capture_ids = [int(np.argmin(abs(t - c))) if c <= t[-1] + 1.e-10 else None for c in reference.times]
        switches = np.any(abs(t[:, None] - reference.times[1:-1]) <= .5, axis=1)
        report.update({
            "last_time_s": float(t[-1]),
            "position_tracking_mm_p95_max": np.percentile(position, [95, 100]).tolist(),
            "rotation_tracking_deg_p95_max": np.percentile(angular, [95, 100]).tolist(),
            "switch_window_position_tracking_mm_max": float(position[switches].max()) if switches.any() else None,
            "switch_window_rotation_tracking_deg_max": float(angular[switches].max()) if switches.any() else None,
            "capture_position_tracking_mm": [None if i is None else float(position[i]) for i in capture_ids],
            "capture_rotation_tracking_deg": [None if i is None else float(angular[i]) for i in capture_ids],
            "reference_position_difference_mm_max": float(1000. * np.linalg.norm(
                expected[:, :3, 3] - expected_position, axis=1,
            ).max()),
            "reference_rotation_difference_deg_max": float(np.rad2deg((expected_rotation.inv() * reported_rotation).magnitude().max())),
            "peak_ff_linear_speed_mm_s": float(1000. * np.linalg.norm(feedforward[:, :3], axis=1).max()),
            "qp_status_counts": result["qp1_status_counts"],
            "qp_pauses": int(result["task_paused_count"]),
            "min_self_collision_m": float(result["minimum_self_collision_distance_m"]),
            "position_orientation_gate": bool(not report["failures"] and t[-1] >= reference.duration_s() - 1.e-10
                                              and position.max() <= 2. and angular.max() <= 2.),
        })
        if report["failures"]:
            failed = trace[-1]
            measured = np.asarray(failed["q_meas_before"])
            kin = bridge.assets.kin
            report["last_tick_joint_margin"] = np.minimum(measured - kin.q_lower, kin.q_upper - measured).tolist()
            report["last_tick_box_width"] = (np.asarray(failed["box_hi"]) - np.asarray(failed["box_lo"])).tolist()
            report["last_tick_qdot_prev"] = np.asarray(failed["qdot_prev_used"]).tolist()
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / "NATIVE_SWITCH_TRACE.npz", time_s=t,
                            reference=expected, actual=actual, twist_ff=feedforward,
                            position_error_mm=position, rotation_error_deg=angular,
                            q_measured=np.asarray([row["q_meas_before"] for row in trace]),
                            qdot=np.asarray([row["qdot"] for row in trace]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "NATIVE_SUMMARY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
