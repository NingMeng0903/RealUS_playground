"""Optional offline native-QP smoke test for the ProDMP reference.

This imports the already audited replay bridge and uses its deterministic model
clock.  It never opens a robot/device connection.  The native binary, URDF,
tool cache, posture preferences, and QP configuration are selected by the
existing ``scan_tracking_profile.yaml`` and are not edited here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
MOTION_ROOT = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_validation_20261110/repair_20261110/qp_cross_track_fix")
SELECTED = Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_module_ladder_20261110/planning/native_regression_20261110/public_prior_layout/SELECTED.json")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MOTION_ROOT))
from motion import ProDMPReference, duration_matrix  # noqa: E402
from scan_replay import ScanReplayBridge  # noqa: E402


class ProDMPReplayBridge(ScanReplayBridge):
    """Use the same replay/controller stack with an exact ProDMP source."""

    def make_pose_reference(self, tcp, speed_m_s, cfg):  # noqa: D401 - inherited contract
        del speed_m_s
        tcp = np.asarray(tcp, dtype=float)
        if tcp.ndim != 3 or tcp.shape[0] < 2:
            raise ValueError("ProDMP native smoke test requires at least two TCP poses")
        anchor = tcp[0].copy()
        relative = np.linalg.inv(anchor) @ tcp[-1]
        start = np.zeros(3)
        goal = np.array([
            relative[0, 3], relative[1, 3],
            Rotation.from_matrix(relative[:3, :3]).as_rotvec()[2],
        ], dtype=float)
        # The test's local actions are planar and have no roll/pitch.  Reject
        # accidental non-planar input rather than silently changing the chart.
        if abs(relative[2, 3]) > 1.e-8 or np.linalg.norm(relative[:3, :3] -
                Rotation.from_rotvec([0., 0., goal[2]]).as_matrix()) > 2.e-7:
            raise ValueError("native smoke input is outside the local planar chart")
        tau = float(duration_matrix(np.vstack([start, goal]), method="prodmp")[0, 1])
        return ProDMPReference(start, goal, anchor, duration_s=tau,
                               euler_order=cfg.euler_order)

    def _run_pose_path(self, *args, **kwargs):
        out = super()._run_pose_path(*args, **kwargs)
        out["actual_reference_kind"] = "ProDMPIntegralBasisReference"
        out["prodmp_contract"] = {
            "phase": "s=t/tau",
            "lambda": 5.0,
            "basis": "4 normalized RBFs, centers [0.12,0.36,0.64,0.88], width 55, exp(-3s) gate",
            "weights": "minimum-norm endpoint position and normalized-velocity solution",
            "twist": "analytic world-frame 6D geometric twist from fixed SE3 anchor",
        }
        return out


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native",), default="native")
    parser.add_argument("--output", type=Path, default=HERE / "motion_validation")
    parser.add_argument("--settle-s", type=float, default=0.3)
    args = parser.parse_args()
    if args.settle_s < 0.:
        raise ValueError("settle-s must be nonnegative")
    payload = json.loads(SELECTED.read_text())
    q_seed = np.asarray(payload["q_seed"], dtype=float)
    anchor = np.asarray(payload["T_world_body"], dtype=float)
    # All segments begin at the independently audited public FK state.  They
    # are intentionally short and independent so a failed controller proposal
    # cannot be hidden by a continuation replay.
    goals = np.asarray([
        [0.004, 0.000, 0.000],
        [0.000, 0.004, 0.080],
        [-0.004, 0.003, -0.080],
        [0.006, -0.004, 0.120],
    ], dtype=float)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    results = {
        "schema": "ab_observation_stress_prodmp_native_v1",
        "scope": "offline deterministic QP replay only; no hardware/device connection",
        "backend": args.backend,
        "source": str(HERE / "motion.py"),
        "public_layout": str(SELECTED),
        "q_seed": q_seed,
        "T_world_body": anchor,
        "goals_local": goals,
        "segments": [],
    }
    bridge = ProDMPReplayBridge()
    for idx, goal in enumerate(goals):
        T_goal = anchor.copy()
        T_goal[:3, :3] = anchor[:3, :3] @ Rotation.from_rotvec([0., 0., goal[2]]).as_matrix()
        T_goal[:3, 3] = anchor[:3, 3] + anchor[:3, :3] @ np.r_[goal[:2], 0.]
        tcp = np.stack([anchor, T_goal])
        rec = {"index": idx, "goal_local": goal, "duration_s": float(duration_matrix(np.vstack([np.zeros(3), goal]), method="prodmp")[0, 1])}
        try:
            result = bridge.query_pose_path(tcp, q_seed, speed_m_s=.02,
                                            settle_s=args.settle_s,
                                            provenance=f"ProDMP short local segment {idx}",
                                            return_trace=True, return_state=False)
            trace = result.pop("trace", [])
            if trace:
                np.savez_compressed(outdir / f"prodmp_segment_{idx}_trace.npz",
                                    **{k: np.asarray([r[k] for r in trace]) for k in trace[0]})
                rec["trace_file"] = str(outdir / f"prodmp_segment_{idx}_trace.npz")
            rec["result"] = result
            rec["status"] = "completed"
        except Exception as exc:  # honest optional validation result
            rec.update(status="error", error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
        results["segments"].append(rec)
        (outdir / "native_results.json").write_text(json.dumps(_jsonable(results), indent=2, allow_nan=False))
        print(idx, rec["status"], rec.get("result", {}).get("motion_tracking_pass_2mm_2deg"), flush=True)
    bridge.close() if hasattr(bridge, "close") else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
