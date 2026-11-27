"""Frozen paired native-QP task-weight diagnostic for the two branch-4 paths.

The runner intentionally owns only this experiment directory.  It uses the
already repaired offline bridge and changes one in-memory config field per
case; no production file or default is modified.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
REPAIR = HERE.parents[1]
DIRD = REPAIR / "dird"
INPUT = REPAIR / "audit" / "phase2_native_subset_inputs.npz"
REFERENCE_FOLLOWUP = REPAIR / "qp_reference_followup"
sys.path.insert(0, str(DIRD))
sys.path.insert(0, str(REFERENCE_FOLLOWUP))

import qp_replay as replay  # noqa: E402
import run_reference_protocol as reference_protocol  # noqa: E402
from common import clean_json  # noqa: E402
from rm75_control.control.joint_admittance_8dof import reference as reference_module  # noqa: E402
from rm75_control.control.joint_admittance_8dof.wbc_rt import client as native_client  # noqa: E402


SOURCE_REFERENCE = reference_protocol.SOURCE_REFERENCE
ANALYTIC_REFERENCE = reference_protocol.Analytic6DReference
BASELINE_WEIGHTS = np.asarray([100.0, 100.0, 100.0, 50.0, 50.0, 50.0], dtype=float)
TOLERANCE_UNIT_W = float(50.0 * (np.deg2rad(2.0) / 0.002) ** 2)
TOLERANCE_UNIT_WEIGHTS = np.asarray(
    [TOLERANCE_UNIT_W, TOLERANCE_UNIT_W, TOLERANCE_UNIT_W, 50.0, 50.0, 50.0],
    dtype=float,
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


class WeightedBridge(replay.QPReplayBridge):
    """QPReplayBridge with one frozen in-memory task-weight replacement."""

    def __init__(self, case: str, weights: np.ndarray) -> None:
        super().__init__("native", "urdf", native_clock="deterministic")
        self.case = str(case)
        self.requested_weights = np.asarray(weights, dtype=float).copy()
        self.raw_weight_before: np.ndarray | None = None
        self.cfg_weights: np.ndarray | None = None
        self.native_dump_path: Path | None = None
        self.native_dump_text: str | None = None
        self.native_dump_weight_line: str | None = None
        self.native_dump_weights: np.ndarray | None = None

    def make_controller(self):
        qp_raw = self.raw["inner"]["qp"]
        self.raw_weight_before = np.asarray(qp_raw["task_weight"], dtype=float).copy()
        if self.raw_weight_before.shape != (6,):
            raise ValueError(f"unexpected source task_weight shape: {self.raw_weight_before.shape}")
        if self.requested_weights.shape != (6,) or not np.isfinite(self.requested_weights).all():
            raise ValueError("requested task_weight must contain six finite values")

        # This is the only raw field changed by this bridge instance.  The
        # source mapping is loaded afresh for each case.
        qp_raw["task_weight"] = self.requested_weights.tolist()
        kin, inner, cfg = super().make_controller()
        self.cfg_weights = np.asarray(cfg.qp.task_weight, dtype=float).copy()
        if not np.array_equal(self.cfg_weights, self.requested_weights):
            raise AssertionError(
                f"cfg task_weight mismatch: {self.cfg_weights} != {self.requested_weights}"
            )

        native = getattr(inner, "_native", None)
        cfg_path = None if native is None else getattr(native, "_cfg_path", None)
        if cfg_path is None or not Path(cfg_path).is_file():
            raise AssertionError("native controller did not expose its dumped cfg path")
        dump_text = Path(cfg_path).read_text()
        lines = [line for line in dump_text.splitlines() if line.startswith("qp.task_weight ")]
        if len(lines) != 1:
            raise AssertionError(f"expected one qp.task_weight line in native cfg, got {len(lines)}")
        dump_line = lines[0]
        dump_weights = np.fromstring(dump_line.split(maxsplit=1)[1], sep=" ", dtype=float)
        if dump_weights.shape != (6,) or not np.array_equal(dump_weights, self.requested_weights):
            raise AssertionError(
                f"native cfg task_weight mismatch: {dump_weights} != {self.requested_weights}"
            )
        self.native_dump_text = dump_text
        self.native_dump_weight_line = dump_line
        self.native_dump_weights = dump_weights.copy()
        self.native_dump_path = HERE / f"{self.case}_native_config.txt"
        self.native_dump_path.write_text(dump_text)
        return kin, inner, cfg


def _reference_for_path(T: np.ndarray, speed_m_s: float = 0.02) -> ANALYTIC_REFERENCE:
    points = np.asarray(T[:, :3, 3], dtype=float)
    rpy = Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz")
    return ANALYTIC_REFERENCE(
        points,
        rpy=rpy,
        speed_m_s=speed_m_s,
        soft_start=True,
        ramp_s=0.4,
        euler_order="xyz",
    )


def _tangent_at_time(ref: ANALYTIC_REFERENCE, t_s: float) -> np.ndarray:
    t_eff = max(0.0, float(t_s) - float(ref._t_anchor))
    tau, _ = reference_module._soft_start_time_warp(t_eff, ref.ramp_s)
    s = min(ref._length, ref.speed_m_s * float(tau)) if ref.speed_m_s > 0.0 else 0.0
    _, _, tangent = ref._interp(s)
    tangent = np.asarray(tangent, dtype=float)
    if np.linalg.norm(tangent) <= 1.0e-12:
        return np.zeros(3, dtype=float)
    return tangent / np.linalg.norm(tangent)


def _augment_error_decomposition(T: np.ndarray, trace: list[dict[str, Any]]) -> None:
    """Keep signed along error and vector cross-path error in every row."""

    ref = _reference_for_path(T)
    for row in trace:
        actual = np.asarray(row["pose_actual"], dtype=float)
        desired = np.asarray(row["pose_desired"], dtype=float)
        translation_error = actual[:3, 3] - desired[:3, 3]
        tangent = _tangent_at_time(ref, row["t_s"])
        along = float(np.dot(translation_error, tangent)) if np.linalg.norm(tangent) else 0.0
        cross = translation_error - along * tangent
        rotvec = Rotation.from_matrix(desired[:3, :3].T @ actual[:3, :3]).as_rotvec()
        row["path_tangent"] = tangent
        row["along_track_error_m"] = along
        row["cross_path_normal_error_vec_m"] = cross
        row["cross_path_normal_error_m"] = float(np.linalg.norm(cross))
        row["rotation_error_rotvec_rad"] = np.asarray(rotvec, dtype=float)


def _save_trace(trace: list[dict[str, Any]], path: Path) -> str:
    if not trace:
        np.savez_compressed(path, trace_empty=np.asarray([True], dtype=bool))
    else:
        fields = {
            key: np.asarray([row[key] for row in trace])
            for key in trace[0]
        }
        np.savez_compressed(path, **fields)
    return sha(path)


def _save_reference(T: np.ndarray, times: np.ndarray, path: Path) -> str:
    ref = _reference_for_path(T)
    poses, velocities = [], []
    for t_s in np.asarray(times, dtype=float):
        sample = ref.sample(float(t_s))
        poses.append(np.asarray(sample.pose_d, dtype=float))
        velocities.append(np.asarray(sample.vel_ff, dtype=float))
    np.savez_compressed(
        path,
        t_s=np.asarray(times, dtype=float),
        pose_d=np.asarray(poses, dtype=float).reshape((-1, 6)),
        vel_ff=np.asarray(velocities, dtype=float).reshape((-1, 6)),
    )
    return sha(path)


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return {"count": 0, "min": None, "max": None, "mean": None, "p05": None, "p95": None}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def _decomposition_summary(T: np.ndarray, raw: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {"path": {}, "full": {}}
    times = np.asarray([row["t_s"] for row in trace], dtype=float)
    path_mask = times < float(raw["nominal_path_duration_s"])
    along = np.asarray([row["along_track_error_m"] for row in trace], dtype=float)
    normal = np.asarray([row["cross_path_normal_error_m"] for row in trace], dtype=float)
    normal_vec = np.asarray([row["cross_path_normal_error_vec_m"] for row in trace], dtype=float)
    rotvec = np.asarray([row["rotation_error_rotvec_rad"] for row in trace], dtype=float)
    rotnorm = np.asarray([row["rotation_error_rad"] for row in trace], dtype=float)

    def group(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {"count": 0}
        out = {
            "count": int(mask.sum()),
            # Signed values are retained.  abs_p95 is supplemental only.
            "along_signed_m": _stats(along[mask]),
            "along_abs_p95_m": float(np.percentile(np.abs(along[mask]), 95)),
            "cross_path_normal_norm_m": _stats(normal[mask]),
            "cross_path_normal_vec_m": {
                axis: _stats(normal_vec[mask, i]) for i, axis in enumerate("xyz")
            },
            "rotation_norm_deg": _stats(np.degrees(rotnorm[mask])),
            "rotation_rotvec_deg": {
                axis: _stats(np.degrees(rotvec[mask, i])) for i, axis in enumerate("xyz")
            },
        }
        return out

    return {"path": group(path_mask), "full": group(np.ones(len(trace), dtype=bool))}


def _trace_arrays(trace: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not trace:
        return {}
    return {
        "t_s": np.asarray([r["t_s"] for r in trace], dtype=float),
        "pose_actual": np.asarray([r["pose_actual"] for r in trace], dtype=float),
        "pose_desired": np.asarray([r["pose_desired"] for r in trace], dtype=float),
        "q": np.asarray([r["q"] for r in trace], dtype=float),
        "qdot": np.asarray([r["qdot"] for r in trace], dtype=float),
    }


def _rotation_angle(R: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))


def _same_reference_sensitivity(
    path: int,
    traces: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    baseline = traces.get(f"path{path}_baseline", {})
    tolerance = traces.get(f"path{path}_tolerance_unit", {})
    n = int(min(len(baseline.get("t_s", [])), len(tolerance.get("t_s", []))))
    out: dict[str, Any] = {
        "path": path,
        "shared_tick_count": n,
        "baseline_tick_count": int(len(baseline.get("t_s", []))),
        "tolerance_unit_tick_count": int(len(tolerance.get("t_s", []))),
        "reference_time_max_abs_diff_s": None,
        "desired_pose_max_translation_diff_m": None,
        "desired_pose_max_rotation_diff_deg": None,
        "actual_pose_max_translation_diff_m": None,
        "actual_pose_max_rotation_diff_deg": None,
        "q_command_max_abs_diff": None,
        "qdot_max_abs_diff": None,
    }
    if n <= 0:
        return out
    bt, tt = baseline["t_s"][:n], tolerance["t_s"][:n]
    out["reference_time_max_abs_diff_s"] = float(np.max(np.abs(bt - tt)))
    desired_b, desired_t = baseline["pose_desired"][:n], tolerance["pose_desired"][:n]
    actual_b, actual_t = baseline["pose_actual"][:n], tolerance["pose_actual"][:n]
    out["desired_pose_max_translation_diff_m"] = float(
        np.max(np.linalg.norm(desired_b[:, :3, 3] - desired_t[:, :3, 3], axis=1))
    )
    out["desired_pose_max_rotation_diff_deg"] = float(
        np.degrees(
            max(
                _rotation_angle(desired_b[i, :3, :3].T @ desired_t[i, :3, :3])
                for i in range(n)
            )
        )
    )
    out["actual_pose_max_translation_diff_m"] = float(
        np.max(np.linalg.norm(actual_b[:, :3, 3] - actual_t[:, :3, 3], axis=1))
    )
    out["actual_pose_max_rotation_diff_deg"] = float(
        np.degrees(
            max(
                _rotation_angle(actual_b[i, :3, :3].T @ actual_t[i, :3, :3])
                for i in range(n)
            )
        )
    )
    out["q_command_max_abs_diff"] = float(np.max(np.abs(baseline["q"][:n] - tolerance["q"][:n])))
    out["qdot_max_abs_diff"] = float(np.max(np.abs(baseline["qdot"][:n] - tolerance["qdot"][:n])))
    return out


def _write_case_result(case: str, raw: dict[str, Any]) -> None:
    # The case JSON deliberately excludes its own hash to avoid a circular
    # digest.  RESULTS.json records the hash of these exact bytes.
    path = HERE / f"{case}_result.json"
    path.write_text(json.dumps(clean_json(raw), indent=2, allow_nan=False) + "\n")
    raw["raw_result_file"] = path.name
    raw["raw_result_sha256"] = sha(path)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def _report(result: dict[str, Any]) -> str:
    lines = [
        "Frozen paired native-QP task-weight diagnostic",
        "Date: 2026-11-10; offline ideal-q replay only; no hardware/default promotion.",
        "",
        f"Input SHA256: {result['input_sha256']}",
        f"Protocol SHA256: {result['protocol_sha256']}",
        f"Runner SHA256: {result['runner_sha256']}",
        f"QP replay SHA256: {result['qp_replay_sha256']}",
        f"Native binary SHA256: {result.get('native_binary_sha256')}",
        f"W = 50*(radians(2)/0.002)^2 = {TOLERANCE_UNIT_W:.17g}",
        "Reference: analytic full 6D pose/Slerp feed-forward, 0.02 m/s, 0.4 s soft-start, 2.0 s settle.",
        "Only bridge.raw['inner']['qp']['task_weight'] changed; no limits, gains, ramp, or defaults changed.",
        "",
        "Per-case results (signed along-track and signed cross-vector components are retained):",
    ]
    for raw in result["runs"]:
        case = raw.get("case", "?")
        cross = raw.get("cross_path_metrics", {})
        path_m = cross.get("path", {})
        full_m = cross.get("full", {})
        along_p = path_m.get("along_signed_m", {})
        along_f = full_m.get("along_signed_m", {})
        normal_p = path_m.get("cross_path_normal_norm_m", {})
        normal_f = full_m.get("cross_path_normal_norm_m", {})
        rot_p = path_m.get("rotation_norm_deg", {})
        rot_f = full_m.get("rotation_norm_deg", {})
        rot_vec_p = path_m.get("rotation_rotvec_deg", {})
        rot_vec_f = full_m.get("rotation_rotvec_deg", {})
        lines.extend(
            [
                "",
                f"{case}: weight={raw.get('weight_case')} path={raw.get('path_index')} "
                f"n={raw.get('n_steps', 0)} whole_duration_s={_fmt(raw.get('whole_motion_duration_s'))} "
                f"nominal_path_s={_fmt(raw.get('nominal_path_duration_s'))}",
                f"  cfg.qp.task_weight={raw.get('cfg_task_weight')} native_dump={raw.get('native_dump_task_weight')}",
                f"  path position p95/max mm={_fmt((raw.get('path_position_p95_m') or 0)*1000)}/"
                f"{_fmt((raw.get('path_position_max_m') or 0)*1000)}; full p95/max mm="
                f"{_fmt((raw.get('full_position_p95_m') or 0)*1000)}/{_fmt((raw.get('full_position_max_m') or 0)*1000)}",
                f"  path rotation p95/max deg={_fmt(rot_p.get('p95'))}/{_fmt(rot_p.get('max'))}; "
                f"full p95/max deg={_fmt(rot_f.get('p95'))}/{_fmt(rot_f.get('max'))}",
                f"  path along signed min/max mm={_fmt((along_p.get('min') or 0)*1000)}/{_fmt((along_p.get('max') or 0)*1000)} "
                f"abs-p95={_fmt((path_m.get('along_abs_p95_m') or 0)*1000)}; "
                f"normal norm p95/max mm={_fmt((normal_p.get('p95') or 0)*1000)}/{_fmt((normal_p.get('max') or 0)*1000)}",
                f"  full along signed min/max mm={_fmt((along_f.get('min') or 0)*1000)}/{_fmt((along_f.get('max') or 0)*1000)} "
                f"normal norm p95/max mm={_fmt((normal_f.get('p95') or 0)*1000)}/{_fmt((normal_f.get('max') or 0)*1000)}",
                f"  mechanical margins min q/v/a/j/collision="
                f"{_fmt(np.min(raw.get('joint_limit_min_margin_by_coordinate')) if raw.get('joint_limit_min_margin_by_coordinate') is not None else None)}/"
                f"{_fmt(raw.get('velocity_margin_min'))}/{_fmt(raw.get('acceleration_margin_min'))}/"
                f"{_fmt(raw.get('jerk_margin_min'))}/{_fmt(raw.get('minimum_self_collision_distance_m'))}",
                f"  gates pose={raw.get('strict_pose_max_gate')} mechanical={raw.get('strict_mechanical_gate')} "
                f"all={raw.get('all_gate_strict')} failures={len(raw.get('failures', []))}",
                f"  trace_sha256={raw.get('trace_sha256')} raw_result_sha256={raw.get('raw_result_sha256')}",
            ]
        )
        normal_vec = path_m.get("cross_path_normal_vec_m", {})
        lines.append(
            "  path cross-vector component min/max mm: "
            + ", ".join(
                f"{axis}={_fmt((stats.get('min') or 0)*1000)}/{_fmt((stats.get('max') or 0)*1000)}"
                for axis, stats in normal_vec.items()
            )
        )
        lines.append(
            "  path rotation-vector component min/max deg: "
            + ", ".join(
                f"{axis}={_fmt(stats.get('min'))}/{_fmt(stats.get('max'))}"
                for axis, stats in rot_vec_p.items()
            )
        )
        lines.append(
            "  full rotation-vector component min/max deg: "
            + ", ".join(
                f"{axis}={_fmt(stats.get('min'))}/{_fmt(stats.get('max'))}"
                for axis, stats in rot_vec_f.items()
            )
        )
    lines.extend(["", "Same-reference paired sensitivity (baseline vs tolerance-unit):"])
    for sens in result.get("same_reference_sensitivity", []):
        lines.append(
            f"path{sens['path']}: shared_ticks={sens['shared_tick_count']} "
            f"reference_time_diff_s={_fmt(sens['reference_time_max_abs_diff_s'])} "
            f"desired_pose_diff_m/deg={_fmt(sens['desired_pose_max_translation_diff_m'])}/"
            f"{_fmt(sens['desired_pose_max_rotation_diff_deg'])} "
            f"actual_pose_diff_m/deg={_fmt(sens['actual_pose_max_translation_diff_m'])}/"
            f"{_fmt(sens['actual_pose_max_rotation_diff_deg'])} "
            f"q/qdot_diff={_fmt(sens['q_command_max_abs_diff'])}/{_fmt(sens['qdot_max_abs_diff'])}"
        )
    lines.extend(
        [
            "",
            "Interpretation: both weights use the same frozen full-6D reference and time law. "
            "Any shared-tick output difference is therefore attributable to task-weighted native-QP "
            "allocation/conditioning and numerical sensitivity, not a changed reference. W is a unit "
            "conversion diagnostic only, not a theoretically optimal tuning. Negative margins/errors "
            "remain visible; no failure was rerun.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not (HERE / "PROTOCOL.txt").is_file():
        raise FileNotFoundError("PROTOCOL.txt must be written before any native replay")
    frozen = np.load(INPUT, allow_pickle=False)
    source_hashes = {
        "input_sha256": sha(INPUT),
        "protocol_sha256": sha(HERE / "PROTOCOL.txt"),
        "runner_sha256": sha(Path(__file__)),
        "qp_replay_sha256": sha(Path(replay.__file__)),
        "production_reference_sha256": sha(Path(reference_module.__file__)),
        "config_sha256": sha(replay.CONFIG),
        "native_binary_sha256": (
            sha(native_client.find_wbc_rt_binary())
            if native_client.find_wbc_rt_binary() is not None
            else None
        ),
    }
    result: dict[str, Any] = {
        "schema": "frozen_native_qp_task_weight_ablation_v1",
        **source_hashes,
        "interpreter": sys.executable,
        "thread_environment": {
            key: __import__("os").environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "weights": {
            "baseline": BASELINE_WEIGHTS,
            "tolerance_unit": TOLERANCE_UNIT_WEIGHTS,
            "W": TOLERANCE_UNIT_W,
            "formula": "50*(radians(2)/0.002)^2",
        },
        "input_keys": [
            "branch4_path0_T",
            "branch4_path0_q_seed",
            "branch4_path1_T",
            "branch4_path1_q_seed",
        ],
        "order": [
            "path0_baseline",
            "path0_tolerance_unit",
            "path1_baseline",
            "path1_tolerance_unit",
        ],
        "runs": [],
    }
    traces: dict[str, dict[str, np.ndarray]] = {}

    for path_index in (0, 1):
        T = np.asarray(frozen[f"branch4_path{path_index}_T"], dtype=float)
        q = np.asarray(frozen[f"branch4_path{path_index}_q_seed"], dtype=float)
        for weight_case, weights in (
            ("baseline", BASELINE_WEIGHTS),
            ("tolerance_unit", TOLERANCE_UNIT_WEIGHTS),
        ):
            case = f"path{path_index}_{weight_case}"
            print(f"START {case} weights={weights.tolist()}", flush=True)
            bridge = WeightedBridge(case, weights)
            trace: list[dict[str, Any]] = []
            try:
                # All four runs use the same analytic full-6D reference class.
                replay.WorldPolylineReference = ANALYTIC_REFERENCE
                ANALYTIC_REFERENCE.samples = []
                raw = bridge.query_pose_path(
                    T,
                    q,
                    speed_m_s=0.02,
                    settle_s=2.0,
                    provenance=f"frozen paired task-weight diagnostic {case}",
                    return_trace=True,
                    return_state=False,
                )
                trace = raw.pop("trace", [])
            except Exception as exc:  # one attempt; retain failure and continue pair
                raw = {
                    "schema": "ultrapoc_motion_qp_path_v1",
                    "provenance": f"frozen paired task-weight diagnostic {case}",
                    "failures": [{"exception": type(exc).__name__, "reason": str(exc)}],
                    "exception": {"type": type(exc).__name__, "reason": str(exc)},
                    "n_steps": 0,
                }
            finally:
                replay.WorldPolylineReference = SOURCE_REFERENCE

            if trace:
                _augment_error_decomposition(T, trace)
            reference_protocol.summarize(raw, trace)
            if trace:
                raw["full_position_p95_m"] = float(
                    np.percentile(
                        np.asarray([row["position_error_m"] for row in trace], dtype=float),
                        95,
                    )
                )
            raw.update(
                {
                    "case": case,
                    "path_index": path_index,
                    "weight_case": weight_case,
                    "requested_task_weight": weights,
                    "cfg_task_weight": bridge.cfg_weights,
                    "raw_task_weight_before": bridge.raw_weight_before,
                    "native_dump_task_weight": bridge.native_dump_weights,
                    "native_dump_weight_line": bridge.native_dump_weight_line,
                    "native_dump_file": None
                    if bridge.native_dump_path is None
                    else bridge.native_dump_path.name,
                    "native_dump_sha256": None
                    if bridge.native_dump_path is None
                    else sha(bridge.native_dump_path),
                    "reference_class": ANALYTIC_REFERENCE.__name__,
                    "reference_speed_m_s": 0.02,
                    "reference_soft_start_ramp_s": 0.4,
                    "settle_s": 2.0,
                    "analytic_sample_count": len(ANALYTIC_REFERENCE.samples),
                    "whole_motion_duration_s": (
                        float(raw.get("integrated_duration_s", 0.0)) if trace else 0.0
                    ),
                }
            )
            raw["cross_path_metrics"] = _decomposition_summary(T, raw, trace)

            trace_path = HERE / f"{case}_trace.npz"
            raw["trace_file"] = trace_path.name
            raw["trace_sha256"] = _save_trace(trace, trace_path)
            if trace:
                reference_path = HERE / f"{case}_reference.npz"
                raw["reference_file"] = reference_path.name
                raw["reference_sha256"] = _save_reference(
                    T,
                    np.asarray([row["t_s"] for row in trace], dtype=float),
                    reference_path,
                )
            else:
                raw["reference_file"] = None
                raw["reference_sha256"] = None
            _write_case_result(case, raw)
            result["runs"].append(raw)
            traces[case] = _trace_arrays(trace)
            (HERE / "RESULTS.json").write_text(
                json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n"
            )
            print(
                f"DONE {case} n={raw.get('n_steps', 0)} "
                f"duration={raw.get('whole_motion_duration_s')} "
                f"pose_gate={raw.get('strict_pose_max_gate')} "
                f"mechanical_gate={raw.get('strict_mechanical_gate')} "
                f"failures={len(raw.get('failures', []))}",
                flush=True,
            )

    result["same_reference_sensitivity"] = [
        _same_reference_sensitivity(path, traces) for path in (0, 1)
    ]
    (HERE / "RESULTS.json").write_text(
        json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n"
    )
    results_sha = sha(HERE / "RESULTS.json")
    (HERE / "RESULTS.sha256").write_text(results_sha + "  RESULTS.json\n")
    (HERE / "REPORT.txt").write_text(_report(result))
    print(f"RESULTS_SHA256 {results_sha}", flush=True)


if __name__ == "__main__":
    main()
