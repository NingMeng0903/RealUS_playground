"""Frozen W-weight + software-lead/timing combination diagnostic.

Only this directory is owned by the experiment.  The native replay and
production configuration are imported read-only; requested changes are made
on each bridge's in-memory raw mapping before its controller is built.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
CROSS_TRACK_FIX = HERE.parents[1]
REPAIR = HERE.parents[2]
DIRD = REPAIR / "dird"
INPUT = REPAIR / "audit" / "phase2_native_subset_inputs.npz"
PARENT_WEIGHT_RESULTS = CROSS_TRACK_FIX / "weight_ablation" / "RESULTS.json"
sys.path.insert(0, str(DIRD))
sys.path.insert(0, str(CROSS_TRACK_FIX))

import run_relaxed_execution as relaxed  # noqa: E402
from common import clean_json  # noqa: E402
from rm75_control.control.joint_admittance_8dof import reference as reference_module  # noqa: E402
from rm75_control.control.joint_admittance_8dof.wbc_rt import client as native_client  # noqa: E402


OLD = relaxed.lead.old
REPLAY = OLD.replay
ANALYTIC_REFERENCE = OLD.Analytic6DReference
SYMMETRIC08 = relaxed.Symmetric08
BASELINE_W = float(50.0 * (np.deg2rad(2.0) / 0.002) ** 2)
TASK_WEIGHTS = np.asarray([BASELINE_W, BASELINE_W, BASELINE_W, 50.0, 50.0, 50.0], dtype=float)
LEAD_DEG = 20.0
LEAD_RAD = float(np.deg2rad(LEAD_DEG))
SPEED_M_S = 0.02
SETTLE_S = 2.0
START_RAMP_S = 0.4
SYMMETRIC_RAMP_S = 0.8
MIDDLE_EDGE_S = 0.8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_factory(T: np.ndarray, reference_kind: str):
    cls = ANALYTIC_REFERENCE if reference_kind == "A_start04" else SYMMETRIC08
    points = np.asarray(T[:, :3, 3], dtype=float)
    rpy = Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz")
    return cls(
        points,
        rpy=rpy,
        speed_m_s=SPEED_M_S,
        soft_start=True,
        ramp_s=START_RAMP_S,
        euler_order="xyz",
    )


def _arc_s_and_speed(ref, t_s: float) -> tuple[float, float]:
    if hasattr(ref, "time_law"):
        s, speed = ref.time_law(float(t_s))
        return float(np.clip(s, 0.0, ref._length)), float(speed)
    t_eff = max(0.0, float(t_s) - float(ref._t_anchor))
    tau, tau_dot = reference_module._soft_start_time_warp(t_eff, ref.ramp_s)
    s = min(ref._length, ref.speed_m_s * float(tau)) if ref.speed_m_s > 0.0 else 0.0
    speed = ref.speed_m_s * float(tau_dot) if s < ref._length else 0.0
    return float(s), float(speed)


def _arc_fraction(ref, t_s: float) -> float:
    if ref._length <= 1.0e-12:
        return 1.0
    s, _ = _arc_s_and_speed(ref, t_s)
    return float(np.clip(s / ref._length, 0.0, 1.0))


def _tangent(ref, t_s: float) -> np.ndarray:
    s, _ = _arc_s_and_speed(ref, t_s)
    _, _, tangent = ref._interp(s)
    tangent = np.asarray(tangent, dtype=float)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-12:
        return np.zeros(3, dtype=float)
    return tangent / norm


class WeightedLeadBridge(REPLAY.QPReplayBridge):
    """Apply only the frozen W task weight and 20-degree software lead."""

    def __init__(self, case: str) -> None:
        super().__init__("native", "urdf", native_clock="deterministic")
        self.case = str(case)
        self.raw_weight_before: np.ndarray | None = None
        self.raw_lead_before: float | None = None
        self.cfg_weights: np.ndarray | None = None
        self.cfg_lead_rad: float | None = None
        self.native_dump_path: Path | None = None
        self.native_dump_weights: np.ndarray | None = None
        self.native_dump_lead_rad: float | None = None
        self.native_dump_weight_line: str | None = None
        self.native_dump_lead_line: str | None = None

    def make_controller(self):
        qp_raw = self.raw["inner"]["qp"]
        hard_raw = self.raw["qpik"]["hard_limits"]
        self.raw_weight_before = np.asarray(qp_raw["task_weight"], dtype=float).copy()
        self.raw_lead_before = float(hard_raw["command_lead_arm_deg"])
        if self.raw_weight_before.shape != (6,):
            raise ValueError(f"unexpected task_weight shape {self.raw_weight_before.shape}")
        if abs(self.raw_lead_before - 6.0) > 1.0e-12:
            raise ValueError(f"unexpected frozen source lead {self.raw_lead_before}")

        # These are the only two in-memory raw changes for this case.
        qp_raw["task_weight"] = TASK_WEIGHTS.tolist()
        hard_raw["command_lead_arm_deg"] = LEAD_DEG
        kin, inner, cfg = super().make_controller()
        self.cfg_weights = np.asarray(cfg.qp.task_weight, dtype=float).copy()
        self.cfg_lead_rad = float(cfg.resync_err_rad)
        if not np.array_equal(self.cfg_weights, TASK_WEIGHTS):
            raise AssertionError(f"cfg task weight mismatch: {self.cfg_weights}")
        if not np.isclose(self.cfg_lead_rad, LEAD_RAD, rtol=0.0, atol=1.0e-12):
            raise AssertionError(f"cfg lead mismatch: {self.cfg_lead_rad} != {LEAD_RAD}")

        native = getattr(inner, "_native", None)
        cfg_path = None if native is None else getattr(native, "_cfg_path", None)
        if cfg_path is None or not Path(cfg_path).is_file():
            raise AssertionError("native cfg dump path unavailable")
        dump_text = Path(cfg_path).read_text()
        weight_lines = [line for line in dump_text.splitlines() if line.startswith("qp.task_weight ")]
        lead_lines = [line for line in dump_text.splitlines() if line.startswith("resync_err_rad ")]
        if len(weight_lines) != 1 or len(lead_lines) != 1:
            raise AssertionError("native cfg missing unique task-weight/lead lines")
        weight_line, lead_line = weight_lines[0], lead_lines[0]
        dump_weights = np.fromstring(weight_line.split(maxsplit=1)[1], sep=" ", dtype=float)
        dump_lead = float(lead_line.split(maxsplit=1)[1])
        if dump_weights.shape != (6,) or not np.array_equal(dump_weights, TASK_WEIGHTS):
            raise AssertionError(f"native cfg task weight mismatch: {dump_weights}")
        if not np.isclose(dump_lead, LEAD_RAD, rtol=0.0, atol=1.0e-12):
            raise AssertionError(f"native cfg lead mismatch: {dump_lead} != {LEAD_RAD}")
        self.native_dump_weights = dump_weights.copy()
        self.native_dump_lead_rad = dump_lead
        self.native_dump_weight_line = weight_line
        self.native_dump_lead_line = lead_line
        self.native_dump_path = HERE / f"{self.case}_native_config.txt"
        self.native_dump_path.write_text(dump_text)
        return kin, inner, cfg


def _augment_trace(T: np.ndarray, trace: list[dict[str, Any]], reference_kind: str) -> dict[str, Any]:
    ref = _reference_factory(T, reference_kind)
    duration = float(ref.duration_s())
    for row in trace:
        actual = np.asarray(row["pose_actual"], dtype=float)
        desired = np.asarray(row["pose_desired"], dtype=float)
        error = actual[:3, 3] - desired[:3, 3]
        tangent = _tangent(ref, row["t_s"])
        along = float(np.dot(error, tangent)) if np.linalg.norm(tangent) else 0.0
        cross = error - along * tangent
        rotvec = Rotation.from_matrix(desired[:3, :3].T @ actual[:3, :3]).as_rotvec()
        row["reference_phase_t_s"] = float(row["t_s"])
        row["reference_arc_fraction"] = _arc_fraction(ref, row["t_s"])
        row["path_tangent"] = tangent
        row["along_track_error_m"] = along
        row["cross_path_normal_error_vec_m"] = cross
        row["cross_path_normal_error_m"] = float(np.linalg.norm(cross))
        row["rotation_error_rotvec_rad"] = np.asarray(rotvec, dtype=float)
        row["middle_phase_window"] = bool(
            MIDDLE_EDGE_S <= float(row["t_s"]) <= duration - MIDDLE_EDGE_S
        )
    return {"duration_s": duration, "reference": ref}


def _stats(values: np.ndarray) -> dict[str, Any]:
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


def _window_metrics(raw: dict[str, Any], trace: list[dict[str, Any]], ref_info: dict[str, Any]) -> dict[str, Any]:
    if not trace:
        return {}
    ref = ref_info["reference"]
    duration = float(ref_info["duration_s"])
    t = np.asarray([row["t_s"] for row in trace], dtype=float)
    path_mask = t < float(raw["nominal_path_duration_s"])
    middle_mask = (t >= MIDDLE_EDGE_S) & (t <= duration - MIDDLE_EDGE_S)
    arc = np.asarray([row["reference_arc_fraction"] for row in trace], dtype=float)
    pos = np.asarray([row["position_error_m"] for row in trace], dtype=float)
    rot = np.asarray([row["rotation_error_rad"] for row in trace], dtype=float)
    along = np.asarray([row["along_track_error_m"] for row in trace], dtype=float)
    normal = np.asarray([row["cross_path_normal_error_m"] for row in trace], dtype=float)
    normal_vec = np.asarray([row["cross_path_normal_error_vec_m"] for row in trace], dtype=float)
    rotvec = np.asarray([row["rotation_error_rotvec_rad"] for row in trace], dtype=float)
    jerk = np.asarray([row["command_jerk"] for row in trace], dtype=float)
    jerk_limits = np.asarray(raw["jerk_limit_by_coordinate"], dtype=float)
    arm_jerk = np.abs(jerk[:, 1:])
    rail_jerk = np.abs(jerk[:, 0])

    def group(mask: np.ndarray, *, label: str, fixed_window: str | None = None) -> dict[str, Any]:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return {"label": label, "count": 0, "fixed_window": fixed_window}
        out: dict[str, Any] = {
            "label": label,
            "count": int(mask.sum()),
            "fixed_window": fixed_window,
            "position_m": _stats(pos[mask]),
            "rotation_deg": _stats(np.degrees(rot[mask])),
            "along_signed_m": _stats(along[mask]),
            "along_abs_p95_m": float(np.percentile(np.abs(along[mask]), 95)),
            "cross_path_normal_norm_m": _stats(normal[mask]),
            "cross_path_normal_vec_m": {
                axis: _stats(normal_vec[mask, i]) for i, axis in enumerate("xyz")
            },
            "rotation_rotvec_deg": {
                axis: _stats(np.degrees(rotvec[mask, i])) for i, axis in enumerate("xyz")
            },
            "jerk_max_abs_arm_rad_s3": float(np.max(arm_jerk[mask])),
            "jerk_max_abs_rail_m_s3": float(np.max(rail_jerk[mask])),
            "jerk_margin_min_arm_rad_s3": float(
                np.min(jerk_limits[1:][None, :] - arm_jerk[mask])
            ),
            "jerk_margin_min_rail_m_s3": float(
                np.min(jerk_limits[0] - rail_jerk[mask])
            ),
            "jerk_margin_min_all": float(
                np.min(jerk_limits[None, :] - np.abs(jerk[mask]))
            ),
            "jerk_excess_arm_coordinate_ticks": int(
                np.sum(arm_jerk[mask] > jerk_limits[1:][None, :])
            ),
            "jerk_excess_rail_ticks": int(np.sum(rail_jerk[mask] > jerk_limits[0])),
            "jerk_excess_max_arm_rad_s3": float(
                np.max(arm_jerk[mask] - jerk_limits[1:][None, :])
            ),
        }
        return out

    windows: dict[str, Any] = {
        "window_contract": {
            "middle_phase_t_ref_s": [MIDDLE_EDGE_S, duration - MIDDLE_EDGE_S],
            "arc20_80": [0.2, 0.8],
            "arc_bins": [[i / 10.0, (i + 1) / 10.0] for i in range(10)],
            "selection_rule": "fixed reference phase/arc windows, not error-selected",
        },
        "full": group(np.ones(len(trace), dtype=bool), label="full_motion"),
        "path": group(path_mask, label="path_only"),
        "middle_phase": group(
            middle_mask,
            label="middle_reference_phase",
            fixed_window=f"[{MIDDLE_EDGE_S},{duration - MIDDLE_EDGE_S}]s",
        ),
        "arc20_80": group(
            path_mask & (arc >= 0.2) & (arc <= 0.8),
            label="path_arc_20_80_percent",
            fixed_window="[0.2,0.8]",
        ),
        "arc_bins_10pct": {},
    }
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        upper = arc <= hi if i == 9 else arc < hi
        windows["arc_bins_10pct"][f"bin_{i:02d}_{(i + 1) * 10:02d}"] = group(
            path_mask & (arc >= lo) & upper,
            label=f"path_arc_{i * 10}_{(i + 1) * 10}_percent",
            fixed_window=f"[{lo},{hi}{']' if i == 9 else ')'}",
        )
    return windows


def _save_trace(trace: list[dict[str, Any]], path: Path) -> str:
    fields = {
        key: np.asarray([row[key] for row in trace]) for key in trace[0]
    }
    np.savez_compressed(path, **fields)
    return sha(path)


def _save_reference(ref, times: np.ndarray, path: Path) -> str:
    poses, velocities, arcs = [], [], []
    for t_s in np.asarray(times, dtype=float):
        sample = ref.sample(float(t_s))
        poses.append(np.asarray(sample.pose_d, dtype=float))
        velocities.append(np.asarray(sample.vel_ff, dtype=float))
        arcs.append(_arc_fraction(ref, float(t_s)))
    np.savez_compressed(
        path,
        t_s=np.asarray(times, dtype=float),
        pose_d=np.asarray(poses, dtype=float).reshape((-1, 6)),
        vel_ff=np.asarray(velocities, dtype=float).reshape((-1, 6)),
        arc_fraction=np.asarray(arcs, dtype=float),
    )
    return sha(path)


def _write_case_result(case: str, raw: dict[str, Any]) -> None:
    path = HERE / f"{case}_result.json"
    path.write_text(json.dumps(clean_json(raw), indent=2, allow_nan=False) + "\n")
    raw["raw_result_file"] = path.name
    raw["raw_result_sha256"] = sha(path)


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}g}"
    return str(v)


def _report(result: dict[str, Any]) -> str:
    lines = [
        "Frozen W-weight + software-lead/timing combination diagnostic",
        "Offline ideal-q native replay only; no hardware/default promotion.",
        "",
        f"Input SHA256: {result['input_sha256']}",
        f"Protocol SHA256: {result['protocol_sha256']}",
        f"Runner SHA256: {result['runner_sha256']}",
        f"QP replay SHA256: {result['qp_replay_sha256']}",
        f"Native binary SHA256: {result.get('native_binary_sha256')}",
        f"W = 50*(radians(2)/0.002)^2 = {BASELINE_W:.17g}; lead = {LEAD_DEG:g} deg ({LEAD_RAD:.17g} rad)",
        "All runs use full 6D feed-forward, 0.02 m/s peak, and 2 s settle.",
        "Physical q/v/a/j/collision gates are unchanged; only software lead is relaxed.",
        "",
        "Window metrics are fixed before results: full, path-only, middle t_ref [.8,duration-.8], arc 20-80%, and fixed 10% arc bins.",
    ]
    prior = result.get("comparison_to_W_only", {})
    if prior:
        lines.extend(["", "Sealed W-only comparison (parent weight_ablation; no rerun):"])
        for path, values in prior.items():
            lines.append(
                f"path{path}: prior W-only full arm jerk margin={_fmt(values.get('prior_W_only_full_jerk_margin_arm_rad_s3'))}; "
                f"lead20 A/B full margins={_fmt(values.get('A_start04_full_jerk_margin_arm_rad_s3'))}/"
                f"{_fmt(values.get('B_symmetric08_full_jerk_margin_arm_rad_s3'))}; "
                f"prior W-only full excess ticks={values.get('prior_W_only_full_jerk_excess_arm_coordinate_ticks')}"
            )
    for raw in result["runs"]:
        w = raw.get("window_metrics", {})
        lines.extend(
            [
                "",
                f"{raw.get('case')}: combo={raw.get('combination')} path={raw.get('path_index')} "
                f"n={raw.get('n_steps', 0)} whole_duration_s={_fmt(raw.get('whole_motion_duration_s'))} "
                f"reference_duration_s={_fmt(raw.get('reference_duration_s'))}",
                f"  weights={raw.get('cfg_task_weight')} native_weights={raw.get('native_dump_task_weight')} "
                f"lead_rad={_fmt(raw.get('effective_lead_rad'))} native_lead_rad={_fmt(raw.get('native_dump_lead_rad'))}",
                f"  cfg_dump={raw.get('native_dump_file')} cfg_sha256={raw.get('native_dump_sha256')}",
                f"  full pose p95/max mm={_fmt((w.get('full', {}).get('position_m', {}).get('p95') or 0)*1000)}/"
                f"{_fmt((w.get('full', {}).get('position_m', {}).get('max') or 0)*1000)}; "
                f"rot p95/max deg={_fmt(w.get('full', {}).get('rotation_deg', {}).get('p95'))}/"
                f"{_fmt(w.get('full', {}).get('rotation_deg', {}).get('max'))}",
                f"  middle pose p95/max mm={_fmt((w.get('middle_phase', {}).get('position_m', {}).get('p95') or 0)*1000)}/"
                f"{_fmt((w.get('middle_phase', {}).get('position_m', {}).get('max') or 0)*1000)}; "
                f"rot p95/max deg={_fmt(w.get('middle_phase', {}).get('rotation_deg', {}).get('p95'))}/"
                f"{_fmt(w.get('middle_phase', {}).get('rotation_deg', {}).get('max'))}",
                f"  middle along signed min/max mm={_fmt((w.get('middle_phase', {}).get('along_signed_m', {}).get('min') or 0)*1000)}/"
                f"{_fmt((w.get('middle_phase', {}).get('along_signed_m', {}).get('max') or 0)*1000)}; "
                f"normal p95/max mm={_fmt((w.get('middle_phase', {}).get('cross_path_normal_norm_m', {}).get('p95') or 0)*1000)}/"
                f"{_fmt((w.get('middle_phase', {}).get('cross_path_normal_norm_m', {}).get('max') or 0)*1000)}",
                f"  middle jerk arm max/margin/excess={_fmt(w.get('middle_phase', {}).get('jerk_max_abs_arm_rad_s3'))}/"
                f"{_fmt(w.get('middle_phase', {}).get('jerk_margin_min_arm_rad_s3'))}/"
                f"{w.get('middle_phase', {}).get('jerk_excess_arm_coordinate_ticks', '-')}; "
                f"full jerk arm max/margin/excess={_fmt(w.get('full', {}).get('jerk_max_abs_arm_rad_s3'))}/"
                f"{_fmt(w.get('full', {}).get('jerk_margin_min_arm_rad_s3'))}/"
                f"{w.get('full', {}).get('jerk_excess_arm_coordinate_ticks', '-')}",
                f"  strict gates pose={raw.get('strict_pose_max_gate')} mechanical={raw.get('strict_mechanical_gate')} "
                f"all={raw.get('all_gate_strict')} failures={len(raw.get('failures', []))}",
                f"  trace_sha256={raw.get('trace_sha256')} raw_result_sha256={raw.get('raw_result_sha256')}",
            ]
        )
        middle = w.get("middle_phase", {})
        vec = middle.get("cross_path_normal_vec_m", {})
        rotvec = middle.get("rotation_rotvec_deg", {})
        lines.append(
            "  middle cross-vector signed component min/max mm: "
            + ", ".join(
                f"{axis}={_fmt((stats.get('min') or 0)*1000)}/{_fmt((stats.get('max') or 0)*1000)}"
                for axis, stats in vec.items()
            )
        )
        lines.append(
            "  middle rotation-vector signed component min/max deg: "
            + ", ".join(
                f"{axis}={_fmt(stats.get('min'))}/{_fmt(stats.get('max'))}"
                for axis, stats in rotvec.items()
            )
        )
        lines.append(
            "  arc20-80 pose p95/max mm="
            f"{_fmt((w.get('arc20_80', {}).get('position_m', {}).get('p95') or 0)*1000)}/"
            f"{_fmt((w.get('arc20_80', {}).get('position_m', {}).get('max') or 0)*1000)}; "
            f"normal p95/max mm={_fmt((w.get('arc20_80', {}).get('cross_path_normal_norm_m', {}).get('p95') or 0)*1000)}/"
            f"{_fmt((w.get('arc20_80', {}).get('cross_path_normal_norm_m', {}).get('max') or 0)*1000)}; "
            f"rot p95/max deg={_fmt(w.get('arc20_80', {}).get('rotation_deg', {}).get('p95'))}/"
            f"{_fmt(w.get('arc20_80', {}).get('rotation_deg', {}).get('max'))}"
        )
        bins = w.get("arc_bins_10pct", {})
        lines.append(
            "  arc10% bins (position max mm / normal max mm / rotation max deg / arm jerk margin): "
            + "; ".join(
                f"{name}={_fmt((m.get('position_m', {}).get('max') or 0)*1000)}/"
                f"{_fmt((m.get('cross_path_normal_norm_m', {}).get('max') or 0)*1000)}/"
                f"{_fmt(m.get('rotation_deg', {}).get('max'))}/"
                f"{_fmt(m.get('jerk_margin_min_arm_rad_s3'))}"
                for name, m in bins.items()
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: A and B retain the same frozen geometry, W task weight, lead20, and physical gates; only the declared reference time law differs. "
            "The middle/arc windows are fixed profile windows, not error-selected. Negative signed errors and jerk margins are intentionally preserved. "
            "A middle-window improvement would not certify startup/end behavior or physical execution.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    if not (HERE / "PROTOCOL.txt").is_file():
        raise FileNotFoundError("PROTOCOL.txt must be present before native runs")
    frozen = np.load(INPUT, allow_pickle=False)
    native_binary = native_client.find_wbc_rt_binary()
    result: dict[str, Any] = {
        "schema": "frozen_w_weight_lead_timing_combination_v1",
        "protocol_sha256": sha(HERE / "PROTOCOL.txt"),
        "runner_sha256": sha(Path(__file__)),
        "input_sha256": sha(INPUT),
        "qp_replay_sha256": sha(Path(REPLAY.__file__)),
        "production_reference_sha256": sha(Path(reference_module.__file__)),
        "native_binary_sha256": None if native_binary is None else sha(native_binary),
        "parent_weight_ablation_results_sha256": (
            sha(PARENT_WEIGHT_RESULTS) if PARENT_WEIGHT_RESULTS.is_file() else None
        ),
        "interpreter": sys.executable,
        "task_weights": TASK_WEIGHTS,
        "W": BASELINE_W,
        "lead_deg": LEAD_DEG,
        "lead_rad": LEAD_RAD,
        "speed_m_s": SPEED_M_S,
        "settle_s": SETTLE_S,
        "order": [
            "path0_A_start04",
            "path0_B_symmetric08",
            "path1_A_start04",
            "path1_B_symmetric08",
        ],
        "runs": [],
    }

    for path_index in (0, 1):
        T = np.asarray(frozen[f"branch4_path{path_index}_T"], dtype=float)
        q = np.asarray(frozen[f"branch4_path{path_index}_q_seed"], dtype=float)
        for combination, reference_kind in (("A_start04", "A_start04"), ("B_symmetric08", "B_symmetric08")):
            case = f"path{path_index}_{combination}"
            print(f"START {case} weights={TASK_WEIGHTS.tolist()} lead_deg={LEAD_DEG:g}", flush=True)
            bridge = WeightedLeadBridge(case)
            trace: list[dict[str, Any]] = []
            try:
                REPLAY.WorldPolylineReference = ANALYTIC_REFERENCE if reference_kind == "A_start04" else SYMMETRIC08
                ANALYTIC_REFERENCE.samples = []
                SYMMETRIC08.samples = []
                raw = bridge.query_pose_path(
                    T,
                    q,
                    speed_m_s=SPEED_M_S,
                    settle_s=SETTLE_S,
                    provenance=f"frozen W-weight lead20 timing combination {case}",
                    return_trace=True,
                    return_state=False,
                )
                trace = raw.pop("trace", [])
            except Exception as exc:  # one attempt; preserve failure and continue
                raw = {
                    "schema": "ultrapoc_motion_qp_path_v1",
                    "provenance": f"frozen W-weight lead20 timing combination {case}",
                    "failures": [{"exception": type(exc).__name__, "reason": str(exc)}],
                    "exception": {"type": type(exc).__name__, "reason": str(exc)},
                    "n_steps": 0,
                }
            finally:
                REPLAY.WorldPolylineReference = ANALYTIC_REFERENCE

            ref_info = _augment_trace(T, trace, reference_kind) if trace else {
                "duration_s": None,
                "reference": _reference_factory(T, reference_kind),
            }
            OLD.summarize(raw, trace)
            if trace:
                raw["full_position_p95_m"] = float(
                    np.percentile([row["position_error_m"] for row in trace], 95)
                )
            raw.update(
                {
                    "case": case,
                    "path_index": path_index,
                    "combination": combination,
                    "reference_kind": reference_kind,
                    "requested_task_weight": TASK_WEIGHTS,
                    "cfg_task_weight": bridge.cfg_weights,
                    "raw_task_weight_before": bridge.raw_weight_before,
                    "native_dump_task_weight": bridge.native_dump_weights,
                    "native_dump_weight_line": bridge.native_dump_weight_line,
                    "requested_lead_deg": LEAD_DEG,
                    "requested_lead_rad": LEAD_RAD,
                    "raw_lead_before_deg": bridge.raw_lead_before,
                    "effective_lead_rad": bridge.cfg_lead_rad,
                    "native_dump_lead_rad": bridge.native_dump_lead_rad,
                    "native_dump_lead_line": bridge.native_dump_lead_line,
                    "native_dump_file": None
                    if bridge.native_dump_path is None
                    else bridge.native_dump_path.name,
                    "native_dump_sha256": None
                    if bridge.native_dump_path is None
                    else sha(bridge.native_dump_path),
                    "reference_duration_s": ref_info["duration_s"],
                    "reference_start_ramp_s": START_RAMP_S if reference_kind == "A_start04" else SYMMETRIC_RAMP_S,
                    "reference_stop_ramp_s": 0.0 if reference_kind == "A_start04" else SYMMETRIC_RAMP_S,
                    "whole_motion_duration_s": float(raw.get("integrated_duration_s", 0.0)) if trace else 0.0,
                }
            )
            raw["window_metrics"] = _window_metrics(raw, trace, ref_info) if trace else {}

            trace_path = HERE / f"{case}_trace.npz"
            if trace:
                raw["trace_sha256"] = _save_trace(trace, trace_path)
                raw["trace_file"] = trace_path.name
                reference_path = HERE / f"{case}_reference.npz"
                raw["reference_sha256"] = _save_reference(
                    ref_info["reference"],
                    np.asarray([row["t_s"] for row in trace], dtype=float),
                    reference_path,
                )
                raw["reference_file"] = reference_path.name
            else:
                raw["trace_file"] = None
                raw["trace_sha256"] = None
                raw["reference_file"] = None
                raw["reference_sha256"] = None
            _write_case_result(case, raw)
            result["runs"].append(raw)
            (HERE / "RESULTS.json").write_text(
                json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n"
            )
            print(
                f"DONE {case} n={raw.get('n_steps', 0)} duration={raw.get('whole_motion_duration_s')} "
                f"pose={raw.get('strict_pose_max_gate')} mechanical={raw.get('strict_mechanical_gate')} "
                f"failures={len(raw.get('failures', []))}",
                flush=True,
            )

    # Compare against the sealed W-only unit-weight runs without rerunning
    # them.  This makes the effect of lead20 on the large arm-jerk violation
    # explicit while keeping the parent artifacts immutable.
    if PARENT_WEIGHT_RESULTS.is_file():
        parent = json.loads(PARENT_WEIGHT_RESULTS.read_text())
        parent_runs = {row.get("case"): row for row in parent.get("runs", [])}
        current_runs = {row.get("case"): row for row in result["runs"]}
        comparison: dict[str, Any] = {}
        for path_index in (0, 1):
            prior = parent_runs.get(f"path{path_index}_tolerance_unit", {})
            prior_trace_file = prior.get("trace_file")
            prior_margin = None
            prior_excess = None
            if prior_trace_file and prior.get("jerk_limit_by_coordinate") is not None:
                prior_trace = np.load(
                    CROSS_TRACK_FIX / "weight_ablation" / prior_trace_file,
                    allow_pickle=False,
                )
                prior_jerk = np.asarray(prior_trace["command_jerk"], dtype=float)
                prior_limits = np.asarray(prior["jerk_limit_by_coordinate"], dtype=float)
                prior_arm = np.abs(prior_jerk[:, 1:])
                prior_margin = float(np.min(prior_limits[1:][None, :] - prior_arm))
                prior_excess = int(np.sum(prior_arm > prior_limits[1:][None, :]))
            a_full = current_runs.get(f"path{path_index}_A_start04", {}).get(
                "window_metrics", {}
            ).get("full", {})
            b_full = current_runs.get(f"path{path_index}_B_symmetric08", {}).get(
                "window_metrics", {}
            ).get("full", {})
            comparison[str(path_index)] = {
                "prior_W_only_full_jerk_margin_arm_rad_s3": prior_margin,
                "prior_W_only_full_jerk_excess_arm_coordinate_ticks": prior_excess,
                "A_start04_full_jerk_margin_arm_rad_s3": a_full.get(
                    "jerk_margin_min_arm_rad_s3"
                ),
                "B_symmetric08_full_jerk_margin_arm_rad_s3": b_full.get(
                    "jerk_margin_min_arm_rad_s3"
                ),
            }
        result["comparison_to_W_only"] = comparison

    (HERE / "RESULTS.json").write_text(
        json.dumps(clean_json(result), indent=2, allow_nan=False) + "\n"
    )
    results_sha = sha(HERE / "RESULTS.json")
    (HERE / "RESULTS.sha256").write_text(results_sha + "  RESULTS.json\n")
    (HERE / "REPORT.txt").write_text(_report(result))
    print(f"RESULTS_SHA256 {results_sha}", flush=True)


if __name__ == "__main__":
    main()
