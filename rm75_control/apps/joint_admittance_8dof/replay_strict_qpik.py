#!/usr/bin/env python3
"""Replay a velocity-level QPIK log without touching robot hardware.

This utility feeds the measured joint state and the logged ``v_cmd`` back
through the production :class:`JointIkController` one source row at a time.
Each tick restores the previous two rows' logged command/velocity history
(``logged_single_tick_snapshot``); a newly solved qdot is never integrated into
the next replay tick.  The rail execution input is taken from ``qdot_meas_0``
whenever that field is finite; this is important for evaluating the strict TCP
allocation while the real rail is lagging a command.  Missing history is
reported explicitly in per-row and summary fallback counters.

The replay is deliberately *kinematic / velocity-level only*.  It does not
reconstruct CANFD, Modbus, FA24, actuator tracking, timing contention, or
physical contact.  ``--disable-cbf`` is an offline counterfactual and must
not be interpreted as a safe hardware configuration.

Examples (from ``rm75_control``)::

    source env.sh
    python apps/joint_admittance_8dof/replay_strict_qpik.py \
        apps/logs/gamepad_vcmd/run_20260816_232307.csv \
        --config configs/joint_admittance_8dof.yaml \
        --output-csv /tmp/replay_232307.csv

    python apps/joint_admittance_8dof/replay_strict_qpik.py \
        apps/logs/gamepad_vcmd/run_20260816_232136.csv \
        --disable-cbf --stride 4 --max-rows 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


AXES = ("vx", "vy", "vz", "wx", "wy", "wz")
Q_FIELDS = tuple(f"q_meas_{i}" for i in range(8))
Q_CMD_FIELDS = tuple(f"q_cmd_{i}" for i in range(8))
QDOT_MEAS_FIELDS = tuple(f"qdot_meas_{i}" for i in range(8))
TWIST_FIELDS = tuple(f"v_cmd_{axis}" for axis in AXES)
QDOT_HISTORY_FIELD = "qpik_final_sent_qdot_json"

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "joint_admittance_8dof.yaml"
)


def _finite_float(value: Any) -> float | None:
    """Parse a CSV cell as a finite float, treating blanks as unavailable."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _required_float(row: Mapping[str, Any], field: str, row_number: int) -> float:
    value = _finite_float(row.get(field))
    if value is None:
        raise ValueError(f"row {row_number}: missing/non-finite {field!r}")
    return value


def _required_vector(
    row: Mapping[str, Any], fields: Iterable[str], row_number: int
) -> np.ndarray:
    return np.asarray(
        [_required_float(row, field, row_number) for field in fields], dtype=float
    )


def _optional_vector(row: Mapping[str, Any] | None, fields: Iterable[str]) -> np.ndarray | None:
    if row is None:
        return None
    values = [_finite_float(row.get(field)) for field in fields]
    if any(value is None for value in values):
        return None
    return np.asarray(values, dtype=float)


def _optional_json_vector(
    row: Mapping[str, Any] | None, field: str, size: int
) -> np.ndarray | None:
    if row is None:
        return None
    text = row.get(field)
    if text is None or not str(text).strip():
        return None
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != size or not np.isfinite(vector).all():
        return None
    return vector.copy()


def _fallback(
    counters: Counter[str], labels: list[str], name: str
) -> None:
    counters[name] += 1
    labels.append(name)


def _resolve_logged_snapshot(
    *,
    current_q_meas: np.ndarray,
    previous_row: Mapping[str, Any] | None,
    previous2_row: Mapping[str, Any] | None,
    fallback_counters: Counter[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Resolve the exact logged state needed for one isolated QPIK tick.

    The source rows are deliberately kept separate from the controller's
    output.  A missing historical command is made explicit in the returned
    labels and falls back to the corresponding previous measured state (or
    the current measured state when even that is unavailable).  Velocity
    history first prefers the logged sent-qdot JSON, then measured qdot, then
    a flat zero/previous-velocity history.
    """

    labels: list[str] = []

    q_prev = _optional_vector(previous_row, Q_CMD_FIELDS)
    if q_prev is None:
        q_prev = _optional_vector(previous_row, Q_FIELDS)
        if q_prev is not None:
            _fallback(fallback_counters, labels, "q_cmd_prev_from_previous_q_meas")
        else:
            q_prev = current_q_meas.copy()
            _fallback(fallback_counters, labels, "q_cmd_prev_from_current_q_meas")

    qdot_prev = _optional_json_vector(previous_row, QDOT_HISTORY_FIELD, 8)
    if qdot_prev is None:
        qdot_prev = _optional_vector(previous_row, QDOT_MEAS_FIELDS)
        if qdot_prev is not None:
            _fallback(fallback_counters, labels, "qdot_prev_from_previous_qdot_meas")
        else:
            qdot_prev = np.zeros(8, dtype=float)
            _fallback(fallback_counters, labels, "qdot_prev_zero")

    qdot_prev2 = _optional_json_vector(previous2_row, QDOT_HISTORY_FIELD, 8)
    if qdot_prev2 is None:
        qdot_prev2 = _optional_vector(previous2_row, QDOT_MEAS_FIELDS)
        if qdot_prev2 is not None:
            _fallback(fallback_counters, labels, "qdot_prev2_from_previous2_qdot_meas")
        else:
            # A flat initial history avoids inventing a jerk event at the
            # beginning of a short fixture while still being explicit.
            qdot_prev2 = qdot_prev.copy()
            _fallback(fallback_counters, labels, "qdot_prev2_flat_qdot_prev")
    return q_prev, qdot_prev, qdot_prev2, labels


def _restore_logged_snapshot(
    controller: JointIkController,
    *,
    q_prev: np.ndarray,
    qdot_prev: np.ndarray,
    qdot_prev2: np.ndarray,
    control_dt: float,
) -> None:
    """Restore mutable per-tick state before calling ``update``.

    ``JointIkController.update`` normally integrates its newly solved qdot
    into ``q_cmd`` and leaves the QP/Safety histories warm.  That is correct
    online, but it forks a replay when the next source row contains a logged
    measured state from a different execution.  The replay therefore restores
    all histories that enter q position, acceleration, jerk, and safety boxes
    immediately before each isolated solve.
    """

    controller.q_cmd = np.asarray(q_prev, dtype=float).copy()
    core = controller.core
    core.qdot_prev = np.asarray(qdot_prev, dtype=float).copy()
    core.qdot_prev2 = np.asarray(qdot_prev2, dtype=float).copy()
    # QpIkController.step shifts _qdot_prev_seen into qdot_prev2 at entry.
    # Seed it with the source row i-2 value so that shift reproduces the log.
    core._qdot_prev_seen = np.asarray(qdot_prev2, dtype=float).copy()
    # SafetyLimiter stores position delta, not velocity.  Set the raw logged
    # history directly; sync_applied_delta would additionally clip it.
    controller.safety._dq_prev = np.asarray(qdot_prev, dtype=float) * float(control_dt)


def _json_value(value: Any) -> str:
    """Encode telemetry arrays/IDs in a stable one-cell CSV representation."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    if isinstance(value, (list, tuple)):
        clean: list[Any] = []
        for item in value:
            if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
                clean.append(None)
            elif isinstance(item, np.generic):
                clean.append(item.item())
            else:
                clean.append(item)
        value = clean
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        value = None
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _timestamp_dt(
    row: Mapping[str, Any], previous_t: float | None, nominal_dt: float
) -> tuple[float, float | None]:
    """Return a positive replay dt and the current wall timestamp if present."""

    current_t = _finite_float(row.get("t_wall_s"))
    dt = _finite_float(row.get("dt_actual_s"))
    if dt is None or dt <= 0.0:
        if current_t is not None and previous_t is not None:
            delta = current_t - previous_t
            if math.isfinite(delta) and delta > 0.0:
                dt = delta
    if dt is None or dt <= 0.0:
        dt = float(nominal_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("replay dt must be finite and positive")
    return float(dt), current_t


def _set_cbf_enabled(cfg: Any, enabled: bool) -> None:
    """Disable both config references before constructing the QPIK core."""

    # ``build_joint_ik_config`` normally shares one CollisionConfig instance,
    # but assigning both keeps this tool compatible with older snapshots where
    # QpConfig held a separate copy.
    cfg.collision.enabled = bool(enabled)
    cfg.qp.collision.enabled = bool(enabled)


def _status(value: Any, default: str = "not_run") -> str:
    text = str(value if value is not None else default)
    return text or default


def _controller_row(
    *,
    source_row: int,
    row: Mapping[str, Any],
    controller: JointIkController,
    q_meas: np.ndarray,
    q_prev: np.ndarray,
    qdot_prev: np.ndarray,
    qdot_prev2: np.ndarray,
    history_fallbacks: list[str],
    rail_meas: float,
    rail_source: str,
    control_dt: float,
    box_dt: float,
    box_dt_source: str,
    timestamp: float | None,
    box_dt_holder: dict[str, float],
    reset_controller: bool,
) -> tuple[dict[str, Any], float | None]:
    twist = np.asarray(
        [_required_float(row, field, source_row) for field in TWIST_FIELDS], dtype=float
    )

    path_twist = np.asarray(
        [
            _finite_float(row.get(f"path_twist_{axis}")) or 0.0
            for axis in AXES
        ],
        dtype=float,
    )
    feedback_twist = np.asarray(
        [
            _finite_float(row.get(f"feedback_twist_{axis}")) or 0.0
            for axis in AXES
        ],
        dtype=float,
    )

    # Initialize task objects once, then restore the source snapshot immediately
    # before every solve.  Do not use the new output as the next q_prev.
    if reset_controller:
        controller.reset(q_meas)
    _restore_logged_snapshot(
        controller,
        q_prev=q_prev,
        qdot_prev=qdot_prev,
        qdot_prev2=qdot_prev2,
        control_dt=control_dt,
    )
    box_dt_holder["value"] = float(box_dt)

    wall_start = time.perf_counter_ns()
    step = controller.update(
        twist,
        control_dt,
        q_meas=q_meas,
        path_twist=path_twist,
        feedback_twist=feedback_twist,
        rail_exec_vel_m_s=float(rail_meas),
    )
    wall_ms = (time.perf_counter_ns() - wall_start) / 1.0e6

    core = controller.core
    measured_rail_contrib = np.asarray(
        getattr(core, "last_rail_exec_contrib", np.zeros(6)), dtype=float
    ).reshape(6)
    arm_contrib = np.asarray(
        getattr(core, "last_arm_contrib", np.zeros(6)), dtype=float
    ).reshape(6)
    residual = np.asarray(step.protected_residual, dtype=float).reshape(6)
    if not np.isfinite(residual).all():
        # Keep the report usable with an older controller snapshot that did
        # not publish protected_residual, while never inventing a zero slack.
        residual = np.asarray(
            getattr(core, "last_task_residual", np.full(6, np.nan)), dtype=float
        ).reshape(6)
    residual_inf = float(np.max(np.abs(residual))) if np.isfinite(residual).all() else float("nan")
    qp_total = _finite_float(getattr(step, "qpik_total_ms", None))
    if qp_total is None or qp_total <= 0.0:
        qp_total = float(wall_ms)
        timing_source = "wall_fallback"
    else:
        timing_source = "qpik_telemetry"
    hard_ids = tuple(getattr(step, "hard_active_constraint_ids", ()) or ())
    qp1_status = _status(getattr(step, "qp1_status", "not_run"))
    qp2_status = _status(getattr(step, "qp2_status", "not_run"))
    qp2_fallback = bool(getattr(step, "qp2_fallback", False))
    fallback_level = _status(getattr(step, "fallback_level", "none"), "none")
    fallback_reason = str(getattr(step, "fallback_reason", "") or "")
    row_out: dict[str, Any] = {
        "source_row": int(source_row),
        "t_wall_s": "" if timestamp is None else float(timestamp),
        "dt_s": float(control_dt),
        "box_dt_s": float(box_dt),
        "box_dt_source": str(box_dt_source),
        "history_mode": "logged_single_tick_snapshot",
        "history_fallbacks_json": _json_value(history_fallbacks),
        "snapshot_q_prev_json": _json_value(q_prev),
        "snapshot_qdot_prev_json": _json_value(qdot_prev),
        "snapshot_qdot_prev2_json": _json_value(qdot_prev2),
        "rail_measurement_source": rail_source,
        "rail_measured_velocity_m_s": float(rail_meas),
        "rail_command_m_s": float(np.asarray(step.qdot, dtype=float)[0]),
        "rail_macro_pref_m_s": float(getattr(step, "rail_macro_pref_v", 0.0)),
        "rail_v_reach_m_s": float(getattr(step, "v_reach", 0.0)),
        "rail_v_escape_m_s": float(getattr(step, "v_escape", 0.0)),
        "rail_v_ff_m_s": float(getattr(step, "v_ff_rail", 0.0)),
        "rail_measured_contribution_json": _json_value(measured_rail_contrib),
        "arm_contribution_json": _json_value(arm_contrib),
        "tcp_achieved_json": _json_value(measured_rail_contrib + arm_contrib),
        "tcp_residual_json": _json_value(residual),
        "tcp_residual_inf_m_s": residual_inf,
        "qp1_status": qp1_status,
        "qp2_status": qp2_status,
        "qp2_fallback": int(qp2_fallback),
        "fallback": int(qp2_fallback or fallback_level not in {"", "none"}),
        "fallback_level": fallback_level,
        "fallback_reason": fallback_reason,
        "hard_active_constraint_ids_json": _json_value(hard_ids),
        "qp1_hard_violation": float(
            getattr(core, "last_qp1_hard_violation", float("nan"))
        ),
        "final_hard_violation": float(
            getattr(core, "last_final_hard_violation", float("nan"))
        ),
        "qp1_solve_ms": float(getattr(step, "qp1_solve_ms", 0.0)),
        "qp2_solve_ms": float(getattr(step, "qp2_solve_ms", 0.0)),
        "qpik_assembly_ms": float(getattr(step, "qp_assembly_ms", 0.0)),
        "qpik_total_ms": float(qp_total),
        "wall_update_ms": float(wall_ms),
        "timing_source": timing_source,
        "timing_over_5ms": int(float(qp_total) > 5.0),
        "qdot_command_json": _json_value(np.asarray(step.qdot, dtype=float)),
        "q_cmd_rail_m": float(np.asarray(controller.q_cmd, dtype=float)[0]),
        "qp1_residual_json": _json_value(
            np.asarray(getattr(core, "last_qp1_residual", residual), dtype=float).reshape(6)
        ),
    }
    for index, value in enumerate(q_meas):
        row_out[f"q_meas_{index}"] = float(value)
    for axis, value in zip(AXES, twist):
        row_out[f"v_cmd_{axis}"] = float(value)
    return row_out, float(rail_meas)


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if finite.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def summarize_replay_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a JSON-serializable aggregate from replay output rows."""

    rows_list = list(rows)
    timing = [float(row["qpik_total_ms"]) for row in rows_list]
    residual = [float(row["tcp_residual_inf_m_s"]) for row in rows_list]
    status1 = Counter(str(row["qp1_status"]) for row in rows_list)
    status2 = Counter(str(row["qp2_status"]) for row in rows_list)
    fallback_count = sum(bool(int(row["fallback"])) for row in rows_list)
    q2_fallback_count = sum(bool(int(row["qp2_fallback"])) for row in rows_list)
    over_5ms = sum(bool(int(row["timing_over_5ms"])) for row in rows_list)
    history_modes = Counter(str(row.get("history_mode", "unknown")) for row in rows_list)
    history_fallbacks: Counter[str] = Counter()
    for row in rows_list:
        try:
            history_fallbacks.update(json.loads(str(row.get("history_fallbacks_json", "[]"))))
        except (TypeError, ValueError, json.JSONDecodeError):
            history_fallbacks["history_fallbacks_json_invalid"] += 1
    box_dt = [float(row["box_dt_s"]) for row in rows_list]
    result = {
        "mode": "velocity_level_log_replay",
        "hardware_execution_reconstructed": False,
        "history_mode": dict(sorted(history_modes.items())),
        "rows": len(rows_list),
        "tcp_residual_inf_m_s": _percentiles(residual),
        "qpik_total_ms": _percentiles(timing),
        "box_dt_s": _percentiles(box_dt),
        "over_5ms_count": int(over_5ms),
        "over_5ms_fraction": float(over_5ms / len(rows_list)) if rows_list else 0.0,
        "qp1_status_counts": dict(sorted(status1.items())),
        "qp2_status_counts": dict(sorted(status2.items())),
        "fallback_count": int(fallback_count),
        "qp2_fallback_count": int(q2_fallback_count),
        "rail_command_m_s": _percentiles(
            float(row["rail_command_m_s"]) for row in rows_list
        ),
        "rail_measurement_sources": dict(
            sorted(Counter(str(row["rail_measurement_source"]) for row in rows_list).items())
        ),
        "history_fallback_counts": dict(sorted(history_fallbacks.items())),
    }
    return result


def _output_fields() -> list[str]:
    return [
        "source_row",
        "t_wall_s",
        "dt_s",
        "box_dt_s",
        "box_dt_source",
        *Q_FIELDS,
        *TWIST_FIELDS,
        "history_mode",
        "history_fallbacks_json",
        "snapshot_q_prev_json",
        "snapshot_qdot_prev_json",
        "snapshot_qdot_prev2_json",
        "rail_measurement_source",
        "rail_measured_velocity_m_s",
        "rail_command_m_s",
        "rail_macro_pref_m_s",
        "rail_v_reach_m_s",
        "rail_v_escape_m_s",
        "rail_v_ff_m_s",
        "q_cmd_rail_m",
        "rail_measured_contribution_json",
        "arm_contribution_json",
        "tcp_achieved_json",
        "tcp_residual_json",
        "qp1_residual_json",
        "tcp_residual_inf_m_s",
        "qp1_status",
        "qp2_status",
        "qp2_fallback",
        "fallback",
        "fallback_level",
        "fallback_reason",
        "hard_active_constraint_ids_json",
        "qp1_hard_violation",
        "final_hard_violation",
        "qp1_solve_ms",
        "qp2_solve_ms",
        "qpik_assembly_ms",
        "qpik_total_ms",
        "wall_update_ms",
        "timing_source",
        "timing_over_5ms",
        "qdot_command_json",
    ]


def write_replay_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write replay rows as a compact diagnostic CSV."""

    fields = _output_fields()
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def replay_csv(
    input_csv: str | Path,
    config_path: str | Path = _DEFAULT_CONFIG,
    *,
    stride: int = 1,
    max_rows: int | None = None,
    disable_cbf: bool = False,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Replay selected CSV rows and return ``{"rows", "summary"}``.

    ``stride`` and ``max_rows`` select source rows before invoking the
    controller.  Task objects are initialized once, while q command,
    acceleration/jerk, and SafetyLimiter histories are restored from adjacent
    logged rows before every selected solve.
    """

    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    if max_rows is not None and int(max_rows) < 1:
        raise ValueError("max_rows must be >= 1 when supplied")
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = build_joint_ik_config(raw)
    if disable_cbf:
        _set_cbf_enabled(cfg, False)
    controller = JointIkController(RobotKinematics(), cfg)
    # ``update`` asks the controller for box_dt internally.  Replace only
    # this instance method so the current source row's measured period is
    # used while the control integration period remains cfg.dt.
    box_dt_holder = {"value": float(cfg.dt)}

    def _snapshot_box_dt(_control_dt: float) -> float:
        return float(box_dt_holder["value"])

    controller._measure_box_dt = _snapshot_box_dt  # type: ignore[method-assign]
    rows_out: list[dict[str, Any]] = []
    previous_rail_velocity: float | None = None
    selected = 0
    previous_t: float | None = None
    previous_row: Mapping[str, Any] | None = None
    previous2_row: Mapping[str, Any] | None = None
    history_fallback_counts: Counter[str] = Counter()

    with Path(input_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing_q = sorted(set(Q_FIELDS) - set(reader.fieldnames))
        missing_twist = sorted(set(TWIST_FIELDS) - set(reader.fieldnames))
        if missing_q or missing_twist:
            raise ValueError(
                "input CSV is missing required fields: "
                + ", ".join(missing_q + missing_twist)
            )
        for source_index, row in enumerate(reader):
            # Keep the immediately preceding *source* rows even when stride
            # skips them.  The snapshot for row i must use log rows i-1/i-2,
            # never the previous selected row.
            current_t = _finite_float(row.get("t_wall_s"))
            dt_logged = _finite_float(row.get("dt_actual_s"))
            box_dt_source = "dt_actual_s"
            if dt_logged is not None and dt_logged > 0.0:
                box_dt = float(dt_logged)
            elif current_t is not None and previous_t is not None:
                delta = current_t - previous_t
                if math.isfinite(delta) and delta > 0.0:
                    box_dt = float(delta)
                    box_dt_source = "t_wall_delta"
                    history_fallback_counts["box_dt_from_t_wall_delta"] += 1
                else:
                    box_dt = float(cfg.dt)
                    box_dt_source = "cfg.dt"
                    history_fallback_counts["box_dt_cfg_dt_invalid_timestamp"] += 1
            else:
                box_dt = float(cfg.dt)
                box_dt_source = "cfg.dt"
                history_fallback_counts["box_dt_cfg_dt_missing"] += 1
            if dt_logged is not None and dt_logged <= 0.0:
                history_fallback_counts["box_dt_invalid_dt_actual_s"] += 1

            selected_now = source_index % int(stride) == 0
            if selected_now:
                if max_rows is not None and selected >= int(max_rows):
                    break
                q_meas = _required_vector(row, Q_FIELDS, source_index)
                q_prev, qdot_prev, qdot_prev2, fallback_labels = _resolve_logged_snapshot(
                    current_q_meas=q_meas,
                    previous_row=previous_row,
                    previous2_row=previous2_row,
                    fallback_counters=history_fallback_counts,
                )
                if box_dt_source == "t_wall_delta":
                    fallback_labels.append("box_dt_from_t_wall_delta")
                elif box_dt_source == "cfg.dt":
                    fallback_labels.append("box_dt_cfg_dt")
                rail_meas = _finite_float(row.get("qdot_meas_0"))
                if rail_meas is not None:
                    rail_source = "qdot_meas_0"
                elif previous_rail_velocity is not None:
                    rail_meas = float(previous_rail_velocity)
                    rail_source = "previous_qdot_meas_0"
                    _fallback(
                        history_fallback_counts,
                        fallback_labels,
                        "rail_qdot_meas_previous_source_row",
                    )
                else:
                    rail_meas = 0.0
                    rail_source = "initial_zero"
                    _fallback(
                        history_fallback_counts,
                        fallback_labels,
                        "rail_qdot_meas_initial_zero",
                    )
                out, previous_rail_velocity = _controller_row(
                    source_row=source_index,
                    row=row,
                    controller=controller,
                    q_meas=q_meas,
                    q_prev=q_prev,
                    qdot_prev=qdot_prev,
                    qdot_prev2=qdot_prev2,
                    history_fallbacks=fallback_labels,
                    rail_meas=float(rail_meas),
                    rail_source=rail_source,
                    control_dt=float(cfg.dt),
                    box_dt=float(box_dt),
                    box_dt_source=box_dt_source,
                    timestamp=current_t,
                    box_dt_holder=box_dt_holder,
                    reset_controller=(selected == 0),
                )
                rows_out.append(out)
                selected += 1

            # ``qdot_meas_0`` fallback is based on the adjacent source row,
            # including unselected rows, so stride cannot change its meaning.
            raw_rail = _finite_float(row.get("qdot_meas_0"))
            if raw_rail is not None:
                previous_rail_velocity = float(raw_rail)
            previous2_row = previous_row
            previous_row = row
            previous_t = current_t

    if not rows_out:
        raise ValueError("input CSV yielded no selected rows")
    summary = summarize_replay_rows(rows_out)
    summary["input_csv"] = str(Path(input_csv))
    summary["config"] = str(config_path)
    summary["stride"] = int(stride)
    summary["max_rows"] = None if max_rows is None else int(max_rows)
    summary["cbf_enabled"] = not bool(disable_cbf)
    summary["history_mode"] = {"logged_single_tick_snapshot": len(rows_out)}
    # Include parser/input fallbacks that are not represented by a selected
    # row (for example a missing dt on an unselected source row).
    summary["history_fallback_counts"] = dict(sorted(history_fallback_counts.items()))
    if output_csv is not None:
        write_replay_csv(output_csv, rows_out)
        summary["output_csv"] = str(Path(output_csv))
    return {"rows": rows_out, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csv", type=Path, help="gamepad/WBC CSV to replay")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--disable-cbf",
        action="store_true",
        help="offline counterfactual: disable collision CBF rows",
    )
    args = parser.parse_args(argv)
    try:
        result = replay_csv(
            args.input_csv,
            args.config,
            stride=args.stride,
            max_rows=args.max_rows,
            disable_cbf=args.disable_cbf,
            output_csv=args.output_csv,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
