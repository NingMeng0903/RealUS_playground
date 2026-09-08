"""Small, side-effect-free descriptions of QPIK publication faults."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


_DEGENERATE_BOX_EPS = 1.0e-9
_DEFAULT_FAULT_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "apps" / "logs" / "peirastic" / "faults"
)


def _field(step: Any, name: str, default: Any = None) -> Any:
    """Read a step field without allowing diagnostic formatting to fault."""

    try:
        if isinstance(step, Mapping):
            return step.get(name, default)
        return getattr(step, name, default)
    except Exception:
        return default


def _first_field(step: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = _field(step, name, None)
        if value is not None:
            return value
    return default


def _status_text(value: Any) -> str:
    try:
        text = str(value).strip()
    except Exception:
        return "?"
    return text or "?"


def _number_text(value: Any, missing: str = "?") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return missing
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0.0 else "-inf"
    return f"{number:.6g}"


def _count_text(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "?"
    if not math.isfinite(number):
        return "?"
    return str(int(number))


def _array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.empty(0, dtype=float)


def _axis_name(index: int) -> str:
    # The eight-element vector is [rail, j1, ..., j7]; q[4] is therefore J4.
    return "rail" if index == 0 else f"j{index}"


def _degenerate_boxes(step: Any) -> str:
    lo = _array(_field(step, "box_lo"))
    hi = _array(_field(step, "box_hi"))
    previous = _array(
        _first_field(step, ("qdot_prev_used", "qdot_prev"), default=None)
    )

    axes: list[str] = []
    for index in range(min(lo.size, hi.size)):
        lo_i = float(lo[index])
        hi_i = float(hi[index])
        if not (math.isfinite(lo_i) and math.isfinite(hi_i)):
            continue
        if hi_i - lo_i > _DEGENERATE_BOX_EPS:
            continue
        previous_i = (
            _number_text(previous[index]) if index < previous.size else "?"
        )
        axes.append(
            f"{_axis_name(index)}(lo={_number_text(lo_i)},"
            f"hi={_number_text(hi_i)},prev={previous_i})"
        )

    if axes:
        return ";".join(axes)
    try:
        flagged = bool(_field(step, "box_degenerate", False))
    except Exception:
        flagged = False
    return "?" if flagged else "none"


def format_qpik_fault_details(step: Any) -> str:
    """Return one compact diagnostic line for a QPIK fault.

    This function only reads ``JointIkStep`` fields.  Missing or malformed
    fields are represented by ``?``/``nan`` so fault handling can safely use
    it after the stop path without adding another failure mode.
    """

    return (
        f"qp1={_status_text(_field(step, 'qp1_status', '?'))} "
        f"qp2={_status_text(_field(step, 'qp2_status', '?'))} "
        f"solve_ms={_number_text(_first_field(step, ('qpik_total_ms', 'qp_solver_solve_ms'), '?'))} "
        f"cbf={_count_text(_field(step, 'n_cbf_active', '?'))} "
        f"degbox={_degenerate_boxes(step)}"
    )


def _json_safe(value: Any) -> Any:
    """Convert diagnostics to strict-JSON values without letting odd fields fail."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        try:
            return _json_safe(value.tolist())
        except Exception:
            return str(value)
    if isinstance(value, Mapping):
        try:
            return {str(key): _json_safe(item) for key, item in value.items()}
        except Exception as exc:
            return f"<unknown mapping: {exc}>"
    if isinstance(value, (list, tuple, set, frozenset)):
        try:
            return [_json_safe(item) for item in value]
        except Exception:
            return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(_dataclass_fields(value))
    try:
        return str(value)
    except Exception:
        return "<unknown>"


def _dataclass_fields(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for item in dataclasses.fields(value):
            try:
                result[item.name] = getattr(value, item.name)
            except Exception as exc:
                result[item.name] = f"<unreadable: {exc}>"
    except Exception as exc:
        result["_unknown"] = f"<dataclass fields unavailable: {exc}>"
    return result


def _step_fields(step: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(step) and not isinstance(step, type):
        return _dataclass_fields(step)
    if isinstance(step, Mapping):
        try:
            return {str(key): value for key, value in step.items()}
        except Exception as exc:
            return {"_unknown": f"<mapping unavailable: {exc}>"}
    try:
        values = vars(step)
    except Exception as exc:
        return {"_unknown": f"{type(step).__name__}: {exc}"}
    try:
        return dict(values)
    except Exception as exc:
        return {"_unknown": f"<fields unavailable: {exc}>"}


def _numeric_vector(value: Any, size: int = 8) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if vector.size != size:
        return None
    return vector


def _margin_vector(value: Any, size: int = 8) -> np.ndarray | None:
    try:
        margin = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if margin.size == 1:
        return np.full(size, float(margin[0]), dtype=float)
    if margin.size != size:
        return None
    return margin


def _position_diagnostics(
    *,
    q_meas: Any,
    q_lower: Any,
    q_upper: Any,
    position_margin: Any,
) -> dict[str, Any]:
    q = _numeric_vector(q_meas)
    lower = _numeric_vector(q_lower)
    upper = _numeric_vector(q_upper)
    margin = _margin_vector(position_margin)
    if lower is None or upper is None or margin is None:
        return {
            "status": "unknown",
            "q_lower": _json_safe(q_lower),
            "q_upper": _json_safe(q_upper),
            "position_margin": _json_safe(position_margin),
            "effective_lower": None,
            "effective_upper": None,
            "measured_minus_lower": None,
            "upper_minus_measured": None,
            "min_arm_position_margin": None,
            "min_arm_position_margin_axis": "unknown",
        }

    effective_lower = lower + margin
    effective_upper = upper - margin
    measured_minus_lower = (
        None if q is None else q - effective_lower
    )
    upper_minus_measured = None if q is None else effective_upper - q
    arm_clearances: list[tuple[float, str]] = []
    if q is not None:
        for index in range(1, 8):
            if not (
                math.isfinite(float(measured_minus_lower[index]))
                and math.isfinite(float(upper_minus_measured[index]))
            ):
                continue
            arm_clearances.append(
                (
                    min(
                        float(measured_minus_lower[index]),
                        float(upper_minus_measured[index]),
                    ),
                    _axis_name(index),
                )
            )
    if arm_clearances:
        min_margin, min_axis = min(arm_clearances, key=lambda item: item[0])
        min_margin_json: float | None = min_margin
    else:
        min_margin_json = None
        min_axis = "unknown"
    complete = bool(
        q is not None
        and np.all(np.isfinite(q))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
        and np.all(np.isfinite(margin))
    )
    return {
        "status": "ok" if complete else "unknown",
        "q_lower": _json_safe(lower),
        "q_upper": _json_safe(upper),
        "position_margin": _json_safe(margin),
        "effective_lower": _json_safe(effective_lower),
        "effective_upper": _json_safe(effective_upper),
        "measured_minus_lower": _json_safe(measured_minus_lower),
        "upper_minus_measured": _json_safe(upper_minus_measured),
        "min_arm_position_margin": min_margin_json,
        "min_arm_position_margin_axis": min_axis,
    }


def _box_diagnostics(step: Any) -> dict[str, Any]:
    lo_value = _field(step, "box_lo")
    hi_value = _field(step, "box_hi")
    lo = _numeric_vector(lo_value)
    hi = _numeric_vector(hi_value)
    axes: list[dict[str, Any]] = []
    if lo is not None and hi is not None:
        for index in range(8):
            lo_i = float(lo[index])
            hi_i = float(hi[index])
            if not (math.isfinite(lo_i) and math.isfinite(hi_i)):
                continue
            if hi_i - lo_i <= _DEGENERATE_BOX_EPS:
                # Rail [0, 0] is a deliberate lock while the arm solves; it
                # is diagnostic state, not evidence of a position conflict.
                state = "frozen" if index == 0 else "degenerate"
                axes.append(
                    {
                        "axis": _axis_name(index),
                        "index": index,
                        "state": state,
                        "lo": lo_i,
                        "hi": hi_i,
                    }
                )
    try:
        flagged = bool(_field(step, "box_degenerate", False))
    except Exception:
        flagged = False
    return {
        "lo": _json_safe(lo_value),
        "hi": _json_safe(hi_value),
        "degenerate_axes": axes,
        "flagged": flagged,
    }


def _new_fault_path(directory: Path) -> Path:
    stamp = time.time_ns()
    pid = os.getpid()
    base = f"qpik_fault_{stamp}_{pid}"
    path = directory / f"{base}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{base}_{suffix}.json"
        suffix += 1
    return path


def save_qpik_fault_snapshot(
    step: Any,
    *,
    reason: Any,
    phase: Any,
    q_meas: Any,
    q_command: Any,
    q_lower: Any,
    q_upper: Any,
    position_margin: Any,
    metadata: Mapping[str, Any] | None = None,
    directory: str | Path | None = None,
) -> Path:
    """Write one strict-JSON QPIK fault snapshot after the stop path.

    The function is intentionally a local serializer.  It does not inspect
    hardware or shared memory and does not invoke any stop/reset operation.
    """

    out_dir = _DEFAULT_FAULT_DIRECTORY if directory is None else Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_s = time.time()
    wall_ns = time.time_ns()
    mono_s = time.monotonic()
    step_fields = _step_fields(step)
    payload: dict[str, Any] = {
        "wall_time_s": wall_s if math.isfinite(wall_s) else None,
        "wall_time_ns": wall_ns,
        "mono_time_s": mono_s if math.isfinite(mono_s) else None,
        "reason": _json_safe(reason),
        "phase": _json_safe(phase),
        "q_meas": _json_safe(q_meas),
        "q_command": _json_safe(q_command),
        "step": _json_safe(step_fields),
        "position": _position_diagnostics(
            q_meas=q_meas,
            q_lower=q_lower,
            q_upper=q_upper,
            position_margin=position_margin,
        ),
        "box": _box_diagnostics(step),
        "metadata": _json_safe({} if metadata is None else metadata),
    }
    path = _new_fault_path(out_dir)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


__all__ = ["format_qpik_fault_details", "save_qpik_fault_snapshot"]
