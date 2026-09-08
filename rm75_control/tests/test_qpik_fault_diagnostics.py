"""Tests for the compact, side-effect-free QPIK fault description."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.fault_diagnostics import (
    format_qpik_fault_details,
    save_qpik_fault_snapshot,
)


def _step(**kwargs) -> SimpleNamespace:
    """Use the JointIkStep field names without importing hardware dependencies."""

    values = {
        "qp1_status": "not_run",
        "qp2_status": "not_run",
        "qpik_total_ms": 0.0,
        "n_cbf_active": 0,
        "box_degenerate": False,
        "box_lo": np.full(8, np.nan),
        "box_hi": np.full(8, np.nan),
        "qdot_prev_used": np.full(8, np.nan),
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_formats_the_j4_degenerate_box() -> None:
    lo = np.full(8, -1.0)
    hi = np.full(8, 1.0)
    lo[4] = hi[4] = -0.6322929399161126
    previous = np.zeros(8)
    previous[4] = -0.003159172409
    details = format_qpik_fault_details(
        _step(
            qp1_status="primal_infeasible",
            qp2_status="not_run",
            qpik_total_ms=2.331835,
            n_cbf_active=4,
            box_degenerate=True,
            box_lo=lo,
            box_hi=hi,
            qdot_prev_used=previous,
        )
    )

    assert "qp1=primal_infeasible" in details
    assert "qp2=not_run" in details
    assert "solve_ms=2.33183" in details
    assert "cbf=4" in details
    assert "degbox=j4(lo=-0.632293,hi=-0.632293,prev=-0.00315917)" in details


def test_missing_or_malformed_fields_are_safe() -> None:
    details = format_qpik_fault_details(
        SimpleNamespace(
            qp1_status="failed",
            box_degenerate=True,
            box_lo="not-an-array",
            box_hi=object(),
        )
    )

    assert details == "qp1=failed qp2=? solve_ms=? cbf=? degbox=?"


def test_default_step_without_a_degenerate_box_is_compact() -> None:
    details = format_qpik_fault_details(_step())

    assert "qp1=not_run" in details
    assert "qp2=not_run" in details
    assert "cbf=0" in details
    assert details.endswith("degbox=none")


@dataclass
class _SnapshotStep:
    qp1_status: str
    qp2_status: str
    qpik_total_ms: float
    n_cbf_active: int
    box_degenerate: bool
    box_lo: np.ndarray
    box_hi: np.ndarray
    qdot_prev_used: np.ndarray
    extra_values: np.ndarray
    unknown_value: object


def _snapshot_step() -> _SnapshotStep:
    lo = np.full(8, -1.0)
    hi = np.full(8, 1.0)
    lo[0] = hi[0] = 0.0
    lo[4] = hi[4] = -0.6322929399161126
    previous = np.zeros(8)
    previous[4] = -0.003159172409
    return _SnapshotStep(
        qp1_status="primal_infeasible",
        qp2_status="not_run",
        qpik_total_ms=2.331835,
        n_cbf_active=4,
        box_degenerate=True,
        box_lo=lo,
        box_hi=hi,
        qdot_prev_used=previous,
        extra_values=np.array([np.nan, np.inf, -np.inf, 1.25]),
        unknown_value=object(),
    )


def test_save_snapshot_serializes_step_position_and_metadata(tmp_path: Path) -> None:
    q_meas = np.array([0.4, 0.0, 0.0, 0.0, 2.95, 0.0, 0.0, 0.0])
    q_command = q_meas.copy()
    q_lower = np.array([0.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    q_upper = np.array([0.8, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    path = save_qpik_fault_snapshot(
        _snapshot_step(),
        reason="publication_infeasible",
        phase="hybrid_contact",
        q_meas=q_meas,
        q_command=q_command,
        q_lower=q_lower,
        q_upper=q_upper,
        position_margin=0.1,
        metadata={
            "urdf": Path("robot.urdf"),
            "tool_offset": np.array([np.nan, 0.002]),
            "dof": 8,
        },
        directory=tmp_path,
    )

    assert path.parent == tmp_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["reason"] == "publication_infeasible"
    assert payload["phase"] == "hybrid_contact"
    assert payload["q_meas"] == pytest.approx(q_meas.tolist())
    assert payload["q_command"] == pytest.approx(q_command.tolist())
    assert payload["step"]["n_cbf_active"] == 4
    assert payload["step"]["qp1_status"] == "primal_infeasible"
    assert payload["step"]["extra_values"] == [None, None, None, 1.25]
    assert isinstance(payload["step"]["unknown_value"], str)
    assert payload["metadata"]["urdf"] == "robot.urdf"
    assert payload["metadata"]["tool_offset"] == [None, 0.002]
    assert payload["position"]["status"] == "ok"
    assert payload["position"]["effective_upper"][4] == pytest.approx(2.9)
    assert payload["position"]["upper_minus_measured"][4] == pytest.approx(-0.05)
    assert payload["position"]["min_arm_position_margin_axis"] == "j4"
    assert payload["position"]["min_arm_position_margin"] == pytest.approx(-0.05)
    rail = next(item for item in payload["box"]["degenerate_axes"] if item["axis"] == "rail")
    assert rail["state"] == "frozen"
    assert "position_conflict" not in rail
    assert "cbf_pair" not in payload["step"]
    json.dumps(payload, allow_nan=False)


def test_save_snapshot_same_time_ns_does_not_overwrite(tmp_path: Path, monkeypatch) -> None:
    import rm75_control.control.joint_admittance_8dof.fault_diagnostics as diagnostics

    monkeypatch.setattr(diagnostics.time, "time_ns", lambda: 123456789)
    kwargs = dict(
        reason="fault",
        phase="scan",
        q_meas=np.zeros(8),
        q_command=np.zeros(8),
        q_lower=np.full(8, -1.0),
        q_upper=np.full(8, 1.0),
        position_margin=np.zeros(8),
        directory=tmp_path,
    )
    first = save_qpik_fault_snapshot(_snapshot_step(), **kwargs)
    second = save_qpik_fault_snapshot(_snapshot_step(), **kwargs)

    assert first != second
    assert first.is_file() and second.is_file()
    assert sorted(path.name for path in tmp_path.glob("*.json")) == sorted(
        (first.name, second.name)
    )


def test_save_snapshot_marks_incomplete_position_data_unknown(tmp_path: Path) -> None:
    path = save_qpik_fault_snapshot(
        _snapshot_step(),
        reason="fault",
        phase="scan",
        q_meas=np.zeros(7),
        q_command=np.zeros(8),
        q_lower=np.zeros(8),
        q_upper=np.ones(8),
        position_margin=np.zeros(8),
        directory=tmp_path,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["position"]["status"] == "unknown"
    assert payload["position"]["min_arm_position_margin"] is None
    assert payload["position"]["min_arm_position_margin_axis"] == "unknown"
