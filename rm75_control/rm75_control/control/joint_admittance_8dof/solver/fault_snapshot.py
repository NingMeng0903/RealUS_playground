"""First-writer-wins in-memory QP fault snapshot.  Export only after stop."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np


def _arr(value) -> list:
    a = np.asarray(value, dtype=float).reshape(-1)
    return [float(x) if np.isfinite(x) else None for x in a]


@dataclass
class FirstFaultSnapshot:
    taken: bool = False
    epoch: int = 0
    reason: str = ""
    qp1_status: str = ""
    qp2_status: str = ""
    qp1_iter: int = 0
    qp2_iter: int = 0
    qp1_primal: float = float("nan")
    qp1_dual: float = float("nan")
    q_meas: list = field(default_factory=list)
    qdot_committed: list = field(default_factory=list)
    qdot_committed2: list = field(default_factory=list)
    qdot_candidate: list = field(default_factory=list)
    rail_meas_m_s: float = float("nan")
    rail_cmd_m_s: float = float("nan")
    v_cmd: list = field(default_factory=list)
    residual: list = field(default_factory=list)
    box_lo: list = field(default_factory=list)
    box_hi: list = field(default_factory=list)
    box_source: list = field(default_factory=list)
    current_eq_residual: list = field(default_factory=list)
    H_diag: list = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def capture(self, **fields: Any) -> bool:
        if self.taken:
            return False
        for key, value in fields.items():
            if not hasattr(self, key) and key != "extra":
                self.extra[key] = value
                continue
            if key in (
                "q_meas",
                "qdot_committed",
                "qdot_committed2",
                "qdot_candidate",
                "v_cmd",
                "residual",
                "box_lo",
                "box_hi",
                "current_eq_residual",
                "H_diag",
            ):
                setattr(self, key, _arr(value))
            else:
                setattr(self, key, value)
        self.taken = True
        self.epoch = int(self.epoch) + 1
        return True

    def to_json(self) -> dict[str, Any]:
        return {
            "taken": self.taken,
            "epoch": int(self.epoch),
            "reason": self.reason,
            "qp1_status": self.qp1_status,
            "qp2_status": self.qp2_status,
            "qp1_iter": int(self.qp1_iter),
            "qp2_iter": int(self.qp2_iter),
            "qp1_primal": float(self.qp1_primal) if self.qp1_primal == self.qp1_primal else None,
            "qp1_dual": float(self.qp1_dual) if self.qp1_dual == self.qp1_dual else None,
            "q_meas": self.q_meas,
            "qdot_committed": self.qdot_committed,
            "qdot_committed2": self.qdot_committed2,
            "qdot_candidate": self.qdot_candidate,
            "rail_meas_m_s": self.rail_meas_m_s,
            "rail_cmd_m_s": self.rail_cmd_m_s,
            "v_cmd": self.v_cmd,
            "residual": self.residual,
            "box_lo": self.box_lo,
            "box_hi": self.box_hi,
            "box_source": list(self.box_source),
            "current_eq_residual": self.current_eq_residual,
            "H_diag": self.H_diag,
            "extra": dict(self.extra),
        }

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")


SNAPSHOT = FirstFaultSnapshot()


def ablate_residual_qp1(snap: dict[str, Any], *, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    """Offline note: residual QP1 has no preview/hold.  Report which groups remain."""

    groups = ("preview", "hold", "cbf", "inset", "j4_design")
    active = [g for g in groups if g not in drop]
    return {
        "model": "residual_qp1",
        "dropped": list(drop),
        "remaining_optional": active,
        "note": (
            "Residual QP1 has no preview/hold witnesses. "
            "Re-solving a directional snapshot without those rows is the "
            "counterfactual; do not use inbox_brake qdot as the QP solution."
        ),
        "v_cmd": snap.get("v_cmd"),
        "rail_meas_m_s": snap.get("rail_meas_m_s"),
        "qp1_status": snap.get("qp1_status"),
    }
