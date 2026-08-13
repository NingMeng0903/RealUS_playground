"""Small, immutable task-model types shared by the strict WBC layers.

The controller has a few task producers (Cartesian, joint, planner and
hardware safety).  They should all agree on the shape of their matrices before
anything reaches a QP backend.  This module is deliberately free of solver
imports and contains only value objects and protocols.  Every numpy value is
copied at construction time and marked read-only; a producer cannot mutate a
task while a real-time solve is consuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _array(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: Any = float,
) -> np.ndarray:
    """Copy ``value`` to a finite, read-only ndarray with exact dimensions."""

    try:
        out = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric ndarray") from exc
    if ndim is not None and out.ndim != int(ndim):
        raise ValueError(f"{name} must have ndim={ndim}, got shape {out.shape}")
    if shape is not None and tuple(out.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    out.setflags(write=False)
    return out


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    out = _array(value, name=name, ndim=1)
    if length is not None and out.size != int(length):
        raise ValueError(f"{name} must have length {length}, got {out.size}")
    return out


def _scalar(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite scalar")
    return out


def _bounds_array(value: Any, *, name: str, length: int) -> np.ndarray:
    """Copy linear bounds, allowing +/-inf as the explicit open side.

    Coefficients and task values remain strictly finite.  Solver bound vectors
    conventionally use ``-inf``/``inf`` for an unbounded side, so those two
    values are accepted here while NaN is always rejected.
    """

    try:
        out = np.array(value, dtype=float, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if out.ndim != 1 or out.size != int(length):
        raise ValueError(f"{name} must have shape ({length},), got {out.shape}")
    if np.isnan(out).any():
        raise ValueError(f"{name} must not contain NaN")
    out.setflags(write=False)
    return out


def _optional_bound(value: Any, *, name: str) -> float | None:
    """Validate a hard bound.

    ``None`` is the explicit representation of an open side.  Using ``None``
    instead of +/-inf keeps the value objects strictly finite while still
    allowing a genuinely one-sided row.
    """

    if value is None:
        return None
    return _scalar(value, name=name)


def _name(value: Any, *, default: str) -> str:
    if value is None:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("name must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RobotState:
    """Snapshot of measured/controller state consumed by a task producer.

    Joint count is intentionally not hard-coded to eight: tests and planners
    can use the same model for a reduced arm or a different robot.  ``dt`` is
    the elapsed control period and must be strictly positive.
    """

    q_meas: np.ndarray
    q_cmd: np.ndarray
    qdot_applied_prev: np.ndarray
    dt: float
    contact_active: bool
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        q_meas = _vector(self.q_meas, name="q_meas")
        n = int(q_meas.size)
        q_cmd = _vector(self.q_cmd, name="q_cmd", length=n)
        qdot_prev = _vector(self.qdot_applied_prev, name="qdot_applied_prev", length=n)
        dt = _scalar(self.dt, name="dt")
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        timestamp = _scalar(self.timestamp, name="timestamp")
        if not isinstance(self.contact_active, (bool, np.bool_)):
            raise ValueError("contact_active must be a bool")
        object.__setattr__(self, "q_meas", q_meas)
        object.__setattr__(self, "q_cmd", q_cmd)
        object.__setattr__(self, "qdot_applied_prev", qdot_prev)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "contact_active", bool(self.contact_active))
        object.__setattr__(self, "timestamp", timestamp)

    @property
    def n_joints(self) -> int:
        return int(self.q_meas.size)

@dataclass(frozen=True, slots=True, init=False)
class HardConstraintRow:
    """One linear velocity row ``lower <= a @ x <= upper``.

    Either bound may be ``None`` to represent an open side.  At least one side
    must be present; explicit NaN/Inf values are rejected at the model boundary
    so a malformed row cannot poison a solver's bounds vector.
    """

    a: np.ndarray
    lower: float | None
    upper: float | None
    name: str

    def __init__(
        self,
        a: Any,
        lower: Any = None,
        upper: Any = None,
        name: str = "constraint",
    ) -> None:
        coeff = _vector(a, name="a")
        lo = _optional_bound(lower, name="lower")
        hi = _optional_bound(upper, name="upper")
        if lo is None and hi is None:
            raise ValueError("a hard-constraint row needs lower or upper")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("lower must be <= upper")
        object.__setattr__(self, "a", coeff)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "name", _name(name, default="constraint"))

    @property
    def dimension(self) -> int:
        return int(self.a.size)

    @property
    def is_one_sided(self) -> bool:
        return self.lower is None or self.upper is None


class OneSidedConstraint(HardConstraintRow):
    """Convenience spelling for a one-sided :class:`HardConstraintRow`.

    ``sense`` accepts ``"<="``, ``"le"``, ``">="`` and ``"ge"``.  The
    explicit ``lower=``/``upper=`` form is also accepted for code that builds
    rows generically.
    """

    def __init__(
        self,
        a: Any,
        bound: Any = None,
        sense: str = "<=",
        name: str = "constraint",
        *,
        lower: Any = None,
        upper: Any = None,
    ) -> None:
        if lower is not None or upper is not None:
            if bound is not None:
                raise ValueError("pass bound or lower/upper, not both")
            super().__init__(a, lower=lower, upper=upper, name=name)
            if self.lower is not None and self.upper is not None:
                raise ValueError("OneSidedConstraint cannot have two bounds")
            return
        if bound is None:
            raise ValueError("bound is required for a one-sided constraint")
        token = str(sense).strip().lower()
        if token in {"<=", "le", "lt", "upper"}:
            super().__init__(a, upper=bound, name=name)
        elif token in {">=", "ge", "gt", "lower"}:
            super().__init__(a, lower=bound, name=name)
        else:
            raise ValueError(f"unknown one-sided sense {sense!r}")

    @property
    def bound(self) -> float:
        return float(self.upper if self.upper is not None else self.lower)

    @property
    def sense(self) -> str:
        return "<=" if self.upper is not None else ">="


@dataclass(frozen=True, slots=True, init=False)
class LinearConstraintSet:
    """Immutable matrix form of a collection of hard linear rows."""

    C: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...]

    def __init__(
        self,
        C: Any,
        lower: Any,
        upper: Any,
        names: Sequence[str] | None = None,
    ) -> None:
        coeff = _array(C, name="C", ndim=2)
        m = int(coeff.shape[0])
        lo = _bounds_array(lower, name="lower", length=m)
        hi = _bounds_array(upper, name="upper", length=m)
        if lo.size != m or hi.size != m:
            raise ValueError(
                f"lower/upper must have length {m}, got {lo.size}/{hi.size}"
            )
        if np.any(lo > hi):
            raise ValueError("lower must be <= upper elementwise")
        if names is None:
            labels = tuple("" for _ in range(m))
        else:
            labels = tuple(names)
            if len(labels) != m:
                raise ValueError(f"names must have length {m}")
            if any(not isinstance(label, str) for label in labels):
                raise ValueError("names must contain strings")
        object.__setattr__(self, "C", coeff)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(self, "names", labels)

    @property
    def n_rows(self) -> int:
        return int(self.C.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.C.shape[1])

    def row(self, index: int) -> HardConstraintRow:
        i = int(index)
        if not 0 <= i < self.n_rows:
            raise IndexError(index)
        name = self.names[i] or "constraint"
        # Linear solver bounds use +/-inf for an open side; convert to the
        # value-model's explicit None representation before constructing the
        # strict finite HardConstraintRow.
        lo = None if np.isneginf(self.lower[i]) else self.lower[i]
        hi = None if np.isposinf(self.upper[i]) else self.upper[i]
        return HardConstraintRow(self.C[i], lo, hi, name)


__all__ = [
    "RobotState",
    "HardConstraintRow",
    "OneSidedConstraint",
    "LinearConstraintSet",
]
