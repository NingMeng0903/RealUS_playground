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
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

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

    @property
    def qdot_prev(self) -> np.ndarray:
        """Compatibility alias used by older task producers."""

        return self.qdot_applied_prev


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


@dataclass(frozen=True, slots=True, init=False)
class ProtectedTask:
    """Task rows that must remain protected by the hierarchy.

    ``A @ qdot`` is compared with ``b`` by the solver.  ``row_scales`` are
    strictly positive dimensionless weights; ``residual_limits`` are optional
    non-negative per-row limits used by the reference governor and diagnostics.
    """

    A: np.ndarray
    b: np.ndarray
    row_scales: np.ndarray
    residual_limits: np.ndarray | None
    one_sided_constraints: tuple[HardConstraintRow, ...]
    name: str

    def __init__(
        self,
        A: Any,
        b: Any,
        row_scales: Any = None,
        residual_limits: Any = None,
        one_sided_constraints: Sequence[HardConstraintRow] = (),
        name: str = "protected",
    ) -> None:
        coeff = _array(A, name="A", ndim=2)
        m, n = (int(v) for v in coeff.shape)
        target = _vector(b, name="b", length=m)
        scales = (
            np.ones(m, dtype=float)
            if row_scales is None
            else _vector(row_scales, name="row_scales", length=m)
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError("row_scales must be finite and > 0")
        scales.setflags(write=False)
        limits: np.ndarray | None
        if residual_limits is None:
            limits = None
        else:
            raw = np.asarray(residual_limits, dtype=float)
            if raw.ndim == 0:
                raw = np.full(m, float(raw), dtype=float)
            limits = _vector(raw, name="residual_limits", length=m)
            if np.any(limits < 0.0):
                raise ValueError("residual_limits must be >= 0")
        rows: list[HardConstraintRow] = []
        for row in tuple(one_sided_constraints):
            if not isinstance(row, HardConstraintRow):
                raise ValueError("one_sided_constraints must contain HardConstraintRow values")
            if row.dimension != n:
                raise ValueError(
                    f"constraint {row.name!r} has dimension {row.dimension}, expected {n}"
                )
            if row.lower is not None and row.upper is not None:
                raise ValueError("one_sided_constraints rows must be one-sided")
            rows.append(row)
        object.__setattr__(self, "A", coeff)
        object.__setattr__(self, "b", target)
        object.__setattr__(self, "row_scales", scales)
        object.__setattr__(self, "residual_limits", limits)
        object.__setattr__(self, "one_sided_constraints", tuple(rows))
        object.__setattr__(self, "name", _name(name, default="protected"))

    @property
    def n_rows(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.A.shape[1])

    @property
    def constraints(self) -> tuple[HardConstraintRow, ...]:
        return self.one_sided_constraints


@dataclass(frozen=True, slots=True, init=False)
class ScalableTask:
    """A soft task whose rows share a governor scale group."""

    A: np.ndarray
    b: np.ndarray
    scale_group_id: str | int
    row_scales: np.ndarray
    slack_limits: np.ndarray | None
    name: str

    def __init__(
        self,
        A: Any,
        b: Any,
        scale_group_id: str | int,
        row_scales: Any = None,
        slack_limits: Any = None,
        name: str = "motion",
    ) -> None:
        coeff = _array(A, name="A", ndim=2)
        m = int(coeff.shape[0])
        target = _vector(b, name="b", length=m)
        scales = (
            np.ones(m, dtype=float)
            if row_scales is None
            else _vector(row_scales, name="row_scales", length=m)
        )
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError("row_scales must be finite and > 0")
        scales.setflags(write=False)
        if isinstance(scale_group_id, (bool, np.bool_)):
            raise ValueError("scale_group_id must be a non-empty string or integer")
        if isinstance(scale_group_id, str):
            if not scale_group_id.strip():
                raise ValueError("scale_group_id must not be empty")
            group: str | int = scale_group_id
        elif isinstance(scale_group_id, (int, np.integer)):
            if int(scale_group_id) < 0:
                raise ValueError("scale_group_id integer must be >= 0")
            group = int(scale_group_id)
        else:
            raise ValueError("scale_group_id must be a non-empty string or integer")

        limits: np.ndarray | None
        if slack_limits is None:
            limits = None
        else:
            raw = np.asarray(slack_limits, dtype=float)
            if raw.ndim == 0:
                raw = np.full(m, float(raw), dtype=float)
            limits = _array(raw, name="slack_limits")
            if limits.ndim == 1:
                if limits.size != m or np.any(limits < 0.0):
                    raise ValueError("slack_limits vector must be length m and >= 0")
            elif limits.ndim == 2:
                if limits.shape != (m, 2) or np.any(limits[:, 0] > limits[:, 1]):
                    raise ValueError("slack_limits matrix must have shape (m, 2) and lo <= hi")
            else:
                raise ValueError("slack_limits must be a scalar, vector, or (m, 2) matrix")
        object.__setattr__(self, "A", coeff)
        object.__setattr__(self, "b", target)
        object.__setattr__(self, "scale_group_id", group)
        object.__setattr__(self, "row_scales", scales)
        object.__setattr__(self, "slack_limits", limits)
        object.__setattr__(self, "name", _name(name, default="motion"))

    @property
    def group_id(self) -> str | int:
        return self.scale_group_id

    @property
    def n_rows(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.A.shape[1])


@dataclass(frozen=True, slots=True, init=False)
class PostureGuide:
    """Planner-produced posture target consumed as a bounded soft guide."""

    q_goal: np.ndarray
    qdot_guide: np.ndarray
    valid_until: float
    quality: float
    planner_state: Any
    source: str | None
    created_at: float | None
    metadata: Mapping[str, Any]

    def __init__(
        self,
        q_goal: Any,
        qdot_guide: Any,
        valid_until: Any,
        quality: Any,
        planner_state: Any,
        source: str | None = None,
        created_at: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        goal = _vector(q_goal, name="q_goal")
        guide = _vector(qdot_guide, name="qdot_guide", length=goal.size)
        expiry = _scalar(valid_until, name="valid_until")
        score = _scalar(quality, name="quality")
        if not 0.0 <= score <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if source is not None and not isinstance(source, str):
            raise ValueError("source must be a string or None")
        created = None if created_at is None else _scalar(created_at, name="created_at")
        meta = {} if metadata is None else dict(metadata)
        object.__setattr__(self, "q_goal", goal)
        object.__setattr__(self, "qdot_guide", guide)
        object.__setattr__(self, "valid_until", expiry)
        object.__setattr__(self, "quality", score)
        object.__setattr__(self, "planner_state", planner_state)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    @property
    def planner_state_name(self) -> str:
        value = self.planner_state
        return value.value if hasattr(value, "value") else str(value)

    @property
    def n_joints(self) -> int:
        return int(self.q_goal.size)

    def is_valid(self, now: float) -> bool:
        """Return whether this guide can be used at monotonic time ``now``."""

        return _scalar(now, name="now") <= self.valid_until

    def age(self, now: float) -> float:
        """Return ``now - created_at`` when a creation timestamp is known."""

        now_f = _scalar(now, name="now")
        if self.created_at is None:
            return 0.0
        return now_f - self.created_at


@runtime_checkable
class TaskReference(Protocol):
    """Protocol for a producer of one absolute/scalable task reference.

    Implementations may return :class:`ProtectedTask`, :class:`ScalableTask`,
    a sequence of either, or a domain-specific immutable value consumed by the
    solver.  ``t_s`` is a reference-clock time and ``state`` is optional to
    support purely time-parameterised trajectories.
    """

    def sample(self, t_s: float, state: RobotState | None = None) -> object:
        ...


@runtime_checkable
class ReferenceHorizon(Protocol):
    """Protocol for a producer of a finite look-ahead reference horizon."""

    def sample(
        self,
        t_s: float,
        horizon_s: float = 0.0,
        state: RobotState | None = None,
    ) -> Sequence[object]:
        ...


__all__ = [
    "RobotState",
    "HardConstraintRow",
    "OneSidedConstraint",
    "LinearConstraintSet",
    "ProtectedTask",
    "ScalableTask",
    "PostureGuide",
    "TaskReference",
    "ReferenceHorizon",
]
