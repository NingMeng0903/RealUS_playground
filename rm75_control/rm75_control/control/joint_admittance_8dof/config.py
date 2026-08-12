"""YAML loader for the generic measured-state two-level QPIK controller."""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    CartesianTaskProfile,
    ScalableRowGroup,
    selection_from_indices,
)
from rm75_control.control.joint_admittance_8dof.health_monitor import HealthThresholds


def _mapping(value, *, name: str) -> dict:
    """Return a mapping config section, rejecting ambiguous YAML values."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_array(value, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Convert a numeric YAML vector/matrix and reject non-finite entries."""

    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _selection_from_config(value, *, name: str, default_indices=()) -> np.ndarray:
    """Parse a generic 6D task selection matrix or row-index list.

    A mapping may use ``selection``/``matrix`` for an explicit matrix or
    ``rows``/``indices`` for Cartesian row indices.  For convenience, a
    one-dimensional integer sequence is interpreted as row indices, while a
    one-dimensional six-element non-integer sequence is one coefficient row.
    This keeps application profiles declarative without assigning any axis
    semantic to the control core.
    """

    if value is None:
        return selection_from_indices(tuple(default_indices), width=6)
    if isinstance(value, dict):
        if "selection" in value:
            value = value["selection"]
        elif "matrix" in value:
            value = value["matrix"]
        elif "rows" in value:
            return _selection_from_config(value["rows"], name=f"{name}.rows")
        elif "indices" in value:
            return _selection_from_config(value["indices"], name=f"{name}.indices")
        else:
            raise ValueError(
                f"{name} must contain selection/matrix or rows/indices"
            )

    try:
        raw = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a row-index sequence or matrix") from exc
    if raw.ndim == 1:
        if raw.size == 0:
            return np.zeros((0, 6), dtype=float)
        # Explicit row-index lists are the common compact form.  Require
        # integral values and bounds rather than silently truncating floats.
        try:
            as_float = raw.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if np.isfinite(as_float).all() and np.allclose(as_float, np.round(as_float)):
            rounded = np.round(as_float).astype(int)
            if np.all((0 <= rounded) & (rounded < 6)):
                return selection_from_indices(tuple(rounded.tolist()), width=6)
        if raw.size == 6:
            matrix = _finite_array(value, name=name, ndim=1).reshape(1, 6)
            return matrix
        raise ValueError(
            f"{name} 1D values must be integer row indices or a 6-vector"
        )
    matrix = _finite_array(value, name=name, ndim=2)
    if matrix.shape[1] != 6:
        raise ValueError(f"{name} matrix must have shape (rows, 6), got {matrix.shape}")
    return matrix


def _optional_vector(value, *, name: str):
    """Preserve omitted values as ``None`` while validating supplied arrays."""

    if value is None:
        return None
    out = _finite_array(value, name=name)
    return out.copy()


def _parse_cartesian_profile(raw: dict) -> CartesianTaskProfile:
    """Build the application-declared protected/scalable Cartesian rows."""

    protected = _mapping(raw.get("protected_task"), name="qpik.protected_task")
    scalable_raw = raw.get("scalable_tasks", ())
    if scalable_raw is None:
        scalable_raw = ()
    if not isinstance(scalable_raw, (list, tuple)):
        raise ValueError("qpik.scalable_tasks must be a sequence")

    protected_spec = protected.get(
        "selection",
        protected.get("matrix", protected.get("rows", protected.get("indices"))),
    )
    # A missing protected block is deliberately all-protected for compatibility
    # with existing callers that have not migrated their profile yet.
    protected_selection = _selection_from_config(
        protected_spec,
        name="qpik.protected_task.selection",
        default_indices=range(6),
    )
    groups: list[ScalableRowGroup] = []
    for idx, item in enumerate(scalable_raw):
        section = _mapping(item, name=f"qpik.scalable_tasks[{idx}]")
        group_id = section.get("group_id", section.get("id", idx))
        spec = section.get(
            "selection",
            section.get("matrix", section.get("rows", section.get("indices"))),
        )
        selection = _selection_from_config(
            spec, name=f"qpik.scalable_tasks[{idx}].selection"
        )
        groups.append(
            ScalableRowGroup(
                selection=selection,
                group_id=group_id,
                row_scales=_optional_vector(
                    section.get("row_scales", section.get("scales")),
                    name=f"qpik.scalable_tasks[{idx}].row_scales",
                ),
                slack_limits=_optional_vector(
                    section.get("slack_limits"),
                    name=f"qpik.scalable_tasks[{idx}].slack_limits",
                ),
                recovery_slack_limits=_optional_vector(
                    section.get("recovery_slack_limits"),
                    name=f"qpik.scalable_tasks[{idx}].recovery_slack_limits",
                ),
                name=str(section.get("name", group_id)),
            )
        )

    return CartesianTaskProfile(
        protected_selection=protected_selection,
        protected_row_scales=_optional_vector(
            protected.get("row_scales", protected.get("scales")),
            name="qpik.protected_task.row_scales",
        ),
        protected_residual_limits=_optional_vector(
            protected.get("residual_limits"),
            name="qpik.protected_task.residual_limits",
        ),
        scalable_groups=tuple(groups),
        name=str(raw.get("name", "cartesian")),
    )


def _parse_health_thresholds(raw: dict) -> HealthThresholds:
    """Read health hysteresis thresholds while accepting compact nested YAML."""

    section = _mapping(raw.get("health"), name="qpik.health")
    arm = _mapping(section.get("arm"), name="qpik.health.arm")
    joint = _mapping(
        section.get("joint_margin", section.get("joint")),
        name="qpik.health.joint_margin",
    )
    wrist = _mapping(
        section.get("wrist_margin", section.get("wrist")),
        name="qpik.health.wrist_margin",
    )

    def pick(section_value, *keys, default=None):
        for key in keys:
            if key in section_value:
                return section_value[key]
        return default

    return HealthThresholds(
        arm_warn=pick(arm, "warn", "warn_rho", default=section.get("arm_warn", 0.08)),
        arm_danger=pick(
            arm, "danger", "danger_rho", default=section.get("arm_danger", 0.04)
        ),
        arm_exit=pick(arm, "exit", "exit_rho", default=section.get("arm_exit", 0.10)),
        joint_danger_deg=pick(
            joint,
            "danger_deg",
            "enter_deg",
            default=section.get("joint_danger_deg", 15.0),
        ),
        joint_warn_deg=pick(
            joint,
            "warn_deg",
            "warn_margin_deg",
            default=section.get("joint_warn_deg", 20.0),
        ),
        joint_exit_deg=pick(
            joint,
            "exit_deg",
            default=section.get("joint_exit_deg", 25.0),
        ),
        wrist_danger_deg=pick(
            wrist,
            "danger_deg",
            "enter_deg",
            default=section.get("wrist_danger_deg", 20.0),
        ),
        wrist_warn_deg=pick(
            wrist,
            "warn_deg",
            "warn_margin_deg",
            default=section.get("wrist_warn_deg", 25.0),
        ),
        wrist_exit_deg=pick(
            wrist,
            "exit_deg",
            default=section.get("wrist_exit_deg", 30.0),
        ),
        settling_s=section.get("settling_s", section.get("settling_time_s", 0.20)),
    )


def _parse_generic_qpik(raw: dict) -> GenericQpikRuntimeConfig:
    """Parse the generic ``qpik`` responsibility blocks.

    Backend/capacity are explicit and task rows are application data rather
    than controller semantics.
    """

    section = _mapping(raw.get("qpik"), name="qpik")
    solver = _mapping(section.get("solver"), name="qpik.solver")
    backend = str(solver.get("backend", "proxqp")).lower()
    if backend not in {"proxqp", "scipy"}:
        raise ValueError(
            "qpik.solver.backend must be explicit 'proxqp' or 'scipy' "
            f"(got {backend!r})"
        )

    def solver_value(name, default=None, *aliases):
        for key in (name, *aliases):
            if key in solver:
                return solver[key]
        return default

    qcfg = TwoLevelQpikConfig(
        backend=backend,
        max_iter=int(solver_value("max_iter", 200)),
        max_rows=int(solver_value("max_rows", 128, "max_constraint_rows", "max_p0_rows")),
        max_scalable_groups=int(
            solver_value("max_scalable_groups", 16, "max_groups")
        ),
        protected_tolerance=float(
            solver_value("protected_tolerance", 1.0e-6, "y_tolerance", "lock_tolerance")
        ),
        feasibility_tolerance=float(solver_value("feasibility_tolerance", 1.0e-6)),
        regularization=solver_value("regularization", 1.0e-6, "reg"),
        previous_velocity_weight=solver_value(
            "previous_velocity_weight", 1.0e-3, "smoothing", "smoothing_weight"
        ),
        alpha_weight=float(solver_value("alpha_weight", 1.0)),
        scalable_weight=float(solver_value("scalable_weight", 1.0)),
        posture_weight=float(solver_value("posture_weight", 1.0e-4)),
        posture_regularization=float(solver_value("posture_regularization", 1.0e-8)),
        row_scale_floor=float(solver_value("row_scale_floor", 1.0e-9)),
        warm_start=bool(solver_value("warm_start", True)),
    )
    profile = _parse_cartesian_profile(section)
    health = _parse_health_thresholds(section)
    task_scales = section.get(
        "task_velocity_scales",
        _mapping(section.get("health"), name="qpik.health").get(
            "task_velocity_scales", [0.10, 0.10, 0.10, 0.50, 0.50, 0.50]
        ),
    )
    task_scales_arr = _finite_array(
        task_scales, name="qpik.task_velocity_scales", ndim=1
    ).reshape(-1)
    if task_scales_arr.size != 6 or np.any(task_scales_arr <= 0.0):
        raise ValueError("qpik.task_velocity_scales must contain six positive values")

    indices = _mapping(section.get("indices"), name="qpik.indices")
    rail_indices = tuple(int(i) for i in indices.get("rail", (0,)))
    wrist_indices = tuple(int(i) for i in indices.get("wrist", (5, 6, 7)))
    compatibility = _mapping(
        section.get("compatibility"), name="qpik.compatibility"
    )
    raw_row = compatibility.get("overforce_task_row", section.get("overforce_task_row"))
    overforce_row = None if raw_row is None else int(raw_row)
    if overforce_row is not None and not 0 <= overforce_row < 6:
        raise ValueError("qpik.compatibility.overforce_task_row must be in [0, 6)")
    return GenericQpikRuntimeConfig(
        solver=qcfg,
        task_profile=profile,
        health=health,
        rail_indices=rail_indices,
        wrist_indices=wrist_indices,
        task_velocity_scales=task_scales_arr,
        overforce_task_row=overforce_row,
        overforce_positive_is_unsafe=bool(
            compatibility.get(
                "overforce_positive_is_unsafe",
                section.get("overforce_positive_is_unsafe", True),
            )
        ),
    )


def _finite_float(value, *, name: str) -> float:
    """Convert a config scalar and reject NaN/Inf at the loader boundary.

    Config values eventually become QP bounds and velocity envelopes.  Letting
    a non-finite value through here usually produces a much less actionable
    solver failure several layers later, so all safety-critical scalars use
    this small fail-fast conversion helper.
    """

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    """Read (mode, locked_style) from yaml.

    Schema::
        rail:
          mode: coupled | locked
          locked_style: hold | rail_only | tcp_fixed   # only if mode=locked
    """
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown qpik.hard_limits.rail.mode: {r.get('mode')!r}")


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    """Build the production controller config from responsibility blocks.

    The preferred schema is ``qpik.solver``, ``qpik.hard_limits``, task
    profiles, health and planner/guide sections.  A small read-only fallback
    for dbb's ``inner.collision`` and ``inner.rail`` is retained so older
    application YAML can be inspected without reviving the retired weighted
    QP, null-space or rail-extension APIs.
    """

    if not isinstance(raw, dict):
        raise ValueError("controller config root must be a mapping")
    timing = _mapping(raw.get("timing"), name="timing")
    inner = _mapping(raw.get("inner"), name="inner")
    qpik = _mapping(raw.get("qpik"), name="qpik")
    hard = _mapping(qpik.get("hard_limits"), name="qpik.hard_limits")
    legacy_qp = _mapping(inner.get("qp"), name="inner.qp")
    euler_order = str(
        _mapping(raw.get("frames"), name="frames").get(
            "euler_order", inner.get("euler_order", "xyz")
        )
    )

    collision_raw = _mapping(
        hard.get("collision", inner.get("collision")),
        name="qpik.hard_limits.collision",
    )
    collision = CollisionConfig(
        enabled=bool(collision_raw.get("enabled", True)),
        d_safe=_finite_float(collision_raw.get("d_safe", 0.01), name="collision.d_safe"),
        d_activate=_finite_float(
            collision_raw.get("d_activate", 0.04), name="collision.d_activate"
        ),
        gamma=_finite_float(collision_raw.get("gamma", 5.0), name="collision.gamma"),
        max_pairs=int(collision_raw.get("max_pairs", 8)),
    )
    if not 0.0 <= collision.d_safe < collision.d_activate:
        raise ValueError("collision distances must satisfy 0 <= d_safe < d_activate")
    if collision.gamma <= 0.0 or collision.max_pairs <= 0:
        raise ValueError("collision gamma/max_pairs must be positive")

    velocity_damper = _mapping(
        hard.get("velocity_damper"), name="qpik.hard_limits.velocity_damper"
    )
    damper_arm = _finite_float(
        velocity_damper.get(
            "arm_band_rad", legacy_qp.get("limit_damper_band_rad", 0.15)
        ),
        name="velocity_damper.arm_band_rad",
    )
    damper_rail = _finite_float(
        velocity_damper.get(
            "rail_band_m", legacy_qp.get("limit_damper_band_rail_m", 0.05)
        ),
        name="velocity_damper.rail_band_m",
    )
    if damper_arm < 0.0 or damper_rail < 0.0:
        raise ValueError("velocity damper bands must be non-negative")

    rail_raw = _mapping(
        hard.get("rail", inner.get("rail")), name="qpik.hard_limits.rail"
    )
    rail_mode, locked_style = _resolve_rail_mode(rail_raw)
    hw_lw = _mapping(
        _mapping(raw.get("hw"), name="hw").get("lw100"), name="hw.lw100"
    )
    soft_min = _finite_float(
        rail_raw.get("soft_min_m", hw_lw.get("soft_min_m", 0.01)),
        name="rail.soft_min_m",
    )
    soft_max = _finite_float(
        rail_raw.get("soft_max_m", hw_lw.get("soft_max_m", 0.78)),
        name="rail.soft_max_m",
    )
    travel = _finite_float(rail_raw.get("travel_m", 0.80), name="rail.travel_m")
    if not 0.0 <= soft_min < soft_max <= travel:
        raise ValueError("rail limits must satisfy 0 <= soft_min < soft_max <= travel")
    if rail_raw and hw_lw and ("soft_min_m" in hw_lw or "soft_max_m" in hw_lw):
        hw_min = _finite_float(hw_lw.get("soft_min_m", soft_min), name="hw rail soft_min")
        hw_max = _finite_float(hw_lw.get("soft_max_m", soft_max), name="hw rail soft_max")
        if abs(hw_min - soft_min) > 1.0e-6 or abs(hw_max - soft_max) > 1.0e-6:
            raise ValueError("rail soft-limit mismatch between QPIK and hardware")
    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(
            None
            if rail_raw.get("q_ref_m") is None
            else _finite_float(rail_raw["q_ref_m"], name="rail.q_ref_m")
        ),
        lock_vel_eps_m_s=_finite_float(
            rail_raw.get("lock_vel_eps_m_s", 0.0), name="rail.lock_vel_eps_m_s"
        ),
        v_max_m_s=(
            None
            if rail_raw.get("v_max_m_s") is None
            else _finite_float(rail_raw["v_max_m_s"], name="rail.v_max_m_s")
        ),
        travel_m=travel,
        soft_min_m=soft_min,
        soft_max_m=soft_max,
    )

    def hard_value(name: str, legacy_name: str, default):
        return hard.get(name, inner.get(legacy_name, default))

    return JointIkConfig(
        dt=_finite_float(timing.get("dt_ms", 5.0), name="timing.dt_ms") / 1000.0,
        feedback_timeout_s=_finite_float(
            timing.get("feedback_timeout_ms", 50.0),
            name="timing.feedback_timeout_ms",
        )
        / 1000.0,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        generic_qpik=_parse_generic_qpik(raw),
        collision=collision,
        limit_damper_band_rad=damper_arm,
        limit_damper_band_rail_m=damper_rail,
        rail=rail,
        v_scale=_finite_float(hard_value("v_scale", "v_scale", 0.5), name="v_scale"),
        a_max_arm_rad_s2=_finite_float(
            hard_value("a_max_arm_rad_s2", "a_max_arm", 20.0), name="a_max_arm_rad_s2"
        ),
        a_max_rail_m_s2=_finite_float(
            hard_value("a_max_rail_m_s2", "a_max_rail_m_s2", 0.30), name="a_max_rail_m_s2"
        ),
        position_margin_rad=math.radians(
            _finite_float(
                hard_value("position_margin_deg", "position_margin_deg", 1.0),
                name="position_margin_deg",
            )
        ),
        position_margin_rail_m=_finite_float(
            hard_value("position_margin_rail_mm", "position_margin_rail_mm", 0.0),
            name="position_margin_rail_mm",
        )
        / 1000.0,
        resync_err_rad=math.radians(
            _finite_float(
                hard_value("command_lead_arm_deg", "resync_err_deg", 6.0),
                name="command_lead_arm_deg",
            )
        ),
        resync_err_rail_m=_finite_float(
            hard_value("command_lead_rail_mm", "resync_err_rail_mm", 20.0),
            name="command_lead_rail_mm",
        )
        / 1000.0,
    )


__all__ = ["build_joint_ik_config"]
