"""RM75+rail adapter for the generic SRS posture planner.

The graph and guide implementations in :mod:`srs_rail_planner` and
:mod:`continuous_guide` are intentionally robot agnostic.  This module is the
small RM75 binding: it turns a measured eight-joint state and an arbitrary
pose/reference horizon into SRS candidates by calling the repository's
closed-form :func:`rm75_control.kinematics.srs_ik.srs_ik` for *every* rail,
swivel, branch and winding candidate.  The resulting short corridor is then
advanced by :class:`ContinuousRailGuide`; only its ``PostureGuide`` is
published to QP2.  No method in this adapter sends a robot command.

The adapter accepts both the generic ``PosturePlanner`` submit contract and a
direct :class:`SrsPlanRequest` for deterministic/offline tests.  A horizon is
optional.  Without one the candidate layer is built from the measured task
pose, so the planner never invents a line, scan direction or periodic motion.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from rm75_control.control.joint_admittance_8dof.continuous_guide import (
    ContinuousGuideLimits,
    ContinuousRailGuide,
    RedundancyState,
    SrsMappingContext,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    PostureGuide,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.posture_planner import (
    PlanComputation,
    PlannerSnapshot,
    PosturePlanner,
    PosturePlanningRequest,
)
from rm75_control.control.joint_admittance_8dof.srs_rail_planner import (
    CandidateLayerContext,
    NodeEvaluation,
    NodeEvaluationContext,
    NoFeasiblePlanError,
    SrsCandidate,
    SrsPlanRequest,
    SrsPlannerConfig,
    SrsRailPlan,
    SrsRailPosturePlanner,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.kinematics.srs_ik import (
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    srs_ik,
)


_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Rm75SrsPlannerConfig:
    """Configuration for the RM75-specific candidate adapter.

    ``rail_candidates`` and ``psi_candidates`` are deliberately explicit
    candidate sets, not commands.  The graph's edge checks and the QP's P0
    limits remain authoritative for continuous execution.
    """

    rail_candidates: tuple[float, ...] | None = None
    psi_candidates: tuple[float, ...] | None = None
    branch_candidates: tuple[int, ...] = tuple(range(8))
    winding_offsets: tuple[int, ...] = (-1, 0, 1)
    default_rail_grid_points: int = 9
    default_psi_grid_points: int = 13
    horizon_times_s: tuple[float, ...] = (0.10, 0.20, 0.30)
    horizon_dt_s: float = 0.10
    rail_origin_y_m: float = -0.40
    minimum_arm_health: float = 0.04
    collision_clearance_min_m: float = 0.0
    reanchor_rail_error_m: float = 0.03
    reanchor_joint_error_rad: float = 0.30

    def __post_init__(self) -> None:
        if self.default_rail_grid_points < 2:
            raise ValueError("default_rail_grid_points must be >= 2")
        if self.default_psi_grid_points < 2:
            raise ValueError("default_psi_grid_points must be >= 2")
        if self.horizon_dt_s <= 0.0 or not math.isfinite(float(self.horizon_dt_s)):
            raise ValueError("horizon_dt_s must be finite and > 0")
        if not math.isfinite(float(self.rail_origin_y_m)):
            raise ValueError("rail_origin_y_m must be finite")
        if self.minimum_arm_health < 0.0 or not math.isfinite(float(self.minimum_arm_health)):
            raise ValueError("minimum_arm_health must be finite and >= 0")
        if self.reanchor_rail_error_m < 0.0 or self.reanchor_joint_error_rad < 0.0:
            raise ValueError("reanchor thresholds must be non-negative")
        if any(not math.isfinite(float(v)) for v in self.horizon_times_s):
            raise ValueError("horizon_times_s must be finite")


def _as_pose(value: Any) -> np.ndarray | None:
    """Extract a six-vector xyz/xyz-Euler pose from common task references."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("pose_meas", "pose", "pose_tcp", "target_pose", "waypoint"):
            if key in value:
                pose = _as_pose(value[key])
                if pose is not None:
                    return pose
        return None
    for key in ("pose_meas", "pose", "pose_tcp", "target_pose", "waypoint"):
        if hasattr(value, key):
            pose = _as_pose(getattr(value, key))
            if pose is not None:
                return pose
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    return arr[:6].copy()


def _call_reference_horizon(
    horizon: Any,
    state: RobotState,
    sample_times: Sequence[float],
) -> tuple[Any, ...]:
    """Resolve a finite horizon without assuming its trajectory geometry."""

    if horizon is None:
        return ()
    if hasattr(horizon, "sample"):
        try:
            # ReferenceHorizon.sample returns a sequence of references.
            values = horizon.sample(float(state.timestamp), state=state)
        except TypeError:
            try:
                values = horizon.sample(float(state.timestamp), 0.0, state)
            except TypeError:
                values = [horizon.sample(float(t), state) for t in sample_times]
        if values is None:
            return ()
        if isinstance(values, (str, bytes)):
            return (values,)
        try:
            return tuple(values)
        except TypeError:
            return (values,)
    if callable(horizon):
        return tuple(horizon(float(t)) for t in sample_times)
    if isinstance(horizon, Mapping):
        for key in ("samples", "references", "waypoints", "poses"):
            if key in horizon:
                value = horizon[key]
                try:
                    return tuple(value)
                except TypeError:
                    return (value,)
    try:
        return tuple(horizon)
    except TypeError:
        return (horizon,)


def _interpolate_pose(source: Any, target: Any, alpha: float) -> Any:
    """Interpolate pose samples for interior edge SRS checks.

    Translation is linear and orientation uses shortest-arc quaternion slerp
    when SciPy is available.  If a caller supplies a non-pose waypoint, the
    target/source object is retained; the adapter will then fail closed when
    SRS IK cannot consume it.
    """

    a = _as_pose(source)
    b = _as_pose(target)
    if a is None or b is None:
        return target if alpha >= 1.0 else source
    out = (1.0 - float(alpha)) * a[:3] + float(alpha) * b[:3]
    try:
        from scipy.spatial.transform import Rotation, Slerp

        rotations = Rotation.from_euler("xyz", np.vstack((a[3:6], b[3:6])))
        interp = Slerp([0.0, 1.0], rotations)([float(alpha)])[0]
        return np.concatenate((out, interp.as_euler("xyz")))
    except Exception:
        return np.concatenate((out, (1.0 - float(alpha)) * a[3:6] + float(alpha) * b[3:6]))


class Rm75SrsPosturePlanner:
    """Non-blocking RM75 SRS/rail posture planner adapter.

    Parameters
    ----------
    kinematics:
        Existing eight-DOF :class:`RobotKinematics` (or a compatible fake in
        tests).  The adapter never mutates it.
    collision_checker, health_checker:
        Optional callbacks receiving ``(q_full, waypoint)``.  A callback may
        instead accept only ``q_full``.  A numeric return value is interpreted
        as a clearance/health margin and must be non-negative.
    ruckig_module:
        Dependency injection point for deterministic tests.  Passing ``False``
        disables continuous output and causes a fail-closed invalid plan.
    """

    def __init__(
        self,
        kinematics: RobotKinematics | Any | None = None,
        *,
        kin: RobotKinematics | Any | None = None,
        adapter_config: Rm75SrsPlannerConfig | None = None,
        config: SrsPlannerConfig | None = None,
        guide_limits: ContinuousGuideLimits | None = None,
        collision_checker: Callable[..., Any] | None = None,
        health_checker: Callable[..., Any] | None = None,
        ruckig_module: Any = None,
        dt_s: float = 0.005,
        clock: Callable[[], float] = time.monotonic,
        stale_after_s: float = 0.5,
        fade_after_s: float = 0.5,
        autostart: bool = True,
    ) -> None:
        if kinematics is not None and kin is not None:
            raise ValueError("provide either kinematics or kin, not both")
        self.kin = kinematics if kinematics is not None else kin
        self.kin = self.kin if self.kin is not None else RobotKinematics()
        self.adapter_config = adapter_config or Rm75SrsPlannerConfig()
        self.guide_limits = guide_limits or ContinuousGuideLimits()
        self._collision_checker = collision_checker
        self._health_checker = health_checker
        self._clock = clock
        self._R_flange_tcp, self._t_flange_tcp = flange_tcp_from_kin(self.kin)
        self._guide = ContinuousRailGuide(
            self._map_srs,
            dt_s=dt_s,
            limits=self.guide_limits,
            ruckig_module=ruckig_module,
        )
        self._guide_initialized = False
        self._guide_discrete_key: tuple[int, int] | None = None
        self._guide_q_anchor: np.ndarray | None = None
        self._guide_target_key: tuple | None = None
        self._current_discrete_key: tuple[int, int] | None = None
        self._active_waypoint: np.ndarray | None = None
        self._active_state: RobotState | None = None
        self._guide_lock = threading.Lock()
        self._last_posture_guide: PostureGuide | None = None

        planner_cfg = config or SrsPlannerConfig(
            rail_v_max_m_s=self.guide_limits.rail_v_max_m_s,
            rail_a_max_m_s2=self.guide_limits.rail_a_max_m_s2,
            arm_v_max_rad_s=self.guide_limits.arm_v_max_rad_s,
            arm_a_max_rad_s2=self.guide_limits.arm_a_max_rad_s2,
            psi_v_max_rad_s=self.guide_limits.psi_v_max_rad_s,
            psi_a_max_rad_s2=self.guide_limits.psi_a_max_rad_s2,
        )
        # The generic graph is synchronous here; the outer generic planner is
        # the sole worker so no nested planner thread can outlive a request.
        self._graph = SrsRailPosturePlanner(
            candidate_provider=self._candidate_provider,
            node_evaluator=self._node_evaluator,
            waypoint_interpolator=_interpolate_pose,
            config=planner_cfg,
            clock=clock,
            autostart=False,
        )
        self._async = PosturePlanner(
            self._calculate_generic,
            stale_after_s=stale_after_s,
            fade_after_s=fade_after_s,
            clock=clock,
            name="rm75-srs-posture-planner",
            autostart=autostart,
        )

    @property
    def guide(self) -> ContinuousRailGuide:
        return self._guide

    @property
    def graph(self) -> SrsRailPosturePlanner:
        return self._graph

    def submit(
        self,
        request: PosturePlanningRequest | None = None,
        *,
        robot_state: RobotState | None = None,
        current_task_reference: object | None = None,
        optional_reference_horizon: object | None = None,
        timestamp_s: float | None = None,
    ) -> int:
        return self._async.submit(
            request,
            robot_state=robot_state,
            current_task_reference=current_task_reference,
            optional_reference_horizon=optional_reference_horizon,
            timestamp_s=timestamp_s,
        )

    def latest(self, *, now_s: float | None = None) -> PlannerSnapshot[PostureGuide]:
        return self._async.latest(now_s=now_s)

    def sample_guide(
        self,
        *,
        robot_state: RobotState,
        current_task_reference: object,
        now_s: float | None = None,
    ) -> PostureGuide | None:
        """Advance the committed rail/psi corridor by one servo tick.

        Graph search remains on the background worker.  Only this bounded
        Ruckig+SRS sample runs on the control thread, so a 10 Hz planner does
        not accidentally produce a 10 Hz/staircase rail guide.
        """

        del now_s
        pose = _as_pose(current_task_reference)
        if pose is None:
            pose = np.asarray(self.kin.fk_pose(robot_state.q_meas), dtype=float).reshape(6)
        with self._guide_lock:
            if not self._guide_initialized or not self._guide.available:
                return self._last_posture_guide
            target = self._guide.target
            q_meas = np.asarray(robot_state.q_meas, dtype=float).reshape(-1)
            if target is not None and q_meas.size == 8:
                psi_meas = float(psi_from_q(q_meas[1:]))
                psi_error = (
                    float(target.psi_rad) - psi_meas + math.pi
                ) % _TWO_PI - math.pi
                if (
                    abs(float(target.rail_m) - float(q_meas[0])) <= 1.0e-8
                    and abs(psi_error) <= 1.0e-7
                    and int(target.branch) == int(branch_from_q(q_meas[1:]))
                    and np.linalg.norm(robot_state.qdot_applied_prev) <= 1.0e-8
                ):
                    prior = self._last_posture_guide
                    metadata = dict(prior.metadata) if prior is not None else {}
                    metadata.update(
                        {
                            "rail_m": float(q_meas[0]),
                            "psi_rad": psi_meas,
                            "rail_velocity_m_s": 0.0,
                            "psi_velocity_rad_s": 0.0,
                            "rail_acceleration_m_s2": 0.0,
                            "psi_acceleration_rad_s2": 0.0,
                            "guide_status": "finished",
                        }
                    )
                    held = PostureGuide(
                        q_goal=q_meas,
                        qdot_guide=np.zeros(8),
                        valid_until=float(robot_state.timestamp)
                        + max(4.0 * robot_state.dt, 0.05),
                        quality=1.0 if prior is None else prior.quality,
                        planner_state="finished",
                        source=type(self).__name__,
                        created_at=float(robot_state.timestamp),
                        metadata=metadata,
                    )
                    self._last_posture_guide = held
                    return held
            sample = self._guide.update(mapping_input={"pose": pose.copy()})
            if not sample.valid or sample.q_goal is None or sample.qdot_guide is None:
                return None
            prior = self._last_posture_guide
            metadata = dict(prior.metadata) if prior is not None else {}
            metadata.update(
                {
                    "branch": sample.branch,
                    "winding": sample.winding,
                    "rail_m": sample.rail_m,
                    "psi_rad": sample.psi_rad,
                    "rail_velocity_m_s": sample.rail_velocity_m_s,
                    "psi_velocity_rad_s": sample.psi_velocity_rad_s,
                    "rail_acceleration_m_s2": sample.rail_acceleration_m_s2,
                    "psi_acceleration_rad_s2": sample.psi_acceleration_rad_s2,
                    "guide_status": sample.status.value,
                }
            )
            quality = 1.0 if prior is None else prior.quality
            guide = PostureGuide(
                q_goal=sample.q_goal,
                qdot_guide=sample.qdot_guide,
                valid_until=float(robot_state.timestamp) + max(4.0 * robot_state.dt, 0.05),
                quality=quality,
                planner_state=sample.status.value,
                source=type(self).__name__,
                created_at=float(robot_state.timestamp),
                metadata=metadata,
            )
            self._last_posture_guide = guide
            return guide

    def wait_for(self, sequence: int | None = None, timeout_s: float = 1.0) -> bool:
        return self._async.wait_for(sequence, timeout_s)

    def shutdown(self, **kwargs: Any) -> bool:
        result = self._async.shutdown(**kwargs)
        self._graph.shutdown(**kwargs)
        return result

    def reset_commit(self) -> None:
        self._graph.reset_commit()
        with self._guide_lock:
            self._guide_initialized = False
            self._guide_discrete_key = None
            self._guide_target_key = None
            self._last_posture_guide = None

    def plan_now(self, request: PosturePlanningRequest | SrsPlanRequest) -> PostureGuide | SrsRailPlan:
        """Synchronously calculate a guide (or graph plan for direct tests)."""

        if isinstance(request, SrsPlanRequest):
            return self._graph.plan_now(request)
        computation = self._calculate_generic(request)
        if not computation.valid or computation.value is None:
            raise NoFeasiblePlanError(computation.reason or "no feasible RM75 posture guide")
        return computation.value

    plan = plan_now

    def _shoulder_y(self, rail_m: float) -> float:
        """Convert the prismatic joint coordinate to SRS shoulder world-Y."""

        return float(self.adapter_config.rail_origin_y_m + float(rail_m))

    @staticmethod
    def _hint(reference: Any, key: str, default: Any = None) -> Any:
        if isinstance(reference, Mapping):
            hints = reference.get("posture_hint", {})
            if isinstance(hints, Mapping) and key in hints:
                return hints[key]
            return reference.get(key, default)
        hints = getattr(reference, "posture_hint", None)
        if isinstance(hints, Mapping) and key in hints:
            return hints[key]
        return getattr(reference, key, default)

    def _current_state(self, request: PosturePlanningRequest) -> tuple[RobotState, np.ndarray, SrsCandidate]:
        state = request.robot_state
        if not isinstance(state, RobotState):
            # Keep the adapter useful with immutable compatible test doubles.
            qdot_prev = getattr(state, "qdot_applied_prev", None)
            if qdot_prev is None:
                qdot_prev = getattr(state, "qdot_prev")
            state = RobotState(
                q_meas=getattr(state, "q_meas"),
                q_cmd=getattr(state, "q_cmd"),
                qdot_applied_prev=qdot_prev,
                dt=getattr(state, "dt"),
                contact_active=bool(getattr(state, "contact_active", False)),
                timestamp=float(getattr(state, "timestamp", self._clock())),
            )
        q = np.asarray(state.q_meas, dtype=float).reshape(-1)
        if q.size != 8 or not np.all(np.isfinite(q)):
            raise ValueError("RM75 SRS planner requires a finite eight-DOF q_meas")
        pose = _as_pose(request.current_task_reference)
        if pose is None:
            pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        q_arm = q[1:]
        psi = float(psi_from_q(q_arm))
        branch = int(branch_from_q(q_arm))
        winding_hint = self._hint(request.current_task_reference, "winding", 0)
        try:
            winding = int(winding_hint)
        except (TypeError, ValueError):
            winding = 0
        current = SrsCandidate(
            rail_m=float(q[0]),
            psi_rad=psi,
            branch=branch,
            winding=winding,
            q=q,
            score=(0.0,),
            tag="measured",
            metadata={
                "pose": pose.copy(),
                "q": q.copy(),
                "rail_m": float(q[0]),
                "psi_rad": psi,
                "branch": branch,
                "winding": winding,
            },
        )
        return state, pose, current

    def _horizon(self, request: PosturePlanningRequest, state: RobotState, current_pose: np.ndarray) -> tuple[np.ndarray, ...]:
        raw = _call_reference_horizon(
            request.optional_reference_horizon,
            state,
            self.adapter_config.horizon_times_s,
        )
        if not raw:
            return ()
        poses: list[np.ndarray] = []
        for value in raw:
            pose = _as_pose(value)
            if pose is None:
                # A malformed horizon is not silently replaced by a line or
                # current pose; fail closed at the graph request boundary.
                raise ValueError("reference horizon sample does not contain a pose6")
            poses.append(pose)
        return tuple(poses)

    def _calculate_generic(self, request: PosturePlanningRequest) -> PlanComputation[PostureGuide]:
        try:
            state, current_pose, current = self._current_state(request)
            self._current_discrete_key = current.discrete_key
            horizon = self._horizon(request, state, current_pose)
            srs_request = SrsPlanRequest(
                current=current,
                horizon=horizon,
                current_waypoint=current_pose,
                contact=bool(state.contact_active),
                layer_dt_s=(
                    tuple(
                        np.diff(np.concatenate(([0.0], np.asarray(self.adapter_config.horizon_times_s[: len(horizon)]))))
                    )
                    if horizon
                    else self.adapter_config.horizon_dt_s
                ),
                q_current=q_current if (q_current := tuple(state.q_meas)) else None,
                qdot_current=tuple(state.qdot_applied_prev),
                context=request.current_task_reference,
                now_s=float(state.timestamp),
            )
            plan = self._graph.plan_now(srs_request)
            guide = self._guide_from_plan(plan, state, current_pose)
            if guide is None:
                return PlanComputation.invalid(
                    self._guide.last_sample.reason if self._guide.last_sample else self._guide.availability.reason
                )
            return PlanComputation(guide)
        except Exception as exc:
            return PlanComputation.invalid(f"{type(exc).__name__}: {exc}")

    def _rail_values(self, current: float) -> tuple[float, ...]:
        configured = self.adapter_config.rail_candidates
        if configured is not None:
            values = tuple(float(v) for v in configured)
        else:
            lo = float(np.asarray(getattr(self.kin, "q_lower", [-math.inf] * 8))[0])
            hi = float(np.asarray(getattr(self.kin, "q_upper", [math.inf] * 8))[0])
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                values = tuple(np.linspace(lo, hi, self.adapter_config.default_rail_grid_points))
            else:
                values = tuple(float(current + v) for v in (-0.2, -0.1, 0.0, 0.1, 0.2))
        return tuple(dict.fromkeys([float(current), *values]))

    def _psi_values(self, current: float) -> tuple[float, ...]:
        configured = self.adapter_config.psi_candidates
        if configured is not None:
            values = tuple(float(v) for v in configured)
        else:
            n = self.adapter_config.default_psi_grid_points
            values = tuple(float(v) for v in np.linspace(-math.pi, math.pi, n, endpoint=False))
        wrapped_current = float((current + math.pi) % _TWO_PI - math.pi)
        values = tuple(dict.fromkeys([wrapped_current, *values]))
        return values

    def _winding_values(self, current: int) -> tuple[int, ...]:
        return tuple(dict.fromkeys(int(current) + int(v) for v in self.adapter_config.winding_offsets))

    def _candidate_provider(self, context: CandidateLayerContext) -> Iterable[SrsCandidate]:
        request = context.request
        current = request.current
        pose = _as_pose(context.waypoint)
        if pose is None:
            return ()
        # Always retain the measured point in each layer.  This gives the
        # graph a local manifold when planning the current pose.  A future
        # horizon layer must not reuse the measured q unless its pose really
        # equals the current pose; doing so would accept a false endpoint.
        current_pose = _as_pose(current.metadata)
        hold_current = (
            context.layer_index == 0
            and current_pose is not None
            and np.allclose(pose, current_pose, atol=1e-9, rtol=0.0)
        )
        values: list[SrsCandidate] = [current] if hold_current else []
        rails = self._rail_values(current.rail_m)
        psis = self._psi_values(current.psi_rad)
        branches = tuple(int(v) & 0b111 for v in self.adapter_config.branch_candidates)
        windings = self._winding_values(current.winding)
        seen: set[tuple] = {current.key()} if hold_current else set()
        for rail in rails:
            for psi in psis:
                for branch in branches:
                    for winding in windings:
                        # Use the exact same quantisation as
                        # ``SrsCandidate.key``.  Mixing raw rounded floats
                        # with the candidate's resolution-scaled integers
                        # used to duplicate the measured hold, after which
                        # the analytically re-solved duplicate could win by a
                        # few microradians and create a non-zero hold guide.
                        key = (
                            round(float(rail) / 1e-4),
                            round(float(psi + _TWO_PI * winding) / 1e-3),
                            branch,
                            winding,
                        )
                        if key in seen:
                            continue
                        q_arm = srs_ik(
                            pose,
                            psi + _TWO_PI * winding,
                            branch,
                            y_rail=self._shoulder_y(rail),
                            R_flange_tcp=self._R_flange_tcp,
                            t_flange_tcp=self._t_flange_tcp,
                        )
                        if q_arm is None:
                            continue
                        q = np.concatenate(([float(rail)], np.asarray(q_arm, dtype=float)))
                        if q.size != 8 or not np.all(np.isfinite(q)):
                            continue
                        score = self._candidate_score(q, current, branch, winding)
                        candidate = SrsCandidate(
                            rail_m=float(rail),
                            psi_rad=float(psi),
                            branch=branch,
                            winding=winding,
                            q=q,
                            score=score,
                            tag="rm75_srs",
                            metadata={
                                "pose": pose.copy(),
                                "q": q.copy(),
                                "rail_m": float(rail),
                                "psi_rad": float(psi),
                                "branch": branch,
                                "winding": winding,
                            },
                        )
                        values.append(candidate)
                        seen.add(key)
        return values

    def _candidate_score(self, q: np.ndarray, current: SrsCandidate, branch: int, winding: int) -> tuple[float, ...]:
        health = self._arm_health(q)
        margin = self._joint_margin(q)
        wrist = self._wrist_margin(q)
        return (
            -float(health),
            -float(margin),
            -float(wrist),
            float(abs(int(branch) - int(current.branch)) > 0),
            float(abs(int(winding) - int(current.winding)) > 0),
            abs(float(q[0]) - float(current.rail_m)),
            abs(float((q[1] if q.size > 1 else 0.0) - current.psi_rad)),
        )

    def _joint_margin(self, q: np.ndarray) -> float:
        try:
            lo = np.asarray(self.kin.q_lower, dtype=float).reshape(-1)
            hi = np.asarray(self.kin.q_upper, dtype=float).reshape(-1)
            width = np.maximum(hi - lo, 1e-9)
            return float(np.min(np.minimum((q - lo) / width, (hi - q) / width)))
        except Exception:
            return 1.0

    def _wrist_margin(self, q: np.ndarray) -> float:
        try:
            lo = float(np.asarray(self.kin.q_lower)[6])
            hi = float(np.asarray(self.kin.q_upper)[6])
            return float(min(q[6] - lo, hi - q[6]) / max(hi - lo, 1e-9))
        except Exception:
            return 1.0

    def _arm_health(self, q: np.ndarray) -> float:
        try:
            J = np.asarray(self.kin.jacobian(q), dtype=float)
            singular = np.linalg.svd(J[:, 1:] if J.shape[1] >= 8 else J, compute_uv=False)
            return float(np.min(singular)) if singular.size else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _callback_value(callback: Callable[..., Any], q: np.ndarray, waypoint: Any) -> Any:
        try:
            return callback(q, waypoint)
        except TypeError:
            return callback(q)

    def _node_evaluator(self, context: NodeEvaluationContext) -> NodeEvaluation:
        candidate = context.candidate
        q = None if candidate.q is None else np.asarray(candidate.q, dtype=float).reshape(-1)
        if q is None:
            # Generic graph edge samples carry only rail/psi/branch/winding;
            # remap each interior sample through the same closed-form SRS IK
            # used by the candidate provider.  Endpoint-only checks are not
            # sufficient for collision or joint-limit safety.
            pose = _as_pose(context.waypoint)
            if pose is not None:
                q_arm = srs_ik(
                    pose,
                    float(candidate.unwrapped_psi_rad),
                    int(candidate.branch),
                    y_rail=self._shoulder_y(candidate.rail_m),
                    R_flange_tcp=self._R_flange_tcp,
                    t_flange_tcp=self._t_flange_tcp,
                )
                if q_arm is not None:
                    q = np.concatenate(([float(candidate.rail_m)], np.asarray(q_arm, dtype=float)))
            # The measured candidate is already a validated eight-vector.  A
            # hold must remain possible even when a non-coaxial TCP makes the
            # analytical SRS model unable to re-solve that exact pose.
            if q is None and candidate.discrete_key == context.request.current.discrete_key:
                psi_delta = (float(candidate.psi_rad) - float(context.request.current.psi_rad) + math.pi) % _TWO_PI - math.pi
                if (
                    abs(float(candidate.rail_m) - float(context.request.current.rail_m)) < 1e-9
                    and abs(psi_delta) < 1e-8
                ):
                    q = np.asarray(context.request.current.q, dtype=float).copy()
        if q is None or q.size != 8 or not np.all(np.isfinite(q)):
            return NodeEvaluation.rejected("RM75 SRS IK did not produce an eight-vector")
        try:
            lo = np.asarray(self.kin.q_lower, dtype=float).reshape(8)
            hi = np.asarray(self.kin.q_upper, dtype=float).reshape(8)
            if np.any(q < lo - 1e-8) or np.any(q > hi + 1e-8):
                return NodeEvaluation.rejected("joint or rail position limit")
        except Exception:
            pass
        health = self._arm_health(q)
        if health < self.adapter_config.minimum_arm_health:
            return NodeEvaluation.rejected("arm health below danger threshold")
        if self._health_checker is not None:
            try:
                result = self._callback_value(self._health_checker, q, context.waypoint)
                if isinstance(result, (bool, np.bool_)):
                    healthy = bool(result)
                else:
                    healthy = float(result) >= self.adapter_config.minimum_arm_health
                if not healthy:
                    return NodeEvaluation.rejected("health checker rejected node")
            except Exception as exc:
                return NodeEvaluation.rejected(f"health checker: {type(exc).__name__}: {exc}")
        if self._collision_checker is not None:
            try:
                result = self._callback_value(self._collision_checker, q, context.waypoint)
                if isinstance(result, (bool, np.bool_)):
                    collision_free = bool(result)
                else:
                    collision_free = float(result) >= self.adapter_config.collision_clearance_min_m
                if not collision_free:
                    return NodeEvaluation.rejected("collision checker rejected node")
            except Exception as exc:
                return NodeEvaluation.rejected(f"collision checker: {type(exc).__name__}: {exc}")
        metadata = {"arm_health": health, "joint_margin": self._joint_margin(q), "wrist_margin": self._wrist_margin(q)}
        return NodeEvaluation(True, q=q, score=candidate.score, metadata=metadata)

    def _map_srs(self, context: SrsMappingContext) -> np.ndarray | None:
        # Preserve an already validated endpoint exactly.  Re-solving the
        # same (rail, psi, branch, winding) state can move a measured hold by
        # a few microradians because the analytic inverse and FK are not
        # bit-for-bit inverses.  The metadata is accepted only when all four
        # redundancy/discrete coordinates match; intermediate Ruckig states
        # must still be mapped through a fresh SRS solve.
        if isinstance(context.mapping_input, Mapping):
            try:
                q_exact = np.asarray(context.mapping_input.get("q"), dtype=float).reshape(-1)
                rail_exact = float(context.mapping_input["rail_m"])
                psi_exact = float(context.mapping_input["psi_rad"])
                branch_exact = int(context.mapping_input["branch"])
                winding_exact = int(context.mapping_input["winding"])
                psi_error = (psi_exact - float(context.psi_rad) + math.pi) % _TWO_PI - math.pi
                if (
                    q_exact.size == 8
                    and np.all(np.isfinite(q_exact))
                    and abs(rail_exact - float(context.rail_m)) <= 1e-10
                    and abs(psi_error) <= 1e-10
                    and branch_exact == int(context.branch)
                    and winding_exact == int(context.winding)
                ):
                    return q_exact.copy()
            except (KeyError, TypeError, ValueError):
                pass
        pose = _as_pose(context.mapping_input)
        if pose is None:
            return None
        q_arm = srs_ik(
            pose,
            float(context.psi_unwrapped_rad),
            int(context.branch),
            y_rail=self._shoulder_y(context.rail_m),
            R_flange_tcp=self._R_flange_tcp,
            t_flange_tcp=self._t_flange_tcp,
        )
        if q_arm is None:
            # Endpoint hold fallback only.  Interior continuous guide states
            # must still obtain a fresh SRS solution for their rail/psi.
            if isinstance(context.mapping_input, Mapping):
                try:
                    fallback = np.asarray(context.mapping_input.get("q"), dtype=float).reshape(-1)
                    if fallback.size == 8 and abs(float(fallback[0]) - float(context.rail_m)) < 1e-8:
                        return fallback
                except (TypeError, ValueError):
                    pass
            return None
        return np.concatenate(([float(context.rail_m)], np.asarray(q_arm, dtype=float)))

    def _guide_from_plan(self, plan: SrsRailPlan, state: RobotState, current_pose: np.ndarray) -> PostureGuide | None:
        with self._guide_lock:
            return self._guide_from_plan_locked(plan, state, current_pose)

    def _guide_from_plan_locked(
        self, plan: SrsRailPlan, state: RobotState, current_pose: np.ndarray
    ) -> PostureGuide | None:
        """Commit a graph target without advancing its servo trajectory."""

        if not self._guide.available:
            return None
        current_q = np.asarray(state.q_meas, dtype=float)
        current_arm = current_q[1:]
        current_discrete = self._current_discrete_key or (int(branch_from_q(current_arm)), 0)
        target = plan.short_target
        # Contact graph filtering already enforces this; retain an explicit
        # guard at the guide boundary so a future graph implementation cannot
        # accidentally command a branch/winding switch during contact.
        if state.contact_active and target.discrete_key != current_discrete:
            return None
        needs_reset = not self._guide_initialized
        if self._guide_discrete_key is not None and self._guide_discrete_key != target.discrete_key:
            needs_reset = True
        if self._guide_q_anchor is not None:
            if abs(float(current_q[0] - self._guide_q_anchor[0])) > self.adapter_config.reanchor_rail_error_m:
                needs_reset = True
            elif float(np.max(np.abs(current_q[1:] - self._guide_q_anchor[1:]))) > self.adapter_config.reanchor_joint_error_rad:
                needs_reset = True
        psi_current = float(psi_from_q(current_arm))
        winding_current = int(self._hint(self._active_state, "winding", 0) or 0) if self._active_state is not None else 0
        if needs_reset:
            self._guide.reset(
                state=RedundancyState(
                    rail_m=float(current_q[0]),
                    psi_rad=psi_current,
                    branch=int(branch_from_q(current_arm)),
                    winding=winding_current,
                    rail_velocity_m_s=float(state.qdot_applied_prev[0]),
                    psi_velocity_rad_s=0.0,
                ),
                q_goal=current_q,
                qdot_guide=state.qdot_applied_prev,
                mapping_input=current_pose,
            )
            self._guide_initialized = True
            self._guide_discrete_key = target.discrete_key
            self._guide_q_anchor = current_q.copy()
        self._active_state = state
        target_pose = _as_pose(target.metadata)
        # Preserve the endpoint q in metadata for a validated measured hold;
        # the guide's mapping callback still extracts the pose field.
        self._active_waypoint = target.metadata if target_pose is not None else current_pose
        self._guide.set_target(
            rail_m=float(target.rail_m),
            psi_rad=float(target.psi_rad),
            branch=int(target.branch),
            winding=int(target.winding),
            mapping_input=self._active_waypoint,
        )
        score = plan.short_target.score
        health = float(np.clip(1.0 / (1.0 + max(0.0, -float(score[0]))), 0.0, 1.0))
        metadata = {
            "corridor": plan.corridor,
            "branch": int(target.branch),
            "winding": int(target.winding),
            "rail_m": float(current_q[0]),
            "psi_rad": psi_current,
            "rail_velocity_m_s": float(state.qdot_applied_prev[0]),
            "psi_velocity_rad_s": 0.0,
            "rail_acceleration_m_s2": 0.0,
            "psi_acceleration_rad_s2": 0.0,
            "guide_status": "committed",
            "score": plan.score,
        }
        guide = PostureGuide(
            q_goal=current_q,
            qdot_guide=np.zeros_like(current_q),
            valid_until=float(state.timestamp) + max(4.0 * state.dt, 0.05),
            quality=health,
            planner_state="committed",
            source=type(self).__name__,
            created_at=float(state.timestamp),
            metadata=metadata,
        )
        self._last_posture_guide = guide
        return guide


RM75SrsPosturePlanner = Rm75SrsPosturePlanner
Rm75SrsRailPlanner = Rm75SrsPosturePlanner
RM75SrsRailPlanner = Rm75SrsPosturePlanner


__all__ = [
    "RM75SrsPosturePlanner",
    "RM75SrsRailPlanner",
    "Rm75SrsPlannerConfig",
    "Rm75SrsPosturePlanner",
    "Rm75SrsRailPlanner",
]
