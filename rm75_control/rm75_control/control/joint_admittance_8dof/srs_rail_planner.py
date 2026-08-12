"""Injectable SRS + rail candidate-graph posture planner.

This module deliberately knows nothing about scan shapes or tool directions.
Callers provide candidate layers and kinematic/safety evaluators.  The planner
owns the invariants which must not be optional: every graph edge is sampled,
angles are unwrapped, velocity/acceleration limits are checked, and collision
and health results fail closed.

Scores are tuples and are compared lexicographically (lower is better).  They
are never collapsed into a weighted scalar.  Only a short committed corridor
is published; the full search path remains an implementation detail.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.posture_planner import (
    PlanComputation,
    PlannerSnapshot,
    PosturePlanner,
)


Score = tuple[float, ...]
_TWO_PI = 2.0 * math.pi


def _score(value: float | Sequence[float] | None) -> Score:
    if value is None:
        return (0.0,)
    if np.isscalar(value):
        return (float(value),)
    out = tuple(float(v) for v in value)
    return out or (0.0,)


def _score_add(*values: Score) -> Score:
    width = max((len(v) for v in values), default=1)
    return tuple(sum(v[i] if i < len(v) else 0.0 for v in values) for i in range(width))


def lexicographic_materially_better(
    challenger: Sequence[float],
    incumbent: Sequence[float],
    margin: float | Sequence[float] = 0.0,
) -> bool:
    """Return whether ``challenger`` has a meaningful lexicographic gain."""

    a, b = _score(challenger), _score(incumbent)
    margins = _score(margin)
    width = max(len(a), len(b))
    for i in range(width):
        ai = a[i] if i < len(a) else 0.0
        bi = b[i] if i < len(b) else 0.0
        mi = max(0.0, margins[i] if i < len(margins) else margins[-1])
        if math.isinf(bi) and math.isfinite(ai):
            return True
        if abs(ai - bi) <= mi:
            continue
        return ai < bi - mi
    return False


@dataclass(frozen=True)
class SrsCandidate:
    """One redundancy state on a task-pose manifold."""

    rail_m: float
    psi_rad: float
    branch: int
    winding: int = 0
    q: tuple[float, ...] | np.ndarray | None = None
    score: Score | Sequence[float] | float = (0.0,)
    tag: str = ""
    metadata: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rail_m", float(self.rail_m))
        object.__setattr__(self, "psi_rad", float(self.psi_rad))
        object.__setattr__(self, "branch", int(self.branch))
        object.__setattr__(self, "winding", int(self.winding))
        object.__setattr__(self, "score", _score(self.score))
        if self.q is not None:
            object.__setattr__(self, "q", tuple(float(v) for v in np.asarray(self.q).reshape(-1)))

    @property
    def rail(self) -> float:
        return self.rail_m

    @property
    def psi(self) -> float:
        return self.psi_rad

    @property
    def unwrapped_psi_rad(self) -> float:
        return self.psi_rad + _TWO_PI * self.winding

    @property
    def discrete_key(self) -> tuple[int, int]:
        return self.branch, self.winding

    def key(self, rail_resolution: float = 1e-4, psi_resolution: float = 1e-3) -> tuple:
        return (
            round(self.rail_m / max(rail_resolution, 1e-12)),
            round(self.unwrapped_psi_rad / max(psi_resolution, 1e-12)),
            self.branch,
            self.winding,
        )


@dataclass(frozen=True)
class SrsPlanRequest:
    """Input for a current-manifold or layered-horizon search.

    With no ``horizon``, ``manifold`` is a single candidate layer at
    ``current_waypoint``.  With a horizon, a caller may supply matching
    ``candidate_layers`` or a constructor-level ``candidate_provider``.
    """

    current: SrsCandidate
    manifold: Sequence[SrsCandidate] = ()
    horizon: Sequence[Any] = ()
    candidate_layers: Sequence[Sequence[SrsCandidate]] = ()
    current_waypoint: Any = None
    contact: bool = False
    layer_dt_s: float | Sequence[float] = 1.0
    q_current: Sequence[float] | None = None
    qdot_current: Sequence[float] | None = None
    psi_velocity_current_rad_s: float = 0.0
    context: Any = None
    now_s: float | None = None


@dataclass(frozen=True)
class CandidateLayerContext:
    request: SrsPlanRequest
    waypoint: Any
    layer_index: int


@dataclass(frozen=True)
class NodeEvaluationContext:
    request: SrsPlanRequest
    candidate: SrsCandidate
    waypoint: Any
    layer_index: int
    is_edge_sample: bool = False
    edge_alpha: float = 1.0
    source_waypoint: Any = None
    target_waypoint: Any = None


@dataclass(frozen=True)
class NodeEvaluation:
    feasible: bool
    q: tuple[float, ...] | np.ndarray | None = None
    score: Score | Sequence[float] | float = (0.0,)
    collision_free: bool = True
    healthy: bool = True
    reason: str = ""
    metadata: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _score(self.score))
        if self.q is not None:
            object.__setattr__(self, "q", tuple(float(v) for v in np.asarray(self.q).reshape(-1)))

    @classmethod
    def accepted(
        cls, q: Sequence[float], score: float | Sequence[float] = (0.0,)
    ) -> "NodeEvaluation":
        return cls(True, q=q, score=score)

    @classmethod
    def rejected(cls, reason: str) -> "NodeEvaluation":
        return cls(False, reason=str(reason))


@dataclass(frozen=True)
class SafetySampleContext:
    node: NodeEvaluationContext
    q: tuple[float, ...]


@dataclass(frozen=True)
class EdgeSample:
    alpha: float
    time_s: float
    candidate: SrsCandidate
    q: tuple[float, ...]
    qdot: tuple[float, ...]
    qddot: tuple[float, ...]


@dataclass(frozen=True)
class EdgeEvaluationContext:
    request: SrsPlanRequest
    source: SrsCandidate
    target: SrsCandidate
    source_waypoint: Any
    target_waypoint: Any
    duration_s: float
    samples: tuple[EdgeSample, ...]
    discrete_switch: bool


@dataclass(frozen=True)
class EdgeEvaluation:
    feasible: bool
    score: Score | Sequence[float] | float = (0.0,)
    reason: str = ""
    metadata: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _score(self.score))


@dataclass(frozen=True)
class SrsRailPlan:
    """Committed near-term output.  It intentionally contains no full path."""

    corridor: tuple[SrsCandidate, ...]
    short_target: SrsCandidate
    score: Score
    generated_at: float
    commit_changed: bool = False
    proposal_count: int = 0
    contact_locked: bool = False
    layers_evaluated: int = 0
    reason: str = ""

    @property
    def target(self) -> SrsCandidate:
        return self.short_target


@dataclass(frozen=True)
class SrsPlannerConfig:
    edge_samples: int = 10
    default_edge_duration_s: float = 1.0
    rail_v_max_m_s: float = 0.08
    rail_a_max_m_s2: float = 0.30
    arm_v_max_rad_s: float = 1.0
    arm_a_max_rad_s2: float = 3.0
    psi_v_max_rad_s: float = 1.0
    psi_a_max_rad_s2: float = 3.0
    require_joint_samples: bool = True
    consistent_plans_to_commit: int = 3
    material_improvement: float | Sequence[float] = 1e-3
    min_dwell_s: float = 1.0
    corridor_nodes: int = 2
    rail_key_resolution_m: float = 1e-4
    psi_key_resolution_rad: float = 1e-3
    stale_after_s: float = 0.5
    fade_after_s: float = 0.5

    def __post_init__(self) -> None:
        if self.edge_samples < 2:
            raise ValueError("edge_samples must be >= 2")
        if self.default_edge_duration_s <= 0.0:
            raise ValueError("default_edge_duration_s must be positive")
        if self.consistent_plans_to_commit < 1:
            raise ValueError("consistent_plans_to_commit must be >= 1")
        if self.corridor_nodes < 1:
            raise ValueError("corridor_nodes must be >= 1")


class NoFeasiblePlanError(RuntimeError):
    pass


@dataclass
class _GraphNode:
    candidate: SrsCandidate
    evaluation: NodeEvaluation
    waypoint: Any


@dataclass
class _GraphState:
    node: _GraphNode
    score: Score
    path: tuple[_GraphNode, ...]
    terminal_q: np.ndarray
    terminal_qdot: np.ndarray
    terminal_psi_velocity: float


def _coerce_node(raw: Any, candidate: SrsCandidate) -> NodeEvaluation:
    if isinstance(raw, NodeEvaluation):
        return raw
    if raw is None:
        return NodeEvaluation.rejected("node evaluator returned None")
    if isinstance(raw, Mapping):
        return NodeEvaluation(
            feasible=bool(raw.get("feasible", raw.get("valid", True))),
            q=raw.get("q", candidate.q),
            score=raw.get("score", candidate.score),
            collision_free=bool(raw.get("collision_free", not raw.get("collision", False))),
            healthy=bool(raw.get("healthy", raw.get("health_ok", True))),
            reason=str(raw.get("reason", "")),
            metadata=raw.get("metadata"),
        )
    if isinstance(raw, (bool, np.bool_)):
        return NodeEvaluation(bool(raw), q=candidate.q, score=candidate.score)
    if isinstance(raw, tuple) and len(raw) == 2 and not np.isscalar(raw[1]):
        return NodeEvaluation(bool(raw[0]), q=raw[1], score=candidate.score)
    # A bare vector is a mapped joint state.
    arr = np.asarray(raw, dtype=float).reshape(-1)
    return NodeEvaluation(True, q=arr, score=candidate.score)


def _coerce_edge(raw: Any) -> EdgeEvaluation:
    if isinstance(raw, EdgeEvaluation):
        return raw
    if raw is None:
        return EdgeEvaluation(True)
    if isinstance(raw, Mapping):
        return EdgeEvaluation(
            feasible=bool(raw.get("feasible", raw.get("valid", True))),
            score=raw.get("score", (0.0,)),
            reason=str(raw.get("reason", "")),
            metadata=raw.get("metadata"),
        )
    if isinstance(raw, (bool, np.bool_)):
        return EdgeEvaluation(bool(raw))
    if np.isscalar(raw):
        return EdgeEvaluation(True, score=(float(raw),))
    return EdgeEvaluation(True, score=tuple(float(v) for v in raw))


class SrsRailPosturePlanner:
    """Synchronous graph core plus non-blocking ``submit/latest`` facade.

    Evaluator signatures use one immutable context argument.  A mapping may
    be supplied instead of a node evaluator and should return either a full
    8-vector or the seven arm joints (the rail entry is then prepended).
    """

    def __init__(
        self,
        *,
        candidate_provider: Callable[[CandidateLayerContext], Iterable[SrsCandidate]] | None = None,
        node_evaluator: Callable[[NodeEvaluationContext], NodeEvaluation | Any] | None = None,
        edge_evaluator: Callable[[EdgeEvaluationContext], EdgeEvaluation | Any] | None = None,
        srs_mapping: Callable[[NodeEvaluationContext], Sequence[float] | Any] | None = None,
        mapping: Callable[[NodeEvaluationContext], Sequence[float] | Any] | None = None,
        collision_evaluator: Callable[[SafetySampleContext], bool] | None = None,
        collision_checker: Callable[[SafetySampleContext], bool] | None = None,
        health_evaluator: Callable[[SafetySampleContext], bool] | None = None,
        health_checker: Callable[[SafetySampleContext], bool] | None = None,
        waypoint_interpolator: Callable[[Any, Any, float], Any] | None = None,
        config: SrsPlannerConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        autostart: bool = True,
    ) -> None:
        self.config = config or SrsPlannerConfig()
        self._candidate_provider = candidate_provider
        self._node_evaluator = node_evaluator
        self._edge_evaluator = edge_evaluator
        if srs_mapping is not None and mapping is not None:
            raise ValueError("provide either srs_mapping or mapping, not both")
        if collision_evaluator is not None and collision_checker is not None:
            raise ValueError(
                "provide either collision_evaluator or collision_checker, not both"
            )
        if health_evaluator is not None and health_checker is not None:
            raise ValueError("provide either health_evaluator or health_checker, not both")
        self._srs_mapping = srs_mapping if srs_mapping is not None else mapping
        self._collision_evaluator = (
            collision_evaluator if collision_evaluator is not None else collision_checker
        )
        self._health_evaluator = (
            health_evaluator if health_evaluator is not None else health_checker
        )
        self._waypoint_interpolator = waypoint_interpolator
        self._clock = clock
        self._state_lock = threading.Lock()
        self._committed: SrsRailPlan | None = None
        self._last_commit_at: float | None = None
        self._proposal_key: tuple | None = None
        self._proposal_count = 0
        self._async = PosturePlanner(
            self._calculate,
            stale_after_s=self.config.stale_after_s,
            fade_after_s=self.config.fade_after_s,
            clock=clock,
            name="srs-rail-posture-planner",
            autostart=autostart,
        )

    def submit(self, request: SrsPlanRequest, *, timestamp_s: float | None = None) -> int:
        return self._async.submit(request, timestamp_s=timestamp_s)

    def latest(self, *, now_s: float | None = None) -> PlannerSnapshot[SrsRailPlan]:
        return self._async.latest(now_s=now_s)

    def wait_for(self, sequence: int | None = None, timeout_s: float = 1.0) -> bool:
        return self._async.wait_for(sequence, timeout_s)

    def shutdown(self, **kwargs) -> bool:
        return self._async.shutdown(**kwargs)

    def reset_commit(self) -> None:
        with self._state_lock:
            self._committed = None
            self._last_commit_at = None
            self._proposal_key = None
            self._proposal_count = 0

    def plan_now(self, request: SrsPlanRequest) -> SrsRailPlan:
        computation = self._calculate(request)
        if not computation.valid or computation.value is None:
            raise NoFeasiblePlanError(computation.reason or "no feasible SRS+rail corridor")
        return computation.value

    plan = plan_now

    def _calculate(self, request: SrsPlanRequest) -> PlanComputation[SrsRailPlan]:
        try:
            proposed = self._search(request)
        except NoFeasiblePlanError as exc:
            return PlanComputation.invalid(str(exc))
        now = float(self._clock() if request.now_s is None else request.now_s)
        with self._state_lock:
            committed = self._apply_commit_hysteresis(request, proposed, now)
        return PlanComputation(committed)

    def _layers(self, request: SrsPlanRequest) -> tuple[tuple[SrsCandidate, ...], tuple[Any, ...]]:
        horizon = tuple(request.horizon)
        if request.candidate_layers:
            layers = tuple(tuple(layer) for layer in request.candidate_layers)
            if horizon and len(horizon) != len(layers):
                raise NoFeasiblePlanError("candidate_layers must match horizon length")
            waypoints = horizon if horizon else tuple(request.current_waypoint for _ in layers)
        elif horizon:
            if self._candidate_provider is None:
                raise NoFeasiblePlanError("horizon requires candidate_layers or candidate_provider")
            layers = tuple(
                tuple(self._candidate_provider(CandidateLayerContext(request, wp, i)))
                for i, wp in enumerate(horizon)
            )
            waypoints = horizon
        else:
            candidates = tuple(request.manifold)
            if not candidates and self._candidate_provider is not None:
                candidates = tuple(
                    self._candidate_provider(
                        CandidateLayerContext(request, request.current_waypoint, 0)
                    )
                )
            layers = (candidates,)
            waypoints = (request.current_waypoint,)
        if not layers or any(not layer for layer in layers):
            raise NoFeasiblePlanError("candidate graph contains an empty layer")
        return layers, waypoints

    def _search(self, request: SrsPlanRequest) -> SrsRailPlan:
        layers, waypoints = self._layers(request)
        current_for_eval = request.current
        if request.q_current is not None:
            q_current = self._normalize_q(request.q_current, request.current.rail_m)
            if q_current is None:
                raise NoFeasiblePlanError("q_current must be a finite 7- or 8-vector")
            current_for_eval = replace(request.current, q=tuple(q_current))
        current_eval = self._evaluate_node(
            NodeEvaluationContext(request, current_for_eval, request.current_waypoint, -1)
        )
        if not current_eval.feasible:
            raise NoFeasiblePlanError(f"current node invalid: {current_eval.reason}")
        current_q = self._normalize_q(current_eval.q, request.current.rail_m)
        if current_q is None:
            current_q = np.array([request.current.rail_m], dtype=float)
        qdot0 = np.zeros(current_q.size)
        if request.qdot_current is not None:
            qdot0 = np.asarray(request.qdot_current, dtype=float).reshape(-1)
        current_node = _GraphNode(request.current, current_eval, request.current_waypoint)
        states = [
            _GraphState(
                current_node,
                (0.0,),
                (),
                current_q,
                qdot0,
                float(request.psi_velocity_current_rad_s),
            )
        ]

        for layer_index, (raw_layer, waypoint) in enumerate(zip(layers, waypoints)):
            nodes: list[_GraphNode] = []
            for candidate in raw_layer:
                if request.contact and candidate.discrete_key != request.current.discrete_key:
                    continue
                ev = self._evaluate_node(
                    NodeEvaluationContext(request, candidate, waypoint, layer_index)
                )
                if ev.feasible:
                    nodes.append(_GraphNode(candidate, ev, waypoint))
            if not nodes:
                raise NoFeasiblePlanError(f"no feasible nodes in layer {layer_index}")

            next_states: list[_GraphState] = []
            duration = self._duration(request.layer_dt_s, layer_index)
            for target in nodes:
                best: _GraphState | None = None
                for source_state in states:
                    edge = self._evaluate_edge(
                        request,
                        source_state,
                        target,
                        duration,
                        layer_index,
                    )
                    if edge is None:
                        continue
                    (
                        edge_eval,
                        terminal_q,
                        terminal_velocity,
                        terminal_psi_velocity,
                    ) = edge
                    total = _score_add(source_state.score, target.evaluation.score, edge_eval.score)
                    state = _GraphState(
                        target,
                        total,
                        source_state.path + (target,),
                        terminal_q,
                        terminal_velocity,
                        terminal_psi_velocity,
                    )
                    if best is None or (state.score, self._path_key(state.path)) < (
                        best.score,
                        self._path_key(best.path),
                    ):
                        best = state
                if best is not None:
                    next_states.append(best)
            if not next_states:
                raise NoFeasiblePlanError(f"no fully validated edges into layer {layer_index}")
            states = next_states

        winner = min(states, key=lambda s: (s.score, self._path_key(s.path)))
        corridor_nodes = winner.path[: self.config.corridor_nodes]
        if not corridor_nodes:
            corridor = (request.current,)
        else:
            corridor = tuple(n.candidate for n in corridor_nodes)
        now = float(self._clock() if request.now_s is None else request.now_s)
        return SrsRailPlan(
            corridor=corridor,
            short_target=corridor[-1],
            score=winner.score,
            generated_at=now,
            contact_locked=bool(request.contact),
            layers_evaluated=len(layers),
            reason="validated candidate graph",
        )

    def _duration(self, value: float | Sequence[float], layer: int) -> float:
        if np.isscalar(value):
            out = float(value)
        else:
            seq = tuple(float(v) for v in value)
            out = seq[min(layer, len(seq) - 1)] if seq else self.config.default_edge_duration_s
        if out <= 0.0:
            raise NoFeasiblePlanError("edge duration must be positive")
        return out

    def _evaluate_node(self, context: NodeEvaluationContext) -> NodeEvaluation:
        candidate = context.candidate
        if self._node_evaluator is not None:
            try:
                ev = _coerce_node(self._node_evaluator(context), candidate)
            except Exception as exc:
                return NodeEvaluation.rejected(f"node evaluator: {type(exc).__name__}: {exc}")
            if ev.feasible and ev.q is None and self._srs_mapping is not None:
                try:
                    mapped = _coerce_node(self._srs_mapping(context), candidate)
                except Exception as exc:
                    return NodeEvaluation.rejected(
                        f"SRS mapping: {type(exc).__name__}: {exc}"
                    )
                if not mapped.feasible:
                    return NodeEvaluation.rejected(mapped.reason or "SRS mapping rejected node")
                ev = replace(
                    ev,
                    q=mapped.q,
                    collision_free=ev.collision_free and mapped.collision_free,
                    healthy=ev.healthy and mapped.healthy,
                )
        elif candidate.q is not None:
            ev = NodeEvaluation(True, candidate.q, candidate.score)
        elif self._srs_mapping is not None:
            try:
                ev = _coerce_node(self._srs_mapping(context), candidate)
            except Exception as exc:
                return NodeEvaluation.rejected(f"SRS mapping: {type(exc).__name__}: {exc}")
        else:
            return NodeEvaluation.rejected("no node evaluator, candidate q, or SRS mapping")

        if not ev.feasible or not ev.collision_free or not ev.healthy:
            return replace(ev, feasible=False, reason=ev.reason or "node safety rejected")
        if any(not math.isfinite(v) for v in ev.score):
            return NodeEvaluation.rejected("non-finite node score")
        q = self._normalize_q(ev.q, candidate.rail_m)
        if q is None and self.config.require_joint_samples:
            return NodeEvaluation.rejected("node did not provide a complete joint sample")
        if q is not None:
            ev = replace(ev, q=tuple(q))
            sample = SafetySampleContext(context, tuple(q))
            try:
                if self._collision_evaluator is not None and not bool(
                    self._collision_evaluator(sample)
                ):
                    return NodeEvaluation.rejected("collision evaluator rejected node")
                if self._health_evaluator is not None and not bool(self._health_evaluator(sample)):
                    return NodeEvaluation.rejected("health evaluator rejected node")
            except Exception as exc:
                return NodeEvaluation.rejected(f"safety evaluator: {type(exc).__name__}: {exc}")
        return ev

    @staticmethod
    def _normalize_q(q: Sequence[float] | None, rail_m: float) -> np.ndarray | None:
        if q is None:
            return None
        arr = np.asarray(q, dtype=float).reshape(-1)
        if arr.size == 7:
            arr = np.concatenate(([float(rail_m)], arr))
        if arr.size != 8 or not np.all(np.isfinite(arr)):
            return None
        return arr

    def _waypoint_at(self, source: Any, target: Any, alpha: float) -> Any:
        if self._waypoint_interpolator is not None:
            return self._waypoint_interpolator(source, target, float(alpha))
        if source is None:
            return target
        if target is None:
            return source
        try:
            a = np.asarray(source, dtype=float)
            b = np.asarray(target, dtype=float)
            if a.shape == b.shape and a.size:
                return (1.0 - alpha) * a + alpha * b
        except (TypeError, ValueError):
            pass
        return target if alpha >= 1.0 else source

    def _evaluate_edge(
        self,
        request: SrsPlanRequest,
        source_state: _GraphState,
        target: _GraphNode,
        duration_s: float,
        layer_index: int,
    ) -> tuple[EdgeEvaluation, np.ndarray, np.ndarray, float] | None:
        source = source_state.node
        if request.contact and source.candidate.discrete_key != target.candidate.discrete_key:
            return None
        q_prev = np.asarray(source_state.terminal_q, dtype=float).reshape(-1).copy()
        if q_prev is None and self.config.require_joint_samples:
            return None
        qdot_prev = np.asarray(source_state.terminal_qdot, dtype=float).reshape(-1)
        if q_prev is not None and qdot_prev.size != q_prev.size:
            qdot_prev = np.zeros(q_prev.size)

        n = self.config.edge_samples
        sample_dt = duration_s / n
        psi0 = source.candidate.unwrapped_psi_rad
        psi1 = target.candidate.unwrapped_psi_rad
        psi_v_prev = float(source_state.terminal_psi_velocity)
        samples: list[EdgeSample] = []
        for i in range(1, n + 1):
            u = i / n
            # Quintic time law gives a finite acceleration edge with zero
            # endpoint velocity; linear interpolation would falsely require
            # an instantaneous velocity jump at every graph node.
            alpha = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            rail = source.candidate.rail_m + alpha * (
                target.candidate.rail_m - source.candidate.rail_m
            )
            psi_unwrapped = psi0 + alpha * (psi1 - psi0)
            sample_candidate = SrsCandidate(
                rail_m=rail,
                psi_rad=psi_unwrapped - _TWO_PI * target.candidate.winding,
                branch=target.candidate.branch,
                winding=target.candidate.winding,
                score=(0.0,),
                tag=target.candidate.tag,
                metadata=target.candidate.metadata,
            )
            waypoint = self._waypoint_at(source.waypoint, target.waypoint, alpha)
            node_context = NodeEvaluationContext(
                request,
                sample_candidate,
                waypoint,
                layer_index,
                True,
                alpha,
                source.waypoint,
                target.waypoint,
            )
            ev = self._evaluate_node(node_context)
            if not ev.feasible:
                return None
            q = self._normalize_q(ev.q, rail)
            if q is None:
                if self.config.require_joint_samples:
                    return None
                q = np.array([rail], dtype=float)
            if q_prev is not None and q.size == q_prev.size:
                q = self._unwrap_q_near(q_prev, q)
                qdot = (q - q_prev) / sample_dt
                qddot = (qdot - qdot_prev) / sample_dt
                if not self._within_joint_limits(qdot, qddot):
                    return None
            else:
                qdot = np.zeros(q.size)
                qddot = np.zeros(q.size)

            psi_v = (psi_unwrapped - (psi0 if i == 1 else samples[-1].candidate.unwrapped_psi_rad)) / sample_dt
            psi_a = (psi_v - psi_v_prev) / sample_dt
            if (
                abs(psi_v) > self.config.psi_v_max_rad_s + 1e-9
                or abs(psi_a) > self.config.psi_a_max_rad_s2 + 1e-9
            ):
                return None
            samples.append(
                EdgeSample(
                    alpha=alpha,
                    time_s=i * sample_dt,
                    candidate=sample_candidate,
                    q=tuple(float(v) for v in q),
                    qdot=tuple(float(v) for v in qdot),
                    qddot=tuple(float(v) for v in qddot),
                )
            )
            q_prev, qdot_prev, psi_v_prev = q, qdot, psi_v

        context = EdgeEvaluationContext(
            request=request,
            source=source.candidate,
            target=target.candidate,
            source_waypoint=source.waypoint,
            target_waypoint=target.waypoint,
            duration_s=duration_s,
            samples=tuple(samples),
            discrete_switch=source.candidate.discrete_key != target.candidate.discrete_key,
        )
        try:
            custom = _coerce_edge(self._edge_evaluator(context) if self._edge_evaluator else None)
        except Exception:
            return None
        if not custom.feasible or any(not math.isfinite(v) for v in custom.score):
            return None
        assert q_prev is not None
        return custom, q_prev, qdot_prev, psi_v_prev

    def _within_joint_limits(self, velocity: np.ndarray, acceleration: np.ndarray) -> bool:
        if velocity.size == 1:
            return bool(
                abs(velocity[0]) <= self.config.rail_v_max_m_s + 1e-9
                and abs(acceleration[0]) <= self.config.rail_a_max_m_s2 + 1e-9
            )
        return bool(
            abs(velocity[0]) <= self.config.rail_v_max_m_s + 1e-9
            and abs(acceleration[0]) <= self.config.rail_a_max_m_s2 + 1e-9
            and np.all(np.abs(velocity[1:]) <= self.config.arm_v_max_rad_s + 1e-9)
            and np.all(np.abs(acceleration[1:]) <= self.config.arm_a_max_rad_s2 + 1e-9)
        )

    @staticmethod
    def _unwrap_q_near(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        out = np.asarray(current, dtype=float).copy()
        if out.size > 1:
            delta = (out[1:] - previous[1:] + math.pi) % _TWO_PI - math.pi
            out[1:] = previous[1:] + delta
        return out

    def _path_key(self, path: Sequence[_GraphNode]) -> tuple:
        return tuple(
            node.candidate.key(
                self.config.rail_key_resolution_m,
                self.config.psi_key_resolution_rad,
            )
            for node in path
        )

    def _apply_commit_hysteresis(
        self, request: SrsPlanRequest, proposed: SrsRailPlan, now: float
    ) -> SrsRailPlan:
        current_hold = SrsRailPlan(
            corridor=(request.current,),
            short_target=request.current,
            score=(math.inf,),
            generated_at=now,
            contact_locked=bool(request.contact),
            layers_evaluated=proposed.layers_evaluated,
            reason="holding current posture pending stable improvement",
        )
        if self._committed is None:
            self._committed = current_hold
            self._last_commit_at = now

        # Contact is a hard topology lock, not a preference subject to dwell.
        # A corridor committed while free-space planning was active must never
        # leak a different branch/winding into a newly-contacting request.
        if (
            request.contact
            and self._committed is not None
            and self._committed.short_target.discrete_key
            != request.current.discrete_key
        ):
            self._committed = current_hold
            self._last_commit_at = now
            self._proposal_key = None
            self._proposal_count = 0

        assert self._committed is not None
        key = proposed.short_target.key(
            self.config.rail_key_resolution_m,
            self.config.psi_key_resolution_rad,
        )
        committed_key = self._committed.short_target.key(
            self.config.rail_key_resolution_m,
            self.config.psi_key_resolution_rad,
        )
        if key == committed_key:
            self._proposal_key = None
            self._proposal_count = 0
            self._committed = replace(
                proposed,
                commit_changed=False,
                proposal_count=0,
                reason="refreshed committed short corridor",
            )
            return self._committed

        better = lexicographic_materially_better(
            proposed.score,
            self._committed.score,
            self.config.material_improvement,
        )
        if not better:
            self._proposal_key = None
            self._proposal_count = 0
            return replace(self._committed, generated_at=now, proposal_count=0)

        if key == self._proposal_key:
            self._proposal_count += 1
        else:
            self._proposal_key = key
            self._proposal_count = 1
        dwell_ok = (
            self._last_commit_at is None
            or now - self._last_commit_at >= self.config.min_dwell_s
        )
        if self._proposal_count >= self.config.consistent_plans_to_commit and dwell_ok:
            self._committed = replace(
                proposed,
                commit_changed=True,
                proposal_count=self._proposal_count,
                reason="committed after consistent, material improvement",
            )
            self._last_commit_at = now
            self._proposal_key = None
            self._proposal_count = 0
            return self._committed
        return replace(
            self._committed,
            generated_at=now,
            proposal_count=self._proposal_count,
            reason="holding committed corridor during consistency/dwell gate",
        )


# Short aliases for integrations which already use generic planner vocabulary.
Candidate = SrsCandidate
PlanRequest = SrsPlanRequest
PostureCorridor = SrsRailPlan


__all__ = [
    "Candidate",
    "CandidateLayerContext",
    "EdgeEvaluation",
    "EdgeEvaluationContext",
    "EdgeSample",
    "NoFeasiblePlanError",
    "NodeEvaluation",
    "NodeEvaluationContext",
    "PlanRequest",
    "PostureCorridor",
    "SafetySampleContext",
    "Score",
    "SrsCandidate",
    "SrsPlanRequest",
    "SrsPlannerConfig",
    "SrsRailPlan",
    "SrsRailPosturePlanner",
    "lexicographic_materially_better",
]
