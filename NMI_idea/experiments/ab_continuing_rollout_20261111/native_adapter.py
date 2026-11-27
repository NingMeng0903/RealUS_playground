"""Small native closed-loop API for the adaptive-stopping experiment.

The public contract is intentionally narrower than :mod:`fixture` and the
controller replay implementation.  A planner can enumerate ``actions``, ask
``checked(state, action_id)`` for one checked continuation, and then call
``execute(state, action_id)`` after it has observed/selected the next action.
Every continuation receives the complete ``q/qdot/acceleration/snapshot``
from the preceding native event.  No deadline or pair-level atomicity is
implemented here.

Planning and execution have separate trajectory directories.  By default an
execution request replays the native controller, which is the strongest
closed-loop check.  ``reuse_checked=True`` is available for deterministic
offline sweeps: it reuses a disk checkpoint only after validating the complete
input state and writes a pointer into the execution directory.  A checkpoint
never contains only q; the replay snapshot and qdot/acceleration are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from fixture import (
    Capture,
    CaptureResult,
    EpisodeState,
    MotionRollouts,
    MotionState,
    SINGLE_DWELLS,
    TERMINALS,
    make_rig,
    single_dwell_action,
)
from robot_motion import NativeMotionEnvironment


PROBE_BASES = ("side_left", "side_right", "fork_minus", "fork_plus")
ACTION_IDS = tuple(
    single_dwell_action(base, dwell)
    for base in PROBE_BASES
    for dwell in SINGLE_DWELLS
) + TERMINALS


@dataclass(frozen=True)
class PredictedEpisodeState:
    """A public mechanical forecast, never an actual native continuation.

    ``q_proxy`` is the declared robot-coordinate proxy used only by the
    public dIRD evaluator.  It is deliberately not exposed as ``q`` or
    ``qdot`` and ``native`` is always ``None``.  A caller may feed this
    representation into another :meth:`NativeAdapter.propose` call, but it
    must call :meth:`checked`/``execute`` on a measured actual state before
    any physical action.
    """

    key: tuple
    elapsed: float
    pose: np.ndarray
    origin_state_fingerprint: str
    q_proxy: np.ndarray
    valid: bool = True
    predicted: bool = True
    native: None = None
    status: str = "predicted_unknown"

    @property
    def actual(self) -> bool:
        return False

    @property
    def state_kind(self) -> str:
        return "predicted_unknown"

    def as_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "elapsed_s": float(self.elapsed),
            "pose": self.pose,
            "origin_state_fingerprint": self.origin_state_fingerprint,
            "q_proxy": self.q_proxy,
            "valid": bool(self.valid),
            "predicted": True,
            "native": None,
            "status": self.status,
        }
        return _json_safe(payload) if json_safe else payload


@dataclass(frozen=True)
class MotionRoute(Mapping[str, Any]):
    """One finite public route candidate from a supplied state."""

    route_id: str
    waypoints: tuple[tuple[float, float, float], ...]
    goal_pose: np.ndarray
    travel_s: float
    dwell_s: float
    duration_s: float
    margin: float
    geometric_status: str = "clear"
    prediction_status: str = "predicted_unknown"
    native_checked: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, {"id": "route_id", "predicted": "prediction_status"}.get(key, key))

    def __iter__(self) -> Iterator[str]:
        return iter(("route_id", "waypoints", "goal_pose", "travel_s", "dwell_s", "duration_s",
                     "margin", "geometric_status", "prediction_status", "native_checked"))

    def __len__(self) -> int:
        return 10

    def as_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        payload = {key: self[key] for key in self}
        return _json_safe(payload) if json_safe else payload


@dataclass(frozen=True)
class MotionProposal(Mapping[str, Any]):
    """Public route/dIRD proposal with an explicit unchecked boundary."""

    action_id: str
    action: dict[str, Any]
    available: bool
    status: str
    routes: tuple[MotionRoute, ...]
    best_route: MotionRoute | None
    predicted_output: PredictedEpisodeState | None
    goal_pose: np.ndarray | None
    travel_s: float | None
    dwell_s: float
    duration_s: float | None
    margin: float | None
    state_fingerprint: str
    cache_hit: bool
    dird_query_count: int
    prediction_kind: str = "public_measured_frame_route_dird"
    native_checked: bool = False
    success_probability: None = None
    dird_query_increment: int = 0

    @property
    def feasible(self) -> bool:
        return self.available

    @property
    def output(self) -> PredictedEpisodeState | None:
        return self.predicted_output

    @property
    def candidates(self) -> tuple[MotionRoute, ...]:
        return self.routes

    @property
    def geometric_status(self) -> str:
        return "clear" if self.routes else "no_public_route"

    @property
    def predicted(self) -> bool:
        return True

    def __getitem__(self, key: str) -> Any:
        aliases = {"route_candidates": "routes", "candidates": "routes", "best": "best_route",
                   "output": "predicted_output", "elapsed": "duration_s", "dird_margin": "margin",
                   "unknown": "status", "geometric": "geometric_status"}
        return getattr(self, aliases.get(key, key))

    def __iter__(self) -> Iterator[str]:
        return iter(("action_id", "action", "available", "status", "routes", "best_route",
                     "predicted_output", "goal_pose", "travel_s", "dwell_s", "duration_s",
                     "margin", "state_fingerprint", "cache_hit", "dird_query_count",
                     "prediction_kind", "native_checked", "success_probability",
                     "dird_query_increment", "geometric_status"))

    def __len__(self) -> int:
        return 20

    def as_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        payload = {key: self[key] for key in self}
        return _json_safe(payload) if json_safe else payload


def _json_safe(value: Any) -> Any:
    """Convert replay state and numpy diagnostics to strict JSON values."""
    if isinstance(value, EpisodeState):
        return _state_payload(value)
    if isinstance(value, (PredictedEpisodeState, MotionRoute, MotionProposal)):
        return value.as_dict(json_safe=True)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _state_payload(state: EpisodeState) -> dict[str, Any]:
    native = state.native
    payload = {
        "key": list(state.key),
        "elapsed_s": float(state.elapsed),
        "pose": _json_safe(np.asarray(state.pose, dtype=float)),
        "valid": bool(state.valid),
        "native": None if native is None else {
            "q": _json_safe(native.q),
            "qdot": _json_safe(native.qdot),
            "acceleration": _json_safe(native.acceleration),
            "snapshot": _json_safe(native.snapshot),
        },
    }
    if bool(getattr(state, "predicted", False)):
        payload.update({
            "predicted": True,
            "status": str(getattr(state, "status", "predicted_unknown")),
            "origin_state_fingerprint": str(getattr(state, "origin_state_fingerprint", "")),
            "q_proxy": _json_safe(np.asarray(getattr(state, "q_proxy"), dtype=float)),
            "native": None,
        })
    return payload


def _state_from_payload(payload: Mapping[str, Any]) -> EpisodeState:
    native_payload = payload.get("native")
    native = None
    if native_payload is not None:
        native = MotionState(
            np.asarray(native_payload["q"], dtype=float),
            np.asarray(native_payload["qdot"], dtype=float),
            np.asarray(native_payload["acceleration"], dtype=float),
            dict(native_payload["snapshot"]),
        )
    return EpisodeState(
        tuple(payload.get("key", ())),
        float(payload["elapsed_s"]),
        np.asarray(payload["pose"], dtype=float),
        native,
        bool(payload.get("valid", True)),
    )


def _state_fingerprint(state: EpisodeState) -> str:
    encoded = json.dumps(_json_safe(_state_payload(state)), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdapterEvent(Mapping[str, Any]):
    """One checked or executed adaptive event.

    Attribute and mapping access are both supported so callers can use either
    ``event.output`` or ``event["output"]``.  ``output`` is an
    :class:`EpisodeState`, not a nominal reset state.
    """

    action_id: str
    action: dict[str, Any]
    feasible: bool
    output: EpisodeState
    actual_output: dict[str, Any]
    elapsed_s: float
    duration_s: float
    margin: float | None
    route_id: str | None
    call_kind: str
    source: str
    verification_calls: int
    execution_calls: int
    verification_count: int
    execution_count: int
    trajectory_records: tuple[str, ...]
    result: dict[str, Any]

    @property
    def elapsed(self) -> float:
        return self.elapsed_s

    @property
    def actual(self) -> dict[str, Any]:
        return self.actual_output

    @property
    def actual_action(self) -> dict[str, Any]:
        """Action metadata paired with the measured output."""
        return self.action

    @property
    def state(self) -> EpisodeState:
        return self.output

    def __getitem__(self, key: str) -> Any:
        aliases = {
            "actual": "actual_output",
            "actual_action": "action",
            "elapsed": "elapsed_s",
            "state": "output",
        }
        return getattr(self, aliases.get(key, key))

    def __iter__(self) -> Iterator[str]:
        return iter((
            "action_id", "action", "feasible", "output", "actual_output",
            "elapsed_s", "duration_s", "margin", "route_id", "call_kind",
            "source", "verification_calls", "execution_calls",
            "verification_count", "execution_count", "trajectory_records", "result",
        ))

    def __len__(self) -> int:
        return 17

    def as_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        value = {key: self[key] for key in self}
        if json_safe:
            return _json_safe(value)
        return value


class NativeAdapter:
    """Adaptive action adapter backed by deterministic native replay.

    Parameters
    ----------
    index, mode:
        The public fixture index and ``"open"``/``"constrained"`` mode.
    output:
        Optional run directory.  If provided, planning traces/checkpoints are
        written under ``planning/`` and execution traces under ``execution/``.
    backend:
        ``"native"`` is the default; ``"python"`` is an explicit development
        fallback using the same replay contract.
    reuse_checked:
        If true, :meth:`execute` may reuse a validated complete-state planning
        checkpoint.  The default false performs a fresh native replay.
    """

    def __init__(self, index: int, mode: str, output: str | Path | None,
                 *, backend: str = "native", reuse_checked: bool = False,
                 native: NativeMotionEnvironment | None = None):
        if mode not in ("open", "constrained"):
            raise ValueError("mode must be 'open' or 'constrained'")
        self.index = int(index)
        self.mode = mode
        self.backend = str(backend)
        self.reuse_checked = bool(reuse_checked)
        self.output = None if output is None else Path(output)
        self.planning_output = None if self.output is None else self.output / "planning"
        self.execution_output = None if self.output is None else self.output / "execution"
        self.checkpoint_output = None if self.output is None else self.output / "planning" / "checkpoints"
        for path in (self.planning_output, self.execution_output, self.checkpoint_output):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

        self.native = native or NativeMotionEnvironment(backend=self.backend)
        self.rig = make_rig(self.index, self.mode, self.native)
        self._planning = MotionRollouts(self.rig, self.native, self.planning_output)
        self._execution = MotionRollouts(self.rig, self.native, self.execution_output)
        self._checked_events: dict[tuple[str, str], AdapterEvent] = {}
        self.verification_calls = 0
        self.execution_calls = 0
        self.verification_count = 0
        self.execution_count = 0
        self._proposal_cache: dict[tuple[str, str, str], MotionProposal] = {}
        self.proposal_cache_hits = 0
        self.proposal_calls = 0
        self.proposal_dird_queries = 0
        self._proposal_source_version: str | None = None
        self._closed = False

    @property
    def start(self) -> EpisodeState:
        return self.rig.start

    @property
    def actions(self) -> tuple[str, ...]:
        return ACTION_IDS

    @property
    def measured_frame(self) -> np.ndarray:
        return self.rig.measured.copy()

    @property
    def physical_frame(self) -> np.ndarray:
        """Private physical frame for renderer construction only.

        It is intentionally not placed in :class:`AdapterEvent` or public
        metadata.  Motion planning/execution always receives ``measured_frame``.
        """
        return self.rig.actual_surface.copy()

    @property
    def address(self) -> dict[str, Any]:
        return dict(self.rig.address)

    @property
    def motion(self) -> "NativeAdapter":
        """Motion proposal facade used by the adaptive planner contract."""
        return self

    def _canonical(self, action_id: str) -> str:
        canonical = self.rig.resolve_action_id(action_id)
        if canonical not in ACTION_IDS:
            raise KeyError(f"action {action_id!r} is not in the adaptive action set")
        return canonical

    @staticmethod
    def _require_actual_state(state: Any) -> None:
        if bool(getattr(state, "predicted", False)) or getattr(state, "native", None) is None and not isinstance(state, EpisodeState):
            raise TypeError("native checked/execute requires a measured actual EpisodeState; use propose for predicted states")

    def _action_payload(self, action_id: str) -> dict[str, Any]:
        capture = self.rig.captures[action_id]
        return {
            "id": action_id,
            "base": capture.base_name,
            "kind": capture.kind,
            "window": capture.window,
            "goal_local": list(map(float, capture.goal)),
            "dwell_s": float(capture.dwell_s),
            "interruptible": capture.kind != "terminal",
        }

    def _proposal_version(self) -> str:
        if self._proposal_source_version is None:
            self._proposal_source_version = str(self._adapter_fingerprint()["sha256"])
        return self._proposal_source_version

    @staticmethod
    def _proposal_q_proxy(state: Any, root_q: np.ndarray) -> np.ndarray:
        proxy = getattr(state, "q_proxy", None)
        if proxy is not None:
            arr = np.asarray(proxy, dtype=float).reshape(-1)
        elif getattr(state, "native", None) is not None:
            arr = np.asarray(state.native.q, dtype=float).reshape(-1)
        else:
            arr = np.asarray(root_q, dtype=float).reshape(-1)
        if arr.shape != (8,) or not np.isfinite(arr).all():
            raise ValueError("proposal q proxy must be finite with shape (8,)")
        return arr.copy()

    def propose(self, state: Any, action_id: str) -> MotionProposal:
        """Predict public route/dIRD consequences without running native.

        The finite route list comes directly from
        ``MotionRollouts.candidates``.  It performs only measured-frame
        geometry, the existing route-duration model and one batched baked-dIRD
        score.  A returned route is ``predicted_unknown`` rather than a native
        feasibility certificate; selected actions still require ``checked``
        and ``execute`` from the measured actual state.
        """
        canonical = self._canonical(action_id)
        self.proposal_calls += 1
        state_fp = _state_fingerprint(state)
        cache_key = (self._proposal_version(), state_fp, canonical)
        cached = self._proposal_cache.get(cache_key)
        if cached is not None:
            self.proposal_cache_hits += 1
            return MotionProposal(**{**cached.__dict__, "cache_hit": True,
                                     "dird_query_increment": 0})

        capture = self.rig.captures[canonical]
        before_queries = self._planning.dird_queries
        candidates = self._planning.candidates(state, capture)
        query_increment = self._planning.dird_queries - before_queries
        self.proposal_dird_queries += query_increment
        goal_pose = self.native.measured_goal_pose(self.rig.measured, capture.goal)
        routes: list[MotionRoute] = []
        for route_name, waypoints, travel_s, margin in candidates:
            route_goal = self.native.measured_goal_pose(self.rig.measured, capture.goal)
            routes.append(MotionRoute(
                route_id=str(route_name),
                waypoints=tuple(tuple(float(x) for x in point) for point in waypoints),
                goal_pose=route_goal,
                travel_s=float(travel_s),
                dwell_s=float(capture.dwell_s),
                duration_s=float(travel_s + capture.dwell_s),
                margin=float(margin),
            ))
        best = routes[0] if routes else None
        root_q = self.native.q_seed
        q_proxy = self._proposal_q_proxy(state, root_q)
        origin_fingerprint = (str(getattr(state, "origin_state_fingerprint", ""))
                              or state_fp)
        predicted_output = None if best is None else PredictedEpisodeState(
            key=tuple(state.key) + (canonical,),
            elapsed=float(state.elapsed + best.duration_s),
            pose=np.asarray(goal_pose, dtype=float).copy(),
            origin_state_fingerprint=origin_fingerprint,
            q_proxy=q_proxy,
        )
        proposal = MotionProposal(
            action_id=canonical,
            action=self._action_payload(canonical),
            available=bool(routes),
            status="predicted_unknown" if routes else "unavailable_no_public_route",
            routes=tuple(routes),
            best_route=best,
            predicted_output=predicted_output,
            goal_pose=np.asarray(goal_pose, dtype=float).copy(),
            travel_s=None if best is None else float(best.travel_s),
            dwell_s=float(capture.dwell_s),
            duration_s=None if best is None else float(best.duration_s),
            margin=None if best is None else float(best.margin),
            state_fingerprint=state_fp,
            cache_hit=False,
            dird_query_count=int(self._planning.dird_queries),
            dird_query_increment=int(query_increment),
        )
        self._proposal_cache[cache_key] = proposal
        if self.planning_output is not None:
            proposal_dir = self.planning_output / "proposals"
            proposal_dir.mkdir(parents=True, exist_ok=True)
            digest = sha256(f"{state_fp}:{canonical}".encode("utf-8")).hexdigest()[:24]
            (proposal_dir / f"{digest}.json").write_text(
                json.dumps(proposal.as_dict(json_safe=True), indent=2, allow_nan=False),
                encoding="utf-8")
        return proposal

    @staticmethod
    def _route_row(rollouts: MotionRollouts, route_id: str | None) -> dict[str, Any]:
        if route_id is None:
            return {}
        for row in reversed(rollouts.rows):
            if row.get("route") == route_id:
                return row
        return {}

    def _event_from_capture(self, state: EpisodeState, action_id: str,
                            captured: CaptureResult, *, call_kind: str,
                            source: str, rollouts: MotionRollouts) -> AdapterEvent:
        transfer = captured.transfer
        output = transfer.output
        row = self._route_row(rollouts, transfer.route_id)
        steps = tuple(step.get("record") for step in row.get("steps", ()) if step.get("record"))
        step_rows = tuple(row.get("steps", ()))
        reference_kind = next((step.get("reference_kind", step.get("reference"))
                               for step in step_rows
                               if step.get("reference_kind", step.get("reference"))), None)
        same_tick_position = max((float(step["same_tick_position_error_max_m"])
                                   for step in step_rows
                                   if step.get("same_tick_position_error_max_m") is not None), default=None)
        same_tick_rotation = max((float(step["same_tick_rotation_error_max_rad"])
                                   for step in step_rows
                                   if step.get("same_tick_rotation_error_max_rad") is not None), default=None)
        jerk_margin = min((float(step["configured_jerk_margin_min"])
                           for step in step_rows
                           if step.get("configured_jerk_margin_min") is not None), default=None)
        native = output.native
        actual_output = {
            "q": None if native is None else native.q.copy(),
            "qdot": None if native is None else native.qdot.copy(),
            "acceleration": None if native is None else native.acceleration.copy(),
            "snapshot": None if native is None else native.snapshot,
            "pose": output.pose.copy(),
            "elapsed_s": float(output.elapsed),
            "valid": bool(output.valid),
            "route_id": transfer.route_id,
            "reference_kind": reference_kind,
            "entry_q": None if not row.get("steps") else np.asarray(row["steps"][0].get("start_q"), dtype=float),
            "entry_qdot": None if not row.get("steps") else np.asarray(row["steps"][0].get("start_qdot"), dtype=float),
            "entry_snapshot": None if not row.get("steps") else row["steps"][0].get("entry_snapshot"),
            "same_tick_position_error_max_m": same_tick_position,
            "same_tick_rotation_error_max_rad": same_tick_rotation,
            "jerk_margin_min": jerk_margin,
        }
        return AdapterEvent(
            action_id=action_id,
            action=self._action_payload(action_id),
            feasible=bool(output.valid),
            output=output,
            actual_output=actual_output,
            elapsed_s=float(output.elapsed),
            duration_s=float(transfer.duration),
            margin=None if row.get("dird_mean_calibrated_margin") is None else float(row["dird_mean_calibrated_margin"]),
            route_id=transfer.route_id,
            call_kind=call_kind,
            source=source,
            verification_calls=self.verification_calls,
            execution_calls=self.execution_calls,
            verification_count=self.verification_count,
            execution_count=self.execution_count,
            trajectory_records=steps,
            result={
                "reason": transfer.reason,
                "route": row,
                "history_continuation": native is not None,
                "full_state_checkpoint": native is None or native.snapshot is not None,
                "reference_kind": reference_kind,
                "entry_q": actual_output["entry_q"],
                "entry_qdot": actual_output["entry_qdot"],
                "entry_snapshot": actual_output["entry_snapshot"],
            },
        )

    def _checkpoint_path(self, state: EpisodeState, action_id: str) -> Path | None:
        if self.checkpoint_output is None:
            return None
        digest = sha256(f"{_state_fingerprint(state)}:{action_id}".encode("utf-8")).hexdigest()[:24]
        return self.checkpoint_output / f"{digest}.json"

    def _adapter_fingerprint(self) -> dict[str, Any]:
        """Source/controller/fixture identity required for checkpoint reuse."""
        replay = getattr(self.native, "replay", None)
        source = replay._state_fingerprint() if replay is not None and hasattr(replay, "_state_fingerprint") else {}
        rig = {
            "name": self.rig.name,
            "address": self.rig.address,
            "measured": _json_safe(self.rig.measured),
            "wall": _json_safe(self.rig.wall),
            "main_side": self.rig.main_side,
        }
        payload = {
            "index": self.index,
            "mode": self.mode,
            "backend": self.backend,
            "adapter_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "motion_source_sha256": sha256(Path(__file__).with_name("robot_motion.py").read_bytes()).hexdigest(),
            "fixture_source_sha256": sha256(Path(__file__).with_name("fixture.py").read_bytes()).hexdigest(),
            "controller": _json_safe(source),
            "rig": rig,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["sha256"] = sha256(encoded.encode("utf-8")).hexdigest()
        return payload

    def _save_checkpoint(self, state: EpisodeState, event: AdapterEvent) -> None:
        path = self._checkpoint_path(state, event.action_id)
        if path is None:
            return
        payload = {
            "schema": "adaptive_native_checkpoint_v1",
            "index": self.index,
            "mode": self.mode,
            "action_id": event.action_id,
            "adapter_fingerprint": self._adapter_fingerprint(),
            "input_fingerprint": _state_fingerprint(state),
            "input_state": _state_payload(state),
            "output_state": _state_payload(event.output),
            "event": event.as_dict(json_safe=True),
        }
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    def _load_checkpoint(self, state: EpisodeState, action_id: str) -> AdapterEvent | None:
        path = self._checkpoint_path(state, action_id)
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "adaptive_native_checkpoint_v1" or payload.get("action_id") != action_id:
            return None
        if payload.get("adapter_fingerprint", {}).get("sha256") != self._adapter_fingerprint().get("sha256"):
            return None
        if payload.get("input_fingerprint") != _state_fingerprint(state):
            return None
        output = _state_from_payload(payload["output_state"])
        raw = payload.get("event", {})
        return AdapterEvent(
            action_id=action_id,
            action=dict(raw.get("action", self._action_payload(action_id))),
            feasible=bool(raw.get("feasible", output.valid)),
            output=output,
            actual_output={
                **dict(raw.get("actual_output", {})),
                "q": None if output.native is None else output.native.q.copy(),
                "qdot": None if output.native is None else output.native.qdot.copy(),
                "acceleration": None if output.native is None else output.native.acceleration.copy(),
                "snapshot": None if output.native is None else output.native.snapshot,
                "pose": output.pose.copy(),
            },
            elapsed_s=float(raw.get("elapsed_s", output.elapsed)),
            duration_s=float(raw.get("duration_s", 0.0)),
            margin=raw.get("margin"),
            route_id=raw.get("route_id"),
            call_kind="execution",
            source="planning_checkpoint",
            verification_calls=self.verification_calls,
            execution_calls=self.execution_calls,
            verification_count=self.verification_count,
            execution_count=self.execution_count,
            trajectory_records=tuple(raw.get("trajectory_records", ())),
            result=dict(raw.get("result", {})),
        )

    def _materialize_reused_execution(self, event: AdapterEvent) -> None:
        if self.execution_output is None:
            return
        payload = {
            "schema": "adaptive_execution_reuse_v1",
            "source": "planning_checkpoint",
            "action_id": event.action_id,
            "route_id": event.route_id,
            "complete_state_reused": True,
            "trajectory_records": list(event.trajectory_records),
        }
        (self.execution_output / f"reuse_{self.execution_count:04d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        # NPZ traces are immutable deterministic products.  Copy selected
        # records into execution/ when available; the JSON pointer remains the
        # source-of-truth if a caller chose not to retain planning traces.
        if self.planning_output is not None:
            for record in event.trajectory_records:
                source = self.planning_output / f"{record}.npz"
                target = self.execution_output / f"{record}.npz"
                if source.exists() and not target.exists():
                    shutil.copy2(source, target)

    def checked(self, state: EpisodeState, action_id: str) -> AdapterEvent:
        """Plan and verify one action from the exact supplied state."""
        self._require_actual_state(state)
        canonical = self._canonical(action_id)
        key = (_state_fingerprint(state), canonical)
        self.verification_calls += 1
        if key in self._checked_events:
            cached = self._checked_events[key]
            self.verification_count += 1
            return AdapterEvent(**{**cached.__dict__, "call_kind": "verification_cache",
                                   "source": "planning_memory_cache",
                                   "verification_calls": self.verification_calls,
                                   "verification_count": self.verification_count})
        captured = self._planning.capture(state, canonical)
        self.verification_count += 1
        event = self._event_from_capture(state, canonical, captured,
                                         call_kind="verification", source="native_planning",
                                         rollouts=self._planning)
        self._checked_events[key] = event
        self._save_checkpoint(state, event)
        return event

    def execute(self, state: EpisodeState, action_id: str) -> AdapterEvent:
        """Execute one action from the supplied actual state.

        The default path invokes native replay again.  With ``reuse_checked``
        a checkpoint is accepted only when its complete input fingerprint
        matches ``state``; q-only or nominal-state reuse is impossible.
        """
        self._require_actual_state(state)
        canonical = self._canonical(action_id)
        self.execution_calls += 1
        self.execution_count += 1
        if self.reuse_checked:
            event = self._checked_events.get((_state_fingerprint(state), canonical))
            if event is None:
                event = self._load_checkpoint(state, canonical)
            if event is not None:
                reused = AdapterEvent(**{**event.__dict__,
                                         "call_kind": "execution_reuse",
                                         "source": "planning_checkpoint",
                                         "execution_calls": self.execution_calls,
                                         "execution_count": self.execution_count})
                self._materialize_reused_execution(reused)
                return reused
        captured = self._execution.capture(state, canonical)
        event = self._event_from_capture(state, canonical, captured,
                                         call_kind="execution", source="native_execution",
                                         rollouts=self._execution)
        self._save_checkpoint(state, event)
        return event

    def finish(self) -> None:
        """Flush route manifests without imposing a task-time cutoff."""
        if self._closed:
            return
        self._planning.finish()
        self._execution.finish()
        if self.output is not None:
            manifest = {
                "schema": "adaptive_native_adapter_v1",
                "index": self.index,
                "mode": self.mode,
                "backend": self.backend,
                "action_ids": list(self.actions),
                "measured_frame": _json_safe(self.measured_frame),
                "material_address": _json_safe(self.address),
                "physical_frame_policy": "private renderer only; never sent to motion or public event",
                "planning_calls": self.verification_calls,
                "execution_calls": self.execution_calls,
                "verification_count": self.verification_count,
                "execution_count": self.execution_count,
                "proposal_calls": self.proposal_calls,
                "proposal_cache_hits": self.proposal_cache_hits,
                "proposal_dird_queries": self.proposal_dird_queries,
                "reuse_checked": self.reuse_checked,
                "adapter_fingerprint": self._adapter_fingerprint(),
                "no_deadline": True,
            }
            (self.output / "ADAPTER.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")

    def close(self) -> None:
        if self._closed:
            return
        self.finish()
        close = getattr(self.native, "close", None)
        if close is not None:
            close()
        self._closed = True

    def __enter__(self) -> "NativeAdapter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = ["ACTION_IDS", "AdapterEvent", "MotionProposal", "MotionRoute",
           "NativeAdapter", "PredictedEpisodeState", "PROBE_BASES"]
