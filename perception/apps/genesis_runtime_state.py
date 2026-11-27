"""Robot-state adapters used by the unified Genesis/UE runtime.

The runtime has two deliberately separate state sources:

``simulation``
    Keeps the authored Genesis home pose.  It never imports or starts a
    controller and is the safe default for scene/video generation.

``shm``
    Reads the existing ``rm75_state`` relay through :class:`RelayStateBus`.
    The relay is owned by the C++/peirastic controller process; this adapter
    only attaches to the shared-memory segment and never opens a RealMan
    socket.

The canonical RM75 8-DOF order is fixed here so callers do not accidentally
pass the seven arm joints in the wrong place::

    [rail_y_m, joint_1_rad, joint_2_rad, ..., joint_7_rad]

``RelayStateBus.q_meas_8dof`` performs the controller feedback conversion
from arm degrees to radians and prepends the rail position in metres.  This
module validates that result and carries the relay clock metadata alongside
the vector for callers that need timestamp diagnostics.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


RM75_8DOF_JOINT_ORDER: tuple[str, ...] = (
    "rail_y",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
)
RM75_8DOF_UNITS: tuple[str, ...] = (
    "m",
    "rad",
    "rad",
    "rad",
    "rad",
    "rad",
    "rad",
    "rad",
)
RM75_8DOF_SIZE = len(RM75_8DOF_JOINT_ORDER)
# The relay's q_deg slot is the measured/controller joint state.  The
# controller may apply camera-calibration zero offsets while computing its
# separately published FK pose, but consumers of q_meas_8dof must not add the
# same offsets a second time.  Keep this contract machine-readable for runtime
# summaries and downstream UE/Genesis metadata.
RM75_8DOF_Q_SEMANTICS = "raw_measured"
RM75_8DOF_ZERO_OFFSET_POLICY = "not_applied_in_adapter"


@dataclass(frozen=True)
class RobotStateSample:
    """Validated robot state in the canonical RM75 8-DOF order."""

    q8dof: np.ndarray
    source: str
    sequence: int = 0
    wall_time_ns: int = 0
    source_time_s: float = 0.0

    def __post_init__(self) -> None:
        q = np.asarray(self.q8dof, dtype=np.float64).reshape(-1)
        if q.size != RM75_8DOF_SIZE:
            raise ValueError(f"expected {RM75_8DOF_SIZE} RM75 values, got {q.size}")
        if not np.all(np.isfinite(q)):
            raise ValueError("RM75 state contains non-finite values")
        object.__setattr__(self, "q8dof", q.copy())


class RobotStateAdapter(Protocol):
    """Small interface consumed by the Genesis playback loop."""

    source: str

    def read(self) -> RobotStateSample | None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class FittedTrackSample:
    """Metadata-preserving SMPL-X sample decoded from the 5598 track stream."""

    pose55: np.ndarray
    transl: np.ndarray
    betas: np.ndarray
    gender: str
    expression: np.ndarray
    source_id: str
    frame_index: int
    timestamp_ns: int
    pelvis_world_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose55, dtype=np.float32).reshape(-1)
        transl = np.asarray(self.transl, dtype=np.float32).reshape(-1)
        if pose.size != 55 * 3:
            raise ValueError(f"fitted track pose must contain 165 values, got {pose.size}")
        if transl.size != 3:
            raise ValueError(f"fitted track translation must contain 3 values, got {transl.size}")
        for name, value in (("pose55", pose), ("transl", transl)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"fitted track {name} contains non-finite values")
        pelvis = None if self.pelvis_world_m is None else np.asarray(self.pelvis_world_m, dtype=np.float32).reshape(-1)
        if pelvis is not None:
            if pelvis.size != 3:
                raise ValueError(f"fitted track pelvis_world_m must contain 3 values, got {pelvis.size}")
            if not np.all(np.isfinite(pelvis)):
                raise ValueError("fitted track pelvis_world_m contains non-finite values")
        beta = np.asarray(self.betas, dtype=np.float32).reshape(-1)
        expression = np.asarray(self.expression, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(expression)):
            raise ValueError("fitted track shape/expression contains non-finite values")
        gender = str(self.gender or "neutral").lower()
        if gender not in {"male", "female", "neutral"}:
            gender = "neutral"
        object.__setattr__(self, "pose55", pose.copy())
        object.__setattr__(self, "transl", transl.copy())
        object.__setattr__(self, "betas", beta[:10].copy())
        object.__setattr__(self, "expression", expression.copy())
        object.__setattr__(self, "gender", gender)
        object.__setattr__(self, "source_id", str(self.source_id or "fitted_easymocap"))
        object.__setattr__(self, "pelvis_world_m", None if pelvis is None else pelvis.copy())


def decode_fitted_track_payload(data: dict[str, Any]) -> FittedTrackSample | None:
    """Decode current 5598 payload variants while preserving fit metadata.

    The normal terminal-8 payload carries ``smplx_pose_aa_165``.  The older
    live tracker publishes a 72D SMPL pose; it is mapped with the repository's
    SMPL/SMPL-H→SMPL-X55 adapter so the runtime has one canonical human shape.
    """
    if str(data.get("payload_kind", "")) == "clear":
        return None
    pose_raw = data.get("smplx_pose_aa_165")
    if pose_raw is None and data.get("Rh") is not None and data.get("poses") is not None:
        try:
            from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import easymocap_fit_to_smplx55

            pose_raw = easymocap_fit_to_smplx55(
                data["Rh"],
                data["poses"],
                gender=str(data.get("gender", "male") or "male"),
            )
        except (ImportError, TypeError, ValueError):
            return None
    if pose_raw is None:
        pose_raw = data.get("pose_aa")
    if pose_raw is None:
        return None
    try:
        from projects.genesis_ue_sync.anatomy_retarget.pose_adapter import pose_to_smplx55_axis_angle

        pose55 = pose_to_smplx55_axis_angle(pose_raw).reshape(-1)
        transl_raw = data.get("Th", data.get("translation_m", (0.0, 0.0, 0.0)))
        transl = np.asarray(transl_raw, dtype=np.float32).reshape(3)
        pelvis_raw = data.get("pelvis_world_m", data.get("smplx_pelvis_world_m"))
        pelvis_world_m = None if pelvis_raw is None else np.asarray(pelvis_raw, dtype=np.float32).reshape(3)
        betas = np.asarray(data.get("betas", data.get("shapes", ())), dtype=np.float32).reshape(-1)[:10]
        expression = np.asarray(data.get("expression", data.get("expr", ())), dtype=np.float32).reshape(-1)
        frame_index = data.get("frame_index")
        if frame_index is None:
            frame_index = 0
        timestamp_ns = data.get("timestamp_ns")
        if timestamp_ns is None:
            timestamp_ns = 0
        return FittedTrackSample(
            pose55=pose55,
            transl=transl,
            betas=betas,
            gender=str(data.get("gender", "neutral") or "neutral"),
            expression=expression,
            source_id=str(data.get("source_id", data.get("source", "fitted_easymocap")) or "fitted_easymocap"),
            frame_index=int(frame_index),
            timestamp_ns=int(timestamp_ns),
            pelvis_world_m=pelvis_world_m,
        )
    except (TypeError, ValueError, OverflowError):
        return None


class FittedTrackMetadataSubscriber:
    """Read-only 5598 subscriber for fitted pose metadata.

    ``TrackPoseSubscriber`` remains responsible for optional Genesis drawing.
    This lightweight companion keeps fields that the drawer intentionally
    does not need (expression, source id and timestamps) for canonical UE
    publication.  It has no controller or hardware dependency.
    """

    def __init__(self, connect: str, *, topic: str = "amongus_multiview_track_v1") -> None:
        self._connect = str(connect)
        self._topic = str(topic).encode("utf-8")
        self._latest: FittedTrackSample | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: Any = None

    def start(self) -> None:
        import zmq

        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 50)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic)
        sock.connect(self._connect)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, name="fitted-track-metadata", daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        import zmq

        while not self._stop.is_set():
            try:
                parts = self._sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                if self._stop.is_set():
                    break
                continue
            if len(parts) < 2:
                continue
            try:
                sample = decode_fitted_track_payload(json.loads(parts[-1].decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                sample = None
            if sample is not None:
                with self._lock:
                    self._latest = sample

    def latest(self) -> FittedTrackSample | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Let the receiver leave its timed ``recv_multipart`` before
            # closing the socket from this thread.  Closing first races
            # libzmq's signaler and can abort the interpreter.
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None


def _as_q8dof(q: Any, *, source: str) -> np.ndarray:
    """Validate and copy an 8-DOF Genesis state vector."""
    arr = np.asarray(q, dtype=np.float64).reshape(-1)
    if arr.size != RM75_8DOF_SIZE:
        raise ValueError(
            f"{source} must provide {RM75_8DOF_SIZE} values in order "
            f"{list(RM75_8DOF_JOINT_ORDER)}, got {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{source} robot state contains non-finite values")
    return arr.copy()


def _home_to_q8dof(home_q: Any) -> np.ndarray:
    """Normalize a Genesis home vector, accepting legacy 7-DOF arm-only poses."""
    arr = np.asarray(home_q, dtype=np.float64).reshape(-1)
    if arr.size == RM75_8DOF_SIZE:
        return _as_q8dof(arr, source="simulation")
    if arr.size == RM75_8DOF_SIZE - 1:
        # Legacy rm75_6f scene entries have no rail DOF.  The unified state
        # interface still exposes the rail explicitly at its zero pose.
        return _as_q8dof(np.concatenate(([0.0], arr)), source="simulation")
    raise ValueError(f"simulation home pose must have 7 or 8 values, got {arr.size}")


class SimulationRobotStateAdapter:
    """Static authored Genesis state; never touches controller or hardware."""

    source = "simulation"

    def __init__(self, home_q: Any) -> None:
        self._q8dof = _home_to_q8dof(home_q)

    @property
    def q8dof(self) -> np.ndarray:
        return self._q8dof.copy()

    def read(self) -> RobotStateSample:
        return RobotStateSample(q8dof=self._q8dof, source=self.source)

    def close(self) -> None:
        return None


class ShmRobotStateAdapter:
    """Read-only adapter for the existing ``RelayStateBus`` state segment.

    ``timeout_s`` is a local freshness guard.  A stale last frame returns
    ``None`` so the playback loop holds the current Genesis pose instead of
    silently presenting old hardware feedback as a live state.
    """

    source = "shm"

    def __init__(
        self,
        name: str = "rm75_state",
        *,
        timeout_s: float = 1.0,
        bus: Any | None = None,
    ) -> None:
        if float(timeout_s) <= 0.0:
            raise ValueError("SHM state timeout_s must be positive")
        self.name = str(name).strip() or "rm75_state"
        self.timeout_s = float(timeout_s)
        if bus is None:
            # Import lazily.  Importing this adapter must not import the
            # controller entrypoint or initialise any hardware connection.
            from rm75_control.control.admittance_common.state_relay import RelayStateBus

            bus = RelayStateBus(self.name)
        self._bus = bus
        self._last_sequence = 0
        self._last_new_frame_mono = 0.0

    @property
    def bus(self) -> Any:
        return self._bus

    def read(self) -> RobotStateSample | None:
        snap = self._bus.read()
        if snap is None or not bool(getattr(snap, "ok", False)):
            return None
        q_deg = getattr(snap, "q_deg", None)
        if q_deg is None:
            return None
        try:
            # Keep this conversion local to the adapter so the public
            # contract remains unambiguous even if a fake/test bus is used.
            from rm75_control.control.admittance_common.state_bus import expand_q_meas_8dof

            q8 = expand_q_meas_8dof(q_deg, float(getattr(self._bus, "last_rail_m", 0.0)))
            q8 = _as_q8dof(q8, source=f"shm:{self.name}")
        except (ImportError, ValueError, TypeError):
            return None

        seq = int(getattr(snap, "seq", 0) or 0)
        now = time.monotonic()
        if seq != self._last_sequence:
            self._last_sequence = seq
            self._last_new_frame_mono = now
        elif self._last_new_frame_mono <= 0.0:
            self._last_new_frame_mono = now
        elif now - self._last_new_frame_mono > self.timeout_s:
            return None

        return RobotStateSample(
            q8dof=q8,
            source=self.source,
            sequence=seq,
            wall_time_ns=int(getattr(snap, "wall_time_ns", 0) or 0),
            source_time_s=float(getattr(snap, "t_s", 0.0) or 0.0),
        )

    def wait_first(self, timeout_s: float) -> RobotStateSample:
        """Wait for one fresh frame, without changing SHM ownership."""
        timeout = float(timeout_s)
        if timeout < 0.0:
            raise ValueError("wait timeout_s must be non-negative")
        deadline = time.monotonic() + timeout
        while True:
            sample = self.read()
            if sample is not None:
                return sample
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no fresh state on read-only SHM {self.name!r} within {timeout:.3f}s"
                )
            time.sleep(min(0.02, max(0.001, deadline - time.monotonic())))

    def close(self) -> None:
        stop = getattr(self._bus, "stop", None)
        if callable(stop):
            stop()


def make_robot_state_adapter(
    source: str,
    *,
    home_q: Any,
    shm_name: str = "rm75_state",
    shm_timeout_s: float = 1.0,
) -> RobotStateAdapter:
    """Build a state adapter for the CLI's explicit source selection."""
    kind = str(source).strip().lower()
    if kind == "simulation":
        return SimulationRobotStateAdapter(home_q)
    if kind == "shm":
        return ShmRobotStateAdapter(shm_name, timeout_s=shm_timeout_s)
    raise ValueError(f"unknown robot state source {source!r}; expected simulation or shm")


__all__ = [
    "FittedTrackMetadataSubscriber",
    "FittedTrackSample",
    "RM75_8DOF_JOINT_ORDER",
    "RM75_8DOF_SIZE",
    "RM75_8DOF_UNITS",
    "RM75_8DOF_Q_SEMANTICS",
    "RM75_8DOF_ZERO_OFFSET_POLICY",
    "RobotStateAdapter",
    "RobotStateSample",
    "ShmRobotStateAdapter",
    "SimulationRobotStateAdapter",
    "decode_fitted_track_payload",
    "make_robot_state_adapter",
]
