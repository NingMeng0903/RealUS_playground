"""Small, non-blocking wrapper for slow posture planners.

The servo thread must never wait for collision checking or a graph search.  A
``PosturePlanner`` therefore owns one worker and a *latest-wins* request slot:
``submit`` only copies the request and wakes the worker, while ``latest`` only
takes a lock long enough to copy a published record.

An unsuccessful or old calculation does not make the last command disappear
in one servo tick.  The last good value is retained and its ``confidence`` is
smoothly brought to zero.  Consumers must still inspect ``status``/``valid``;
the retained value is a hand-over aid, not a claim that an invalid plan became
valid again.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable, Generic, Mapping, TypeVar

import numpy as np


RequestT = TypeVar("RequestT")
ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class PosturePlanningRequest:
    """Trajectory-agnostic input published by the control thread."""

    robot_state: object
    current_task_reference: object
    optional_reference_horizon: object | None = None


class PlannerStatus(str, Enum):
    """State of the value returned by :meth:`PosturePlanner.latest`."""

    IDLE = "idle"
    PENDING = "pending"
    VALID = "valid"
    STALE = "stale"
    INVALID = "invalid"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class PlanComputation(Generic[ValueT]):
    """Result returned by the calculation callback.

    ``valid_for_s`` overrides the planner-wide staleness interval for one
    result.  Returning ``None`` from a callback is equivalent to
    ``PlanComputation.invalid()``.
    """

    value: ValueT | None
    valid: bool = True
    reason: str = ""
    valid_for_s: float | None = None

    @classmethod
    def invalid(cls, reason: str = "invalid plan") -> "PlanComputation[ValueT]":
        return cls(value=None, valid=False, reason=str(reason))


@dataclass(frozen=True)
class PlannerSnapshot(Generic[ValueT]):
    """Immutable, lock-free-after-return view of planner state.

    ``value`` is the most recent *good* value, including during an invalid or
    stale hand-over.  ``confidence`` is C2-continuous at both ends of the fade
    and lies in ``[0, 1]``.
    """

    sequence: int
    status: PlannerStatus
    value: ValueT | None
    submitted_at: float | None
    completed_at: float | None
    age_s: float
    confidence: float
    reason: str = ""
    planning: bool = False

    @property
    def valid(self) -> bool:
        return self.status is PlannerStatus.VALID

    @property
    def stale(self) -> bool:
        return self.status is PlannerStatus.STALE

    @property
    def invalid(self) -> bool:
        return self.status is PlannerStatus.INVALID

    @property
    def blend(self) -> float:
        """Alias used by guide/controller integrations."""

        return self.confidence


@dataclass
class _Request(Generic[RequestT]):
    sequence: int
    value: RequestT
    submitted_at: float


@dataclass
class _Published(Generic[ValueT]):
    sequence: int = 0
    value: ValueT | None = None
    submitted_at: float | None = None
    completed_at: float | None = None
    valid_for_s: float | None = None
    failure_at: float | None = None
    reason: str = ""
    has_good_value: bool = False


def _smooth_falloff(elapsed: float, duration: float) -> float:
    """One minus quintic smoothstep, clamped to [0, 1]."""

    if duration <= 0.0:
        return 0.0 if elapsed > 0.0 else 1.0
    u = min(1.0, max(0.0, float(elapsed) / float(duration)))
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return float(1.0 - s)


def _freeze_value(value):
    """Best-effort deep immutable copy without changing useful value types."""

    if isinstance(value, np.ndarray):
        out = np.array(value, copy=True)
        out.setflags(write=False)
        return out
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(v) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {f.name: _freeze_value(getattr(value, f.name)) for f in fields(value)}
        try:
            return replace(value, **updates)
        except (TypeError, ValueError):
            # Some user dataclasses have custom ``__init__`` methods.  They
            # still get an isolated deep copy even if their class cannot be
            # made structurally immutable here.
            return copy.deepcopy(value)
    return copy.deepcopy(value)


class PosturePlanner(Generic[RequestT, ValueT]):
    """Run ``calculate(request)`` on one background thread.

    Parameters
    ----------
    calculate:
        Callback returning a value, :class:`PlanComputation`, or ``None``.
        Exceptions are published as an invalid result; they never escape on
        the servo thread.
    stale_after_s:
        Age at which a good value changes to ``STALE``.  ``math.inf`` disables
        age staleness.
    fade_after_s:
        C2 fade duration after staleness or a failed calculation.
    validator:
        Optional final fail-closed predicate for otherwise successful values.
    clock:
        Monotonic clock injection used by deterministic tests.
    """

    def __init__(
        self,
        calculate: Callable[[RequestT], ValueT | PlanComputation[ValueT] | None],
        *,
        stale_after_s: float = 0.5,
        fade_after_s: float = 0.5,
        validator: Callable[[ValueT], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        name: str = "posture-planner",
        autostart: bool = True,
    ) -> None:
        if stale_after_s < 0.0:
            raise ValueError("stale_after_s must be non-negative")
        if fade_after_s < 0.0:
            raise ValueError("fade_after_s must be non-negative")
        self._calculate = calculate
        self._validator = validator
        self._stale_after_s = float(stale_after_s)
        self._fade_after_s = float(fade_after_s)
        self._clock = clock
        self._condition = threading.Condition(threading.Lock())
        self._pending: _Request[RequestT] | None = None
        self._inflight_sequence: int | None = None
        self._last_submitted_sequence = 0
        self._published: _Published[ValueT] = _Published()
        self._stop = False
        self._started = False
        self._thread = threading.Thread(target=self._worker, name=name, daemon=True)
        if autostart:
            self.start()

    def start(self) -> None:
        """Start the worker (idempotent until shutdown)."""

        with self._condition:
            if self._stop:
                raise RuntimeError("planner has been shut down")
            if self._started:
                return
            self._started = True
            self._thread.start()

    def submit(
        self,
        request: RequestT | None = None,
        *,
        robot_state: object | None = None,
        current_task_reference: object | None = None,
        optional_reference_horizon: object | None = None,
        timestamp_s: float | None = None,
    ) -> int:
        """Copy and enqueue ``request`` without waiting for calculation.

        There is deliberately no unbounded queue.  If submissions outrun the
        planner, pending requests are replaced and an obsolete in-flight
        result is not published.
        """

        if request is not None and (
            robot_state is not None
            or current_task_reference is not None
            or optional_reference_horizon is not None
        ):
            raise ValueError("pass request or generic posture-planning fields, not both")
        if request is None:
            if robot_state is None or current_task_reference is None:
                raise ValueError(
                    "generic submit requires robot_state and current_task_reference"
                )
            request = PosturePlanningRequest(
                robot_state=robot_state,
                current_task_reference=current_task_reference,
                optional_reference_horizon=optional_reference_horizon,
            )  # type: ignore[assignment]
        submitted_at = float(self._clock() if timestamp_s is None else timestamp_s)
        request_copy = copy.deepcopy(request)
        with self._condition:
            if self._stop:
                raise RuntimeError("cannot submit to a shut down planner")
            if not self._started:
                self._started = True
                self._thread.start()
            self._last_submitted_sequence += 1
            sequence = self._last_submitted_sequence
            self._pending = _Request(sequence, request_copy, submitted_at)
            self._condition.notify()
            return sequence

    def latest(self, *, now_s: float | None = None) -> PlannerSnapshot[ValueT]:
        """Return an immutable snapshot immediately."""

        now = float(self._clock() if now_s is None else now_s)
        with self._condition:
            pub = copy.copy(self._published)
            pending = self._pending is not None or self._inflight_sequence is not None
            stopped = self._stop

        if pub.completed_at is None:
            status = PlannerStatus.SHUTDOWN if stopped else (
                PlannerStatus.PENDING if pending else PlannerStatus.IDLE
            )
            return PlannerSnapshot(
                sequence=pub.sequence,
                status=status,
                value=None,
                submitted_at=pub.submitted_at,
                completed_at=None,
                age_s=math.inf,
                confidence=0.0,
                reason=pub.reason,
                planning=pending,
            )

        age = max(0.0, now - pub.completed_at)
        stale_after = (
            self._stale_after_s if pub.valid_for_s is None else max(0.0, pub.valid_for_s)
        )
        stale_confidence = 1.0
        stale = age > stale_after
        if stale:
            stale_confidence = _smooth_falloff(age - stale_after, self._fade_after_s)

        if pub.failure_at is not None:
            failed_for = max(0.0, now - pub.failure_at)
            failure_confidence = _smooth_falloff(failed_for, self._fade_after_s)
            confidence = min(stale_confidence, failure_confidence)
            status = PlannerStatus.INVALID
        elif stale:
            confidence = stale_confidence
            status = PlannerStatus.STALE
        elif stopped:
            # A good final snapshot remains readable, but callers can see that
            # no more plans will arrive.
            confidence = stale_confidence
            status = PlannerStatus.SHUTDOWN
        else:
            confidence = 1.0
            status = PlannerStatus.VALID

        return PlannerSnapshot(
            sequence=pub.sequence,
            status=status,
            value=_freeze_value(pub.value) if pub.has_good_value else None,
            submitted_at=pub.submitted_at,
            completed_at=pub.completed_at,
            age_s=age,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reason=pub.reason,
            planning=pending,
        )

    @property
    def latest_snapshot(self) -> PlannerSnapshot[ValueT]:
        return self.latest()

    def wait_for(self, sequence: int | None = None, timeout_s: float = 1.0) -> bool:
        """Wait for test/setup code; real-time consumers should use ``latest``.

        Returns ``True`` when ``sequence`` (or the latest submitted sequence)
        was processed, including a published invalid result.
        """

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            target = self._last_submitted_sequence if sequence is None else int(sequence)
            while self._published.sequence < target and not self._stop:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(remaining)
            return self._published.sequence >= target

    def shutdown(
        self,
        *,
        wait: bool = True,
        timeout_s: float | None = 1.0,
        cancel_pending: bool = True,
    ) -> bool:
        """Stop accepting work and, optionally, join the worker.

        Python cannot safely pre-empt a callback already executing.  A bounded
        ``timeout_s`` therefore lets teardown remain deterministic.
        """

        with self._condition:
            self._stop = True
            if cancel_pending:
                self._pending = None
            self._condition.notify_all()
            started = self._started
        if wait and started and threading.current_thread() is not self._thread:
            self._thread.join(timeout=None if timeout_s is None else max(0.0, timeout_s))
        return not (started and self._thread.is_alive())

    @property
    def is_alive(self) -> bool:
        return bool(self._started and self._thread.is_alive())

    def __enter__(self) -> "PosturePlanner[RequestT, ValueT]":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._stop and self._pending is None:
                    self._inflight_sequence = None
                    self._condition.notify_all()
                    return
                item = self._pending
                self._pending = None
                assert item is not None
                self._inflight_sequence = item.sequence

            try:
                raw = self._calculate(item.value)
                if isinstance(raw, PlanComputation):
                    result = raw
                elif raw is None:
                    result = PlanComputation.invalid("planner returned no result")
                else:
                    result = PlanComputation(value=raw)
                if result.valid and result.value is not None and self._validator is not None:
                    if not bool(self._validator(result.value)):
                        result = PlanComputation.invalid("planner result failed validation")
                if result.valid and result.value is None:
                    result = PlanComputation.invalid("valid planner result has no value")
            except Exception as exc:  # fail closed at the thread boundary
                result = PlanComputation.invalid(f"{type(exc).__name__}: {exc}")

            frozen_value = None
            if result.valid:
                try:
                    frozen_value = _freeze_value(result.value)
                except Exception as exc:
                    result = PlanComputation.invalid(
                        f"result snapshot failed: {type(exc).__name__}: {exc}"
                    )

            completed_at = float(self._clock())
            with self._condition:
                self._inflight_sequence = None
                # Latest-wins publication.  A callback that finished after a
                # newer submit must not overwrite the newer request's future.
                obsolete = item.sequence < self._last_submitted_sequence
                if not obsolete:
                    self._published.sequence = item.sequence
                    self._published.submitted_at = item.submitted_at
                    if result.valid:
                        self._published.value = frozen_value
                        self._published.completed_at = completed_at
                        self._published.valid_for_s = result.valid_for_s
                        self._published.failure_at = None
                        self._published.reason = result.reason
                        self._published.has_good_value = True
                    else:
                        # Preserve the last good value and its completion time
                        # for a smooth hand-over.  With no previous good value,
                        # age is anchored at this failed calculation.
                        if not self._published.has_good_value:
                            self._published.completed_at = completed_at
                        self._published.failure_at = completed_at
                        self._published.reason = result.reason or "invalid plan"
                self._condition.notify_all()


# Descriptive aliases used by a few integrations.
PosturePlanSnapshot = PlannerSnapshot
PosturePlanResult = PlanComputation


__all__ = [
    "PlanComputation",
    "PlannerSnapshot",
    "PlannerStatus",
    "PosturePlanningRequest",
    "PosturePlanResult",
    "PosturePlanSnapshot",
    "PosturePlanner",
]
