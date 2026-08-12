"""Continuous rail/swivel guide backed by Ruckig.

Ruckig advances the two redundancy coordinates ``(rail, psi_unwrapped)``.
The complete eight-joint target is *not* interpolated: an injected SRS mapping
is called on every tick, so the target remains on the current task-pose
manifold.  Missing Ruckig support and mapping failures are explicit sample
states; there is intentionally no quiet linear/smoothstep fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_TWO_PI = 2.0 * math.pi


class GuideStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    FINISHED = "finished"
    UNAVAILABLE = "ruckig_unavailable"
    MAPPING_FAILED = "mapping_failed"
    RUCKIG_ERROR = "ruckig_error"
    LIMIT_VIOLATION = "limit_violation"


class RuckigUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GuideAvailability:
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class ContinuousGuideLimits:
    """Redundancy and mapped-arm guide limits.

    The rail defaults are the production contract: 0.08 m/s, 0.30 m/s2 and
    1.5 m/s3.  Every arm joint uses the same velocity/acceleration bound.
    """

    rail_v_max_m_s: float = 0.08
    rail_a_max_m_s2: float = 0.30
    rail_j_max_m_s3: float = 1.5
    psi_v_max_rad_s: float = 1.0
    psi_a_max_rad_s2: float = 3.0
    psi_j_max_rad_s3: float = 10.0
    arm_v_max_rad_s: float = 1.0
    arm_a_max_rad_s2: float = 3.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def rail_max_velocity(self) -> float:
        return self.rail_v_max_m_s

    @property
    def rail_max_acceleration(self) -> float:
        return self.rail_a_max_m_s2

    @property
    def rail_max_jerk(self) -> float:
        return self.rail_j_max_m_s3


@dataclass(frozen=True)
class RedundancyState:
    rail_m: float
    psi_rad: float
    branch: int
    winding: int = 0
    rail_velocity_m_s: float = 0.0
    psi_velocity_rad_s: float = 0.0
    rail_acceleration_m_s2: float = 0.0
    psi_acceleration_rad_s2: float = 0.0

    @property
    def psi_unwrapped_rad(self) -> float:
        return float(self.psi_rad + _TWO_PI * self.winding)


@dataclass(frozen=True)
class GuideTarget:
    rail_m: float
    psi_rad: float
    branch: int
    winding: int | None = None
    mapping_input: Any = None


@dataclass(frozen=True)
class SrsMappingContext:
    rail_m: float
    psi_rad: float
    psi_unwrapped_rad: float
    branch: int
    winding: int
    rail_velocity_m_s: float
    psi_velocity_rad_s: float
    rail_acceleration_m_s2: float
    psi_acceleration_rad_s2: float
    mapping_input: Any
    previous_q_goal: np.ndarray | None


@dataclass(frozen=True)
class SrsMappingResult:
    q_goal: Sequence[float]
    qdot_guide: Sequence[float] | None = None


def _readonly(array: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    out = np.asarray(array, dtype=float).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ContinuousGuideSample:
    status: GuideStatus
    rail_m: float
    psi_rad: float
    psi_unwrapped_rad: float
    branch: int
    winding: int
    rail_velocity_m_s: float
    psi_velocity_rad_s: float
    rail_acceleration_m_s2: float
    psi_acceleration_rad_s2: float
    q_goal: np.ndarray | None
    qdot_guide: np.ndarray | None
    qddot_guide: np.ndarray | None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_goal", _readonly(self.q_goal))
        object.__setattr__(self, "qdot_guide", _readonly(self.qdot_guide))
        object.__setattr__(self, "qddot_guide", _readonly(self.qddot_guide))

    @property
    def valid(self) -> bool:
        return self.status in (GuideStatus.WORKING, GuideStatus.FINISHED)

    @property
    def finished(self) -> bool:
        return self.status is GuideStatus.FINISHED

    @property
    def available(self) -> bool:
        return self.status is not GuideStatus.UNAVAILABLE


def _load_ruckig(explicit: Any) -> tuple[Any | None, GuideAvailability]:
    if explicit is False:
        return None, GuideAvailability(False, "Ruckig explicitly disabled/unavailable")
    if explicit is not None:
        missing = [
            name
            for name in ("Ruckig", "InputParameter", "OutputParameter", "Result")
            if not hasattr(explicit, name)
        ]
        if missing:
            return None, GuideAvailability(
                False, f"injected Ruckig API missing: {', '.join(missing)}"
            )
        return explicit, GuideAvailability(True)
    try:
        import ruckig  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        return None, GuideAvailability(False, f"Ruckig import failed: {exc}")
    return ruckig, GuideAvailability(True)


class ContinuousRailGuide:
    """Online Ruckig guide for a committed rail/psi short target."""

    def __init__(
        self,
        srs_mapping: Callable[[SrsMappingContext], Any] | None = None,
        *,
        mapping: Callable[[SrsMappingContext], Any] | None = None,
        dt_s: float = 0.01,
        limits: ContinuousGuideLimits | None = None,
        ruckig_module: Any = None,
    ) -> None:
        if dt_s <= 0.0 or not math.isfinite(float(dt_s)):
            raise ValueError("dt_s must be finite and positive")
        if srs_mapping is not None and mapping is not None:
            raise ValueError("provide either srs_mapping or mapping, not both")
        self.dt_s = float(dt_s)
        self.limits = limits or ContinuousGuideLimits()
        self._mapping = srs_mapping if srs_mapping is not None else mapping
        if self._mapping is None:
            raise ValueError("an SRS mapping callback is required")
        self._api, self.availability = _load_ruckig(ruckig_module)
        self._otg = None
        self._input = None
        self._output = None
        self._initialized = False
        self._target: GuideTarget | None = None
        self._branch = 0
        self._winding = 0
        self._mapping_input: Any = None
        self._previous_q_goal: np.ndarray | None = None
        self._previous_qdot = np.zeros(8)
        self._previous_redundancy_acceleration = np.zeros(2)
        self._last_sample: ContinuousGuideSample | None = None
        self.status = (
            GuideStatus.IDLE if self.availability.available else GuideStatus.UNAVAILABLE
        )
        if self.availability.available:
            assert self._api is not None
            try:
                self._otg = self._api.Ruckig(2, self.dt_s)
                self._input = self._api.InputParameter(2)
                self._output = self._api.OutputParameter(2)
            except Exception as exc:
                self._api = None
                self.availability = GuideAvailability(
                    False, f"Ruckig initialization failed: {type(exc).__name__}: {exc}"
                )
                self.status = GuideStatus.UNAVAILABLE

    @property
    def available(self) -> bool:
        return self.availability.available

    @property
    def last_sample(self) -> ContinuousGuideSample | None:
        return self._last_sample

    @property
    def target(self) -> GuideTarget | None:
        return self._target

    def require_available(self) -> None:
        if not self.available:
            raise RuckigUnavailableError(self.availability.reason)

    def reset(
        self,
        state: RedundancyState | None = None,
        *,
        rail_m: float | None = None,
        psi_rad: float | None = None,
        branch: int = 0,
        winding: int = 0,
        rail_velocity_m_s: float = 0.0,
        psi_velocity_rad_s: float = 0.0,
        rail_acceleration_m_s2: float = 0.0,
        psi_acceleration_rad_s2: float = 0.0,
        q_goal: Sequence[float] | None = None,
        qdot_guide: Sequence[float] | None = None,
        mapping_input: Any = None,
    ) -> GuideStatus:
        """Set measured redundancy state; no trajectory fallback is created."""

        if not self.available:
            self.status = GuideStatus.UNAVAILABLE
            return self.status
        if state is None:
            if rail_m is None or psi_rad is None:
                raise ValueError("reset requires state or both rail_m and psi_rad")
            state = RedundancyState(
                rail_m=float(rail_m),
                psi_rad=float(psi_rad),
                branch=int(branch),
                winding=int(winding),
                rail_velocity_m_s=float(rail_velocity_m_s),
                psi_velocity_rad_s=float(psi_velocity_rad_s),
                rail_acceleration_m_s2=float(rail_acceleration_m_s2),
                psi_acceleration_rad_s2=float(psi_acceleration_rad_s2),
            )
        assert self._input is not None
        self._input.current_position = [state.rail_m, state.psi_unwrapped_rad]
        self._input.current_velocity = [
            state.rail_velocity_m_s,
            state.psi_velocity_rad_s,
        ]
        self._input.current_acceleration = [
            state.rail_acceleration_m_s2,
            state.psi_acceleration_rad_s2,
        ]
        self._input.target_position = [state.rail_m, state.psi_unwrapped_rad]
        self._input.target_velocity = [0.0, 0.0]
        self._input.target_acceleration = [0.0, 0.0]
        self._input.max_velocity = [
            self.limits.rail_v_max_m_s,
            self.limits.psi_v_max_rad_s,
        ]
        self._input.max_acceleration = [
            self.limits.rail_a_max_m_s2,
            self.limits.psi_a_max_rad_s2,
        ]
        self._input.max_jerk = [
            self.limits.rail_j_max_m_s3,
            self.limits.psi_j_max_rad_s3,
        ]
        self._branch = int(state.branch)
        self._winding = int(state.winding)
        self._mapping_input = mapping_input
        self._target = GuideTarget(
            state.rail_m,
            state.psi_rad,
            state.branch,
            state.winding,
            mapping_input,
        )
        self._previous_q_goal = self._normalize_q(q_goal, state.rail_m)
        qdot = self._normalize_vector(qdot_guide)
        self._previous_qdot = np.zeros(8) if qdot is None else qdot
        self._previous_qdot[0] = float(state.rail_velocity_m_s)
        self._previous_redundancy_acceleration = np.array(
            [state.rail_acceleration_m_s2, state.psi_acceleration_rad_s2], dtype=float
        )
        self._initialized = True
        self.status = GuideStatus.IDLE
        self._last_sample = None
        return self.status

    def set_target(
        self,
        target: GuideTarget | None = None,
        *,
        rail_m: float | None = None,
        psi_rad: float | None = None,
        branch: int | None = None,
        winding: int | None = None,
        mapping_input: Any = None,
    ) -> GuideTarget:
        if not self._initialized:
            raise RuntimeError("reset must be called before set_target")
        if target is None:
            if rail_m is None or psi_rad is None:
                raise ValueError("set_target requires target or rail_m and psi_rad")
            target = GuideTarget(
                float(rail_m),
                float(psi_rad),
                self._branch if branch is None else int(branch),
                winding,
                mapping_input,
            )
        assert self._input is not None
        current_psi = float(self._input.current_position[1])
        if target.winding is None:
            wrapped = (float(target.psi_rad) + math.pi) % _TWO_PI - math.pi
            resolved_winding = int(round((current_psi - wrapped) / _TWO_PI))
            target_psi = wrapped + resolved_winding * _TWO_PI
        else:
            resolved_winding = int(target.winding)
            target_psi = float(target.psi_rad) + resolved_winding * _TWO_PI
        self._input.target_position = [float(target.rail_m), target_psi]
        self._input.target_velocity = [0.0, 0.0]
        self._input.target_acceleration = [0.0, 0.0]
        self._branch = int(target.branch)
        self._winding = resolved_winding
        self._mapping_input = target.mapping_input if mapping_input is None else mapping_input
        self._target = GuideTarget(
            float(target.rail_m),
            float(target.psi_rad),
            int(target.branch),
            resolved_winding,
            self._mapping_input,
        )
        return self._target

    def update(self, *, mapping_input: Any = None) -> ContinuousGuideSample:
        """Advance one tick and remap the complete joint target once."""

        if not self.available:
            return self._failure_sample(GuideStatus.UNAVAILABLE, self.availability.reason)
        if not self._initialized:
            return self._failure_sample(GuideStatus.IDLE, "guide has not been reset")
        assert self._api is not None and self._otg is not None
        assert self._input is not None and self._output is not None
        try:
            result = self._otg.update(self._input, self._output)
        except Exception as exc:
            return self._failure_sample(
                GuideStatus.RUCKIG_ERROR,
                f"Ruckig update failed: {type(exc).__name__}: {exc}",
            )
        working = getattr(self._api.Result, "Working")
        finished = getattr(self._api.Result, "Finished")
        if result not in (working, finished):
            return self._failure_sample(GuideStatus.RUCKIG_ERROR, f"Ruckig result {result}")

        position = np.asarray(self._output.new_position, dtype=float).reshape(2)
        velocity = np.asarray(self._output.new_velocity, dtype=float).reshape(2)
        acceleration = np.asarray(self._output.new_acceleration, dtype=float).reshape(2)
        if not self._redundancy_within_limits(velocity, acceleration):
            return self._failure_sample(
                GuideStatus.LIMIT_VIOLATION,
                "Ruckig output exceeded configured redundancy limits",
                position,
                velocity,
                acceleration,
            )

        psi_unwrapped = float(position[1])
        psi = psi_unwrapped - self._winding * _TWO_PI
        context = SrsMappingContext(
            rail_m=float(position[0]),
            psi_rad=psi,
            psi_unwrapped_rad=psi_unwrapped,
            branch=self._branch,
            winding=self._winding,
            rail_velocity_m_s=float(velocity[0]),
            psi_velocity_rad_s=float(velocity[1]),
            rail_acceleration_m_s2=float(acceleration[0]),
            psi_acceleration_rad_s2=float(acceleration[1]),
            mapping_input=self._mapping_input if mapping_input is None else mapping_input,
            previous_q_goal=_readonly(self._previous_q_goal),
        )
        try:
            mapped = self._coerce_mapping(self._mapping(context), float(position[0]))
        except Exception as exc:
            return self._failure_sample(
                GuideStatus.MAPPING_FAILED,
                f"SRS mapping failed: {type(exc).__name__}: {exc}",
                position,
                velocity,
                acceleration,
            )
        if mapped is None:
            return self._failure_sample(
                GuideStatus.MAPPING_FAILED,
                "SRS mapping returned no finite 7/8-DOF target",
                position,
                velocity,
                acceleration,
            )
        q_goal, supplied_qdot = mapped
        q_goal[0] = float(position[0])
        raw_qdot = supplied_qdot
        if raw_qdot is None:
            raw_qdot = np.zeros(8)
            if self._previous_q_goal is not None:
                delta = self._joint_delta(self._previous_q_goal, q_goal)
                raw_qdot = delta / self.dt_s
        raw_qdot[0] = float(velocity[0])
        qdot, qddot = self._limit_mapped_guide(raw_qdot)
        qdot[0] = float(velocity[0])
        qddot[0] = float(acceleration[0])

        # Commit OTG state only after the SRS mapping and safety shaping pass.
        self._output.pass_to_input(self._input)
        self._previous_q_goal = q_goal.copy()
        self._previous_qdot = qdot.copy()
        self._previous_redundancy_acceleration = acceleration.copy()
        status = GuideStatus.FINISHED if result == finished else GuideStatus.WORKING
        self.status = status
        sample = ContinuousGuideSample(
            status=status,
            rail_m=float(position[0]),
            psi_rad=psi,
            psi_unwrapped_rad=psi_unwrapped,
            branch=self._branch,
            winding=self._winding,
            rail_velocity_m_s=float(velocity[0]),
            psi_velocity_rad_s=float(velocity[1]),
            rail_acceleration_m_s2=float(acceleration[0]),
            psi_acceleration_rad_s2=float(acceleration[1]),
            q_goal=q_goal,
            qdot_guide=qdot,
            qddot_guide=qddot,
        )
        self._last_sample = sample
        return sample

    step = update
    tick = update

    def _redundancy_within_limits(
        self, velocity: np.ndarray, acceleration: np.ndarray
    ) -> bool:
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(acceleration)):
            return False
        jerk = (acceleration - self._previous_redundancy_acceleration) / self.dt_s
        return bool(
            abs(velocity[0]) <= self.limits.rail_v_max_m_s + 1e-7
            and abs(acceleration[0]) <= self.limits.rail_a_max_m_s2 + 1e-7
            and abs(jerk[0]) <= self.limits.rail_j_max_m_s3 + 1e-5
            and abs(velocity[1]) <= self.limits.psi_v_max_rad_s + 1e-7
            and abs(acceleration[1]) <= self.limits.psi_a_max_rad_s2 + 1e-7
            and abs(jerk[1]) <= self.limits.psi_j_max_rad_s3 + 1e-5
        )

    def _limit_mapped_guide(self, raw_qdot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        desired = np.asarray(raw_qdot, dtype=float).reshape(8).copy()
        arm = desired[1:]
        peak = float(np.max(np.abs(arm)))
        if peak > self.limits.arm_v_max_rad_s:
            arm *= self.limits.arm_v_max_rad_s / peak
        previous = self._previous_qdot[1:]
        delta = arm - previous
        peak_delta = float(np.max(np.abs(delta)))
        allowed_delta = self.limits.arm_a_max_rad_s2 * self.dt_s
        if peak_delta > allowed_delta:
            arm = previous + delta * (allowed_delta / peak_delta)
        desired[1:] = arm
        qddot = (desired - self._previous_qdot) / self.dt_s
        return desired, qddot

    @staticmethod
    def _joint_delta(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        delta = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
        if delta.size > 1:
            delta[1:] = (delta[1:] + math.pi) % _TWO_PI - math.pi
        return delta

    @staticmethod
    def _normalize_q(q: Sequence[float] | None, rail_m: float) -> np.ndarray | None:
        if q is None:
            return None
        out = np.asarray(q, dtype=float).reshape(-1)
        if out.size == 7:
            out = np.concatenate(([float(rail_m)], out))
        if out.size != 8 or not np.all(np.isfinite(out)):
            return None
        return out.copy()

    @staticmethod
    def _normalize_vector(value: Sequence[float] | None) -> np.ndarray | None:
        if value is None:
            return None
        out = np.asarray(value, dtype=float).reshape(-1)
        if out.size == 7:
            out = np.concatenate(([0.0], out))
        if out.size != 8 or not np.all(np.isfinite(out)):
            raise ValueError("qdot_guide must be a finite 7- or 8-vector")
        return out.copy()

    def _coerce_mapping(
        self, raw: Any, rail_m: float
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        qdot = None
        if isinstance(raw, SrsMappingResult):
            q, qdot_raw = raw.q_goal, raw.qdot_guide
        elif isinstance(raw, Mapping):
            q = raw.get("q_goal", raw.get("q"))
            qdot_raw = raw.get("qdot_guide", raw.get("qdot"))
        elif isinstance(raw, tuple) and len(raw) == 2:
            q, qdot_raw = raw
        else:
            q, qdot_raw = raw, None
        q_goal = self._normalize_q(q, rail_m)
        if q_goal is None:
            return None
        if qdot_raw is not None:
            qdot = self._normalize_vector(qdot_raw)
        return q_goal, qdot

    def _failure_sample(
        self,
        status: GuideStatus,
        reason: str,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
    ) -> ContinuousGuideSample:
        self.status = status
        if position is None:
            if self._initialized and self._input is not None:
                position = np.asarray(self._input.current_position, dtype=float)
                velocity = np.asarray(self._input.current_velocity, dtype=float)
                acceleration = np.asarray(self._input.current_acceleration, dtype=float)
            else:
                position = np.zeros(2)
                velocity = np.zeros(2)
                acceleration = np.zeros(2)
        assert velocity is not None and acceleration is not None
        psi_unwrapped = float(position[1])
        sample = ContinuousGuideSample(
            status=status,
            rail_m=float(position[0]),
            psi_rad=psi_unwrapped - self._winding * _TWO_PI,
            psi_unwrapped_rad=psi_unwrapped,
            branch=self._branch,
            winding=self._winding,
            rail_velocity_m_s=float(velocity[0]),
            psi_velocity_rad_s=float(velocity[1]),
            rail_acceleration_m_s2=float(acceleration[0]),
            psi_acceleration_rad_s2=float(acceleration[1]),
            q_goal=_readonly(self._previous_q_goal),
            qdot_guide=np.zeros(8) if self._previous_q_goal is not None else None,
            qddot_guide=np.zeros(8) if self._previous_q_goal is not None else None,
            reason=str(reason),
        )
        self._last_sample = sample
        return sample


# Integration-friendly aliases.
ContinuousSrsGuide = ContinuousRailGuide
RuckigSrsGuide = ContinuousRailGuide
GuideLimits = ContinuousGuideLimits
GuideSample = ContinuousGuideSample


__all__ = [
    "ContinuousGuideLimits",
    "ContinuousGuideSample",
    "ContinuousRailGuide",
    "ContinuousSrsGuide",
    "GuideAvailability",
    "GuideLimits",
    "GuideSample",
    "GuideStatus",
    "GuideTarget",
    "RedundancyState",
    "RuckigSrsGuide",
    "RuckigUnavailableError",
    "SrsMappingContext",
    "SrsMappingResult",
]
