"""One mechanical energy ledger with earmarked exposure and measured settlement.

Positive P = -W_environment · V is energy delivered by the robot. Reservations
sum positive worst-case segment work, hence dominate every prefix without
spending predicted recovery. They are liabilities against ONE settled balance.
Hardware mode remains monitoring until port/time/stop bounds are verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .types import positive, vector


@dataclass(frozen=True)
class PortBounds:
    wrench_error: np.ndarray = field(default_factory=lambda: np.zeros(6))
    velocity_error: np.ndarray = field(default_factory=lambda: np.zeros(6))
    power_rate_bound_w_s: float = 0.
    max_sample_interval_s: float = .02
    reference: str = "tcp_tool"
    calibration_version: str = "unverified"
    verified: bool = False

    def __post_init__(self):
        if type(self.verified) is not bool:
            raise ValueError("verified must be a boolean")
        for name in ("wrench_error", "velocity_error"):
            v = vector(getattr(self, name), (6,), name=name)
            if np.any(v < 0):
                raise ValueError("port error bounds must be nonnegative")
            object.__setattr__(self, name, v)
        positive(self.power_rate_bound_w_s, "power rate bound", zero=True)
        positive(self.max_sample_interval_s, "maximum sample interval")
        if not self.reference or not self.calibration_version:
            raise ValueError("explicit port reference and calibration version required")

    def power_upper(self, wrench_environment, velocity):
        w, v = vector(wrench_environment, (6,)), vector(velocity, (6,))
        with np.errstate(over="ignore", invalid="ignore"):
            error = abs(w) @ self.velocity_error + abs(v) @ self.wrench_error + self.wrench_error @ self.velocity_error
            power = float(-w @ v+error)
        if not math.isfinite(power):
            raise ValueError("nonfinite derived port power")
        return power


@dataclass(frozen=True)
class ExposureSegment:
    start_s: float
    end_s: float
    power_upper_w: float
    source: str = "old_or_mixed"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.start_s, self.end_s, self.power_upper_w)) or self.end_s <= self.start_s:
            raise ValueError("finite forward exposure interval required")
        if not self.source:
            raise ValueError("exposure source required")

    @property
    def liability_j(self):
        value = max(0., self.power_upper_w)*(self.end_s-self.start_s)
        if not math.isfinite(value):
            raise ValueError("nonfinite exposure work")
        return value


def branch_exposure(start_s, duration_s, wrench_environment, velocity_branches, bounds, *, source="old_or_mixed"):
    """Include old, new, mixed-device and stopping branches supplied by adapter."""
    velocities = np.asarray(velocity_branches, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] != 6 or not len(velocities):
        raise ValueError("explicit nonempty six-dimensional execution branches required")
    horizon = positive(duration_s, "declared execution horizon")
    powers = [bounds.power_upper(wrench_environment, v) for v in velocities]
    # W and V bounds are declared over the full horizon; this additional term
    # covers the declared possible variation of power inside that horizon.
    upper = max(powers)+bounds.power_rate_bound_w_s*horizon
    return ExposureSegment(float(start_s), float(start_s+horizon), upper, source)


def mixed_device_branches(old_arm, old_rail, new_arm, new_rail, stopping, *, stop_arm=None, stop_rail=None):
    oa, orail, na, nr, stop = [vector(v, (6,)) for v in (old_arm, old_rail, new_arm, new_rail, stopping)]
    if (stop_arm is None) != (stop_rail is None):
        raise ValueError("both per-device stopping components required")
    if stop_arm is None:
        if np.any(stop != 0):
            raise ValueError("nonzero stopping branch requires per-device velocities")
        sa = sr = np.zeros(6)
    else:
        sa, sr = vector(stop_arm, (6,)), vector(stop_rail, (6,))
        if not np.allclose(sa+sr, stop, atol=1e-12, rtol=0.):
            raise ValueError("stopping components do not sum to stopping twist")
    return np.stack([a+r for a in (oa, na, sa) for r in (orail, nr, sr)])


@dataclass(frozen=True)
class PortInterval:
    start_s: float
    end_s: float
    wrench_start: np.ndarray
    wrench_end: np.ndarray
    velocity_start: np.ndarray
    velocity_end: np.ndarray
    reference: str = "tcp_tool"
    calibration_version: str = "unverified"
    valid: bool = True
    time_aligned: bool = False

    def __post_init__(self):
        if type(self.valid) is not bool or type(self.time_aligned) is not bool:
            raise ValueError("port validity flags must be booleans")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError("finite forward physical interval required")
        for name in ("wrench_start", "wrench_end", "velocity_start", "velocity_end"):
            object.__setattr__(self, name, vector(getattr(self, name), (6,), name=name))


@dataclass
class _Reservation:
    segments: list[ExposureSegment]
    kind: str
    stop_epoch: int
    exposure_possible: bool = True

    @property
    def liability_j(self):
        return sum(s.liability_j for s in self.segments)


class EnergyLedger:
    def __init__(self, initial_j, capacity_j, stopping_reserve_j, bounds: PortBounds):
        self.capacity_j = positive(capacity_j, "capacity")
        self.balance_j = positive(initial_j, "initial balance", zero=True)
        self.stopping_reserve_j = positive(stopping_reserve_j, "stopping reserve", zero=True)
        if not self.stopping_reserve_j <= self.balance_j <= self.capacity_j:
            raise ValueError("stopping reserve <= initial energy <= capacity required")
        self.bounds = bounds
        self.certification_valid = bool(bounds.verified)
        self.certification_reason = "" if bounds.verified else "port_bounds_unverified_monitoring_only"
        self._reservations = {}
        self._used_ids = set()
        self._settled = []
        self.events = []

    @property
    def reserved_j(self):
        return sum(r.liability_j for r in self._reservations.values())

    @property
    def available_j(self):
        return self.balance_j-self.reserved_j-self.stopping_reserve_j

    def invalidate(self, reason):
        self.certification_valid = False
        self.certification_reason = str(reason)
        self.events.append({"event": "certification_invalid", "reason": str(reason)})

    def reserve(self, command_id, segments, *, stop_epoch=0, stopping=False):
        if type(stopping) is not bool:
            raise ValueError("stopping must be a boolean")
        if self.bounds.verified and not self.certification_valid and not stopping:
            self.events.append(dict(event="certification_refused", command_id=command_id))
            return False
        if command_id in self._used_ids:
            raise ValueError("command reservation identity cannot be reused")
        items = list(segments)
        if not items or any(not isinstance(s, ExposureSegment) for s in items):
            raise ValueError("explicit exposure segments required")
        request = sum(s.liability_j for s in items)
        if not math.isfinite(request):
            raise ValueError("nonfinite total reserved work")
        free = self.balance_j-self.reserved_j-(0. if stopping else self.stopping_reserve_j)
        if request > free+1e-12:
            self.events.append(dict(event="budget_insufficient", command_id=command_id, required_j=request, available_j=free))
            return False
        self._used_ids.add(command_id)
        self._reservations[command_id] = _Reservation(items, "stop" if stopping else "ordinary", int(stop_epoch))
        self.events.append(dict(event="reserved", command_id=command_id, energy_j=request))
        return True

    def liability(self, command_id):
        reservation = self._reservations.get(command_id)
        return 0. if reservation is None else reservation.liability_j

    def covers(self, command_id, start_s, end_s):
        reservation = self._reservations.get(command_id)
        if reservation is None or not all(math.isfinite(x) for x in (start_s, end_s)) or end_s <= start_s:
            return False
        cursor = start_s
        for segment in sorted(reservation.segments, key=lambda s: s.start_s):
            if segment.start_s > cursor:
                return False
            cursor = max(cursor, segment.end_s)
            if cursor >= end_s:
                return True
        return False

    def record_unfunded_exposure(self, command_id, segments, *, stop_epoch=0):
        """Necessary stopping/late exposure is a debt even without admission."""
        if command_id in self._used_ids:
            raise ValueError("command reservation identity cannot be reused")
        items = list(segments)
        if not items or not math.isfinite(sum(s.liability_j for s in items)):
            raise ValueError("finite explicit emergency exposure required")
        self._used_ids.add(command_id)
        self._reservations[command_id] = _Reservation(items, "unfunded", int(stop_epoch))
        self.invalidate("unfunded_execution_exposure")

    def mark_no_send(self, command_id, *, old_exposure_proven_absent=False):
        """A reject is not a refund: old/mixed exposure needs its own proof."""
        if type(old_exposure_proven_absent) is not bool:
            raise ValueError("exposure proof must be a boolean")
        reservation = self._reservations.get(command_id)
        if reservation is None:
            return False
        reservation.exposure_possible = any(s.source != "new_only" for s in reservation.segments) and not old_exposure_proven_absent
        if reservation.exposure_possible:
            self.events.append(dict(event="no_send_old_exposure_retained", command_id=command_id))
            return False
        del self._reservations[command_id]
        self.events.append(dict(event="unexposed_release", command_id=command_id))
        return True

    def settle(self, interval: PortInterval, *, now_s):
        if not math.isfinite(now_s):
            raise ValueError("finite settlement clock required")
        if interval.end_s > now_s+1e-12:
            raise ValueError("cannot spend future measured recovery")
        for start, end in self._settled:
            if interval.start_s < end and interval.end_s > start:
                raise ValueError("physical interval overlaps already settled work")
        b = self.bounds
        dt = interval.end_s-interval.start_s
        if (not interval.valid or not interval.time_aligned or dt > b.max_sample_interval_s+1e-12
                or interval.reference != b.reference or interval.calibration_version != b.calibration_version):
            self.events.append(dict(event="settlement_missing_or_unaligned", start_s=interval.start_s, end_s=interval.end_s))
            if dt > b.max_sample_interval_s+1e-12 or interval.reference != b.reference or interval.calibration_version != b.calibration_version:
                self.invalidate("physical_port_declaration_violated")
            return False
        try:
            upper = max(b.power_upper(interval.wrench_start, interval.velocity_start),
                        b.power_upper(interval.wrench_end, interval.velocity_end))+b.power_rate_bound_w_s*dt/2
            work = upper*dt
            if not all(math.isfinite(x) for x in (upper, work, self.balance_j-work)):
                raise ValueError("nonfinite derived energy")
        except ValueError:
            self.invalidate("nonfinite_port_arithmetic")
            return False
        # Every physical interval debits only once even when several overlapping
        # reservations conservatively cover it. Matching earmarks release here.
        remaining = {}
        released = 0.
        envelope = []
        for identity, reservation in self._reservations.items():
            pieces = []
            for segment in reservation.segments:
                lo, hi = max(segment.start_s, interval.start_s), min(segment.end_s, interval.end_s)
                if hi <= lo:
                    pieces.append(segment)
                    continue
                released += max(0., segment.power_upper_w)*(hi-lo)
                envelope.append((lo, hi, segment.power_upper_w))
                if segment.start_s < lo:
                    pieces.append(ExposureSegment(segment.start_s, lo, segment.power_upper_w, segment.source))
                if hi < segment.end_s:
                    pieces.append(ExposureSegment(hi, segment.end_s, segment.power_upper_w, segment.source))
            reservation.segments = pieces
            if pieces:
                remaining[identity] = reservation
        self._reservations = remaining
        merged = []
        for start, end in sorted((*self._settled, (interval.start_s, interval.end_s))):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._settled = merged
        self.balance_j = min(self.capacity_j, self.balance_j-work)  # no lower clipping / no minted energy
        if self.balance_j < -1e-12:
            self.invalidate("measured_energy_overdraw")
        # Detect consumption beyond the declared envelope at any subinterval.
        grid = sorted({interval.start_s, interval.end_s, *(a for a, _, _ in envelope), *(z for _, z, _ in envelope)})
        for start, end in zip(grid[:-1], grid[1:]):
            covering = [p for a, z, p in envelope if a <= start+1e-12 and z >= end-1e-12]
            if upper > max(covering, default=0.)+1e-10:
                self.invalidate("execution_exceeded_reserved_power_envelope")
                break
        self.events.append(dict(event="settled", start_s=interval.start_s, end_s=interval.end_s,
                                work_upper_j=work, released_j=released, balance_j=self.balance_j))
        return True

    def snapshot(self):
        return dict(balance_j=self.balance_j, reserved_j=self.reserved_j, available_j=self.available_j,
                    stopping_reserve_j=self.stopping_reserve_j, certification_valid=self.certification_valid,
                    certification_reason=self.certification_reason, pending_count=len(self._reservations))
