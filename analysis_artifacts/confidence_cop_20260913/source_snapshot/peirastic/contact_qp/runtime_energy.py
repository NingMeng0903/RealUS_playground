"""Uncertified runtime estimates and optional command budgeting on ONE ledger.

Actual measured TCP/tool W and V are distinct from command-model velocity.
Settlement uses EnergyLedger's endpoint-power/error/rate estimate, not an exact
work integral or a certified physical bound. Missing evidence never refunds a
reservation. This adapter neither sends commands nor admits mechanical stops.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .energy import EnergyLedger, ExposureSegment, PortInterval
from .port_constraint import PortEnergyConstraint
from .types import positive, vector


def _time(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a numeric timestamp")
    return positive(value, name, zero=True)


@dataclass(frozen=True)
class _PortSample:
    source_t_s: float
    source_id: object
    wrench: np.ndarray
    velocity: np.ndarray


class RuntimeEnergy:
    """Use aligned measured work without conflating validity and certification.

    ``physical_w_checked`` attests that environment-on-probe wrench sign, SI
    units and reference point/frame are actually bound by the caller. It is
    independent of ``valid`` and ``time_aligned``. ``source_id`` identifies the
    pair of actual wrench/velocity samples; it must never identify a command or
    predicted motion. The source timestamp is their aligned physical time.

    Command budgeting is optional. With it disabled, ``snapshot`` returns None
    so the caller passes QpInput.energy=None. Neither mode certifies the physical
    port. The caller always uses its existing independent mechanical stop path.
    """

    def __init__(self, ledger: EnergyLedger, *, command_budget_enforced=False,
                 max_measurement_age_s=0.05):
        if not isinstance(ledger, EnergyLedger) or ledger.bounds.verified:
            raise ValueError("runtime estimates require one ledger with unverified port bounds")
        if type(command_budget_enforced) is not bool:
            raise ValueError("command_budget_enforced must be a boolean")
        self.ledger = ledger
        self.command_budget_enforced = command_budget_enforced
        self.max_measurement_age_s = positive(max_measurement_age_s, "maximum measurement age")
        self.events = []
        self.monitor_status = "monitor_unavailable"
        self.monitor_reason = "no_completed_physical_interval"
        self._anchor = None
        self._last_source_s = -math.inf
        self._last_now_s = -math.inf
        self._seen_sources = set()
        self._snapshot = None
        self._snapshot_now_s = None
        self._snapshot_source_s = None
        self._reserved = {}
        self._dissipation_model = None

    def bind_dissipation(self, parameters):
        """Freeze the SAME Pc,d model consumed by the outer QP constraint."""
        pc=np.asarray(parameters.get("contact_map",np.zeros((0,6))),dtype=float)
        damping=np.asarray(parameters.get("damping",np.zeros(len(pc))),dtype=float)
        if pc.ndim!=2 or pc.shape[1]!=6 or damping.shape!=(len(pc),):
            raise ValueError("invalid shared contact dissipation model")
        pc=vector(pc,pc.shape);damping=vector(damping,damping.shape)
        if np.any(damping<0):raise ValueError("negative damping")
        previous=self._dissipation_model
        if previous is not None and (not np.array_equal(pc,previous[0]) or not np.array_equal(damping,previous[1])):
            raise ValueError("QP and ledger dissipation models differ; start a new study version")
        self._dissipation_model=(pc,damping)

    def _dissipation_estimate(self, interval):
        if self._dissipation_model is None:self.bind_dissipation({})
        pc,damping=self._dissipation_model
        with np.errstate(over="ignore",invalid="ignore"):
            average=.5*(interval.velocity_start+interval.velocity_end)
            work=float((interval.end_s-interval.start_s)*(damping @ np.square(pc @ average)))
        if not math.isfinite(work):raise ValueError("nonfinite dissipation estimate")
        return work

    def observe_interval(self, interval, *, now_s, physical_w_checked, provenance=None):
        """Consume one complete measured interval, never interpolate it again.

        Mean-velocity squared dissipation is a MONITOR estimate, not an upper
        bound on unresolved motion. Conditional certification instead requires
        an integral upper bound, e.g. dt*sum(d*rbar**2) over a verified interval.
        """
        try:
            now=_time(now_s,"interval receipt time")
            if not isinstance(interval,PortInterval):raise ValueError("PortInterval required")
            if physical_w_checked is not True:return self._unavailable("physical_wrench_semantics_unchecked")
            if now<self._last_now_s or interval.end_s>now or now-interval.start_s>self.max_measurement_age_s:
                return self._unavailable("physical_interval_clock_or_age_invalid")
            self._last_now_s=now
            # Do not accept a caller-supplied second damping model/charge.
            charged=replace(interval,dissipation_work_j=self._dissipation_estimate(interval))
            before=self.ledger.balance_j
            if not self.ledger.settle(charged,now_s=now):
                return self._unavailable("physical_interval_unsettled")
        except (TypeError,ValueError,OverflowError):
            return self._unavailable("physical_interval_rejected")
        self.monitor_status="monitor_estimated"
        self.monitor_reason="measured_bracket_interval_unverified"
        settled=self.ledger.events[-1]
        self.events.append(dict(event="measured_port_interval",start_s=charged.start_s,end_s=charged.end_s,
            port_output_work_upper_j=settled['port_output_work_upper_j'],
            dissipation_work_j=charged.dissipation_work_j,total_charge_j=settled['total_charge_j'],
            balance_change_j=self.ledger.balance_j-before,estimated=True,certified=False,
            dissipation_estimator="dt_sum_d_Pc_mean_velocity_squared_not_upper_bound",
            contact_map=self._dissipation_model[0].tolist(),damping=self._dissipation_model[1].tolist(),
            provenance=provenance or {}))
        return True

    @property
    def facts(self):
        return dict(self.ledger.snapshot(),
                    command_budget_enforced=self.command_budget_enforced,
                    energy_constraint_enabled=self.command_budget_enforced,
                    physical_assurance="unverified", port_assurance="unverified",
                    certified=False, estimated=True,
                    monitor_status=self.monitor_status, monitor_reason=self.monitor_reason)

    def drain_events(self):
        """Transfer log batches to the caller without discarding any liability.

        Call regularly and hand the batches to a bounded nonblocking recorder.
        Ledger certification/envelope messages remain uncertified diagnostics;
        in pure monitoring they are not command admission or stop conditions.
        """
        events = {"runtime": self.events, "ledger": self.ledger.events}
        self.events = []
        self.ledger.events = []
        return events

    def _unavailable(self, reason, **facts):
        self._anchor = None
        self.monitor_status = "monitor_unavailable"
        self.monitor_reason = reason
        self.events.append(dict(event="monitor_unavailable", reason=reason,
                                certified=False, **facts))
        return False

    def observe_port(self, wrench_environment, measured_velocity, *, source_t_s,
                     source_id, now_s, valid, time_aligned, physical_w_checked,
                     reference, calibration_version):
        """Settle only a completed consecutive interval; return settlement fact.

        Any unusable observation fences interpolation. A subsequent usable
        observation first establishes a fresh anchor; it cannot bridge a gap.
        Delayed observations are usable only within the explicit age limit.
        """
        try:
            now = _time(now_s, "observation now")
            stamp = _time(source_t_s, "physical source time")
            if now < self._last_now_s:
                return self._unavailable("observation_clock_regressed")
            self._last_now_s = now
            if stamp > now:
                return self._unavailable("future_physical_sample")
            anchor = self._anchor
            if (anchor is not None and stamp == anchor.source_t_s
                    and source_id == anchor.source_id and valid is True
                    and time_aligned is True and physical_w_checked is True
                    and reference == self.ledger.bounds.reference
                    and calibration_version == self.ledger.bounds.calibration_version
                    and now - stamp <= self.max_measurement_age_s
                    and np.array_equal(vector(wrench_environment, (6,)), anchor.wrench)
                    and np.array_equal(vector(measured_velocity, (6,)), anchor.velocity)):
                self.events.append(dict(event="duplicate_physical_sample_ignored",
                                        source_id=source_id, source_t_s=stamp))
                return False
            if stamp <= self._last_source_s:
                return self._unavailable("duplicate_or_out_of_order_physical_time")
            self._last_source_s = stamp
            if source_id is None or source_id == "" or isinstance(source_id, bool):
                return self._unavailable("missing_physical_source_identity")
            hash(source_id)
            if source_id in self._seen_sources:
                return self._unavailable("reused_physical_source_identity")
            self._seen_sources.add(source_id)
            if any(type(flag) is not bool for flag in (valid, time_aligned, physical_w_checked)):
                return self._unavailable("invalid_physical_validity_flags")
            if not physical_w_checked:
                return self._unavailable("physical_wrench_semantics_unchecked")
            if not valid or not time_aligned:
                return self._unavailable("physical_measurement_missing_or_unaligned")
            b = self.ledger.bounds
            if reference != b.reference or calibration_version != b.calibration_version:
                return self._unavailable("physical_port_version_mismatch")
            if now - stamp > self.max_measurement_age_s:
                return self._unavailable("physical_measurement_too_old")
            current = _PortSample(stamp, source_id,
                                  vector(wrench_environment, (6,), name="physical wrench"),
                                  vector(measured_velocity, (6,), name="actual measured velocity"))
            # Reject derived power overflow before it can become an anchor.
            b.power_upper(current.wrench, current.velocity)
        except (TypeError, ValueError, OverflowError):
            return self._unavailable("invalid_physical_sample")

        previous = self._anchor
        if previous is None:
            self._anchor = current
            self.monitor_status = "monitor_unavailable"
            self.monitor_reason = "awaiting_consecutive_physical_sample"
            return False
        if current.source_t_s - previous.source_t_s > b.max_sample_interval_s:
            self._unavailable("physical_measurement_gap", start_s=previous.source_t_s,
                              end_s=current.source_t_s)
            self._anchor = current
            return False

        interval = PortInterval(previous.source_t_s, current.source_t_s,
                                previous.wrench, current.wrench,
                                previous.velocity, current.velocity,
                                reference=reference, calibration_version=calibration_version,
                                valid=True, time_aligned=True)
        before = self.ledger.balance_j
        try:
            interval=replace(interval,dissipation_work_j=self._dissipation_estimate(interval))
            settled = self.ledger.settle(interval, now_s=now)
        except (TypeError, ValueError, OverflowError):
            return self._unavailable("physical_interval_rejected")
        if not settled:
            return self._unavailable("physical_interval_unsettled")
        self._anchor = current
        self.monitor_status = "monitor_estimated"
        self.monitor_reason = "aligned_endpoint_power_estimate_unverified"
        self.events.append(dict(
            event="measured_port_estimate", start_s=interval.start_s, end_s=interval.end_s,
            settled_at_s=now, source_start=previous.source_id, source_end=current.source_id,
            wrench_start=previous.wrench.tolist(), wrench_end=current.wrench.tolist(),
            velocity_start=previous.velocity.tolist(), velocity_end=current.velocity.tolist(),
            wrench_error=b.wrench_error.tolist(), velocity_error=b.velocity_error.tolist(),
            power_rate_bound_w_s=b.power_rate_bound_w_s,
            reference=reference, calibration_version=calibration_version,
            work_estimate_j=self.ledger.events[-1]["work_upper_j"],
            balance_change_j=self.ledger.balance_j - before,
            balance_j=self.ledger.balance_j, reserved_j=self.ledger.reserved_j,
            physical_w_checked=True, time_aligned=True, valid=True,
            estimated=True, certified=False,
            estimator="max_endpoint_net_output_power_plus_declared_errors_and_rate"))
        return True

    def snapshot(self, *, now_s, hold_s, wrench_environment, source_t_s, **parameters):
        """Build a read-only QP budget, aging source bounds before solving.

        ``parameters`` are PortEnergyConstraint fields (wrench_error/rate,
        tracking_error, contact_map, damping, contact_speed_bound, beta,
        assurance, bounds_version). The available balance always comes from
        this ledger. A new snapshot retires the previous candidate identity.
        """
        self.bind_dissipation(parameters)
        self._snapshot = None
        if not self.command_budget_enforced:
            return None
        now = _time(now_s, "snapshot now")
        source = _time(source_t_s, "wrench source time")
        if source > now or now - source > self.max_measurement_age_s:
            raise ValueError("command wrench source is future or too old")
        if parameters.get("assurance", "command_model") not in ("command_model", "monitor"):
            raise ValueError("runtime command budgets cannot certify declared physical bounds")
        raw = PortEnergyConstraint(wrench_environment=wrench_environment,
                                   available_j=self.ledger.available_j, hold_s=hold_s,
                                   **parameters)
        snapshot = raw.aged(now - source)
        self._snapshot = snapshot
        self._snapshot_now_s = now
        self._snapshot_source_s = source
        return snapshot

    def reserve(self, command_id, snapshot, final_velocity, *, now_s):
        """Recheck the FINAL payload model at actual publication time.

        Call after all payload adjustments and before the first device send.
        Only the candidate's nonnegative modeled expenditure is reserved;
        predicted recovery never changes the balance. This is no H-subspace or
        native execution certificate and must not gate an emergency stop.
        """
        if not self.command_budget_enforced:
            raise ValueError("pure monitoring does not reserve or constrain commands")
        if snapshot is None or snapshot is not self._snapshot:
            raise ValueError("unknown, retired, or already consumed budget snapshot")
        self._snapshot = None  # One candidate can be admitted at most once.
        try:
            now = _time(now_s, "publication now")
            if now < self._snapshot_now_s or now - self._snapshot_source_s > self.max_measurement_age_s:
                raise ValueError("publication time is regressed or wrench source expired")
            final = vector(final_velocity, (6,), name="final payload-model velocity")
            fresh = replace(snapshot.aged(now - self._snapshot_now_s),
                            available_j=self.ledger.available_j)
            if not fresh.admissible(final, tolerance_w=0.0, velocity_tolerance=0.0):
                raise ValueError("final command violates the current energy budget")
            work = fresh.lower_work_j(final, fresh.hold_s)
            power = max(0.0, -work) / fresh.hold_s
            segment = ExposureSegment(now, now + fresh.hold_s, power, "new_only")
            if not self.ledger.reserve(command_id, [segment]):
                raise ValueError("current ledger cannot reserve final command")
        except (TypeError, ValueError, OverflowError) as exc:
            self.events.append(dict(event="command_energy_refused", command_id=command_id,
                                    reason=str(exc), certified=False))
            return False
        self._reserved[command_id] = False
        self.events.append(dict(event="command_energy_reserved", command_id=command_id,
                                start_s=segment.start_s, end_s=segment.end_s,
                                model_lower_work_j=work, reserved_j=segment.liability_j,
                                source_t_s=self._snapshot_source_s,
                                wrench_error=fresh.wrench_error.tolist(),
                                final_velocity=final.tolist(), predicted_recovery_credited_j=0.0,
                                assurance=fresh.assurance, certified=False))
        return True

    def publication_started(self, command_id):
        """Fence refunds immediately before any device send can take effect."""
        if command_id not in self._reserved:
            raise ValueError("unknown runtime command reservation")
        self._reserved[command_id] = True

    def reject_new_only(self, command_id, *, definitely_not_sent):
        """Release only this new candidate proven unexposed, never old work."""
        if type(definitely_not_sent) is not bool:
            raise ValueError("definitely_not_sent must be an explicit boolean")
        if (not definitely_not_sent or command_id not in self._reserved
                or self._reserved[command_id]):
            return False
        released = self.ledger.mark_no_send(command_id)
        if released:
            del self._reserved[command_id]
        return released
