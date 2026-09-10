"""Publication facts, final command admission, and bounded reference commits.

Pure Python: adapters perform device I/O and report what actually happened.
Software abort never rewrites device facts or refunds possible exposure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from types import MappingProxyType

import numpy as np

from .energy import EnergyLedger, branch_exposure, mixed_device_branches
from .geometry import accepted_alpha
from .types import TwistConstraints, positive, vector


class DeviceState(str, Enum):
    NOT_SENT = "not_sent"
    STARTED = "send_started"
    SENT = "published"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected_before_send"
    UNKNOWN = "execution_unknown"


@dataclass(frozen=True)
class ExecutionBounds:
    publish_delay_s: float = .05
    command_hold_s: float = .05
    stop_tail_s: float = .25
    verified: bool = False
    # Bounds at the common instantaneous TCP over publication and stop tails,
    # including changes of Jacobian/frame, not just a frozen command estimate.
    velocity_abs_bound: np.ndarray = field(default_factory=lambda: np.array([.1, .1, .02, 1., 1., 1.]))

    def __post_init__(self):
        for name in ("publish_delay_s", "command_hold_s", "stop_tail_s"):
            positive(getattr(self, name), name)
        if type(self.verified) is not bool:
            raise ValueError("verified must be a boolean")
        value = vector(self.velocity_abs_bound, (6,))
        if np.any(value <= 0):
            raise ValueError("positive full-horizon TCP velocity bounds required")
        object.__setattr__(self, "velocity_abs_bound", value)

    @property
    def horizon_s(self):
        return self.publish_delay_s+self.command_hold_s+self.stop_tail_s


@dataclass(frozen=True)
class FinalCommand:
    sequence: int
    stop_epoch: int
    arm_payload: np.ndarray
    rail_target_m: float
    sent_tool: np.ndarray
    predicted_tool: np.ndarray
    arm_predicted_tool: np.ndarray
    rail_predicted_tool: np.ndarray
    mechanical_valid: bool

    def __post_init__(self):
        if type(self.sequence) is not int or type(self.stop_epoch) is not int or self.sequence <= 0 or self.stop_epoch < 0:
            raise ValueError("positive command identity and nonnegative epoch required")
        object.__setattr__(self, "arm_payload", vector(self.arm_payload, (7,)))
        for name in ("sent_tool", "predicted_tool", "arm_predicted_tool", "rail_predicted_tool"):
            object.__setattr__(self, name, vector(getattr(self, name), (6,)))
        if not math.isfinite(self.rail_target_m) or type(self.mechanical_valid) is not bool:
            raise ValueError("finite rail target and explicit mechanical review required")
        if not np.allclose(self.arm_predicted_tool+self.rail_predicted_tool, self.predicted_tool, atol=1e-12, rtol=0):
            raise ValueError("predicted device branches do not sum to predicted twist")

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        h.update(f"{self.sequence}:{self.stop_epoch}:{self.mechanical_valid}".encode())
        for item in (self.arm_payload, [self.rail_target_m], self.sent_tool, self.predicted_tool,
                     self.arm_predicted_tool, self.rail_predicted_tool):
            h.update(np.asarray(item, dtype="<f8").tobytes())
        return h.hexdigest()


@dataclass
class _Facts:
    states: dict = field(default_factory=lambda: {d: DeviceState.NOT_SENT for d in ("arm", "rail")})
    events: list = field(default_factory=list)
    committed: bool = False
    aborted: bool = False


@dataclass(frozen=True)
class Publication:
    command: FinalCommand
    certificate: TwistConstraints
    reviewed_s: float
    progress_s: float
    _facts: _Facts = field(default_factory=_Facts, repr=False, compare=False)

    @property
    def states(self):
        return MappingProxyType(self._facts.states)

    @property
    def events(self):
        return tuple(MappingProxyType(e) for e in self._facts.events)

    @property
    def committed(self):
        return self._facts.committed

    @property
    def aborted(self):
        return self._facts.aborted


class ExecutionCoordinator:
    def __init__(self, ledger: EnergyLedger, bounds: ExecutionBounds, *, command_tolerance=None):
        self.ledger, self.bounds = ledger, bounds
        if not bounds.verified:
            ledger.invalidate("execution_bounds_unverified_monitoring_only")
        self.tolerance = vector(command_tolerance if command_tolerance is not None else [1e-6]*3+[1e-5]*3, (6,))
        self.stop_epoch = 0
        self.last_sequence = 0
        self.reference_s = 0.
        self.publications = {}
        self.events = []
        self.held = {"arm": np.zeros(6), "rail": np.zeros(6)}
        self.execution_known = True

    def _exposure(self, now_s, wrench_environment):
        # Box corners contain every declared asynchronous arm/rail/stop branch
        # and each previous in-flight command at the current port reference.
        # Absolute motion bounds are a commissioning assumption, not inferred
        # from send success. Using all 64 corners also covers sign reversals.
        bits = np.arange(64, dtype=np.uint64)[:, None] >> np.arange(6, dtype=np.uint64)
        branches = ((bits & 1).astype(float)*2-1)*self.bounds.velocity_abs_bound
        return branch_exposure(now_s, self.bounds.horizon_s, wrench_environment, branches, self.ledger.bounds)

    def review(self, command: FinalCommand, certificate: TwistConstraints, *, now_s, h, alpha, dt_s, wrench_environment):
        """Call after the last payload modification, before either device send."""
        positive(now_s, "review time", zero=True)
        dt = positive(dt_s, "effective command step")
        if dt > .01 or command.stop_epoch != self.stop_epoch or certificate.stop_epoch != self.stop_epoch:
            raise ValueError("invalid effective step or obsolete stop epoch")
        if command.sequence <= self.last_sequence or certificate.sequence != command.sequence:
            raise ValueError("obsolete or mismatched command sequence")
        if certificate.frame != "tcp_tool" or not math.isfinite(certificate.valid_until_s) or now_s > certificate.valid_until_s:
            raise ValueError("invalid or expired final certificate")
        if not command.mechanical_valid or max(certificate.violation(command.sent_tool), certificate.violation(command.predicted_tool)) > 1e-8:
            raise ValueError("final hard constraint review failed")
        if not self.execution_known:
            raise ValueError("execution must be reconciled before ordinary publication")
        if any(not p.committed and not p.aborted for p in self.publications.values()):
            raise ValueError("previous command candidate is still outstanding")
        accepted = accepted_alpha(h, alpha, command.sent_tool, command.predicted_tool, self.tolerance)
        branches = mixed_device_branches(self.held["arm"], self.held["rail"],
                                          command.arm_predicted_tool, command.rail_predicted_tool, np.zeros(6))
        if np.any(abs(branches) > self.bounds.velocity_abs_bound+1e-12):
            self.ledger.invalidate("execution_branch_exceeds_velocity_declaration")
            raise ValueError("execution branch exceeds declared bounds")
        exposure = self._exposure(now_s, wrench_environment)
        if not self.ledger.reserve(command.sequence, [exposure], stop_epoch=self.stop_epoch):
            raise ValueError("budget_insufficient_or_uncertified")
        self.last_sequence = command.sequence
        publication = Publication(command, certificate, now_s, accepted*dt)
        self.publications[command.sequence] = publication
        self.events.append(dict(event="final_review", sequence=command.sequence, fingerprint=command.fingerprint))
        return publication

    def before_send(self, publication, device, *, now_s, command=None):
        """Each device gets an epoch, payload and expiry check immediately at I/O."""
        p = self._owned(publication)
        positive(now_s, "send time", zero=True)
        if device not in p.states or p.states[device] != DeviceState.NOT_SENT:
            raise ValueError("device send identity already consumed")
        if (p.aborted or p.command.stop_epoch != self.stop_epoch or now_s > p.certificate.valid_until_s
                or now_s > p.reviewed_s+self.bounds.publish_delay_s or p.command.sequence != self.last_sequence
                or now_s < self._latest_time(p) or (command is not None and command.fingerprint != p.command.fingerprint)):
            self.abort(p, "stale_or_modified_before_send")
            raise ValueError("stale or modified command before send")
        if not self.ledger.covers(p.command.sequence, now_s, p.reviewed_s+self.bounds.horizon_s) or (self.ledger.bounds.verified and not self.ledger.certification_valid):
            self.abort(p, "reservation_invalid")
            raise ValueError("energy reservation invalid")
        p._facts.states[device] = DeviceState.STARTED
        p._facts.events.append(dict(device=device, state=DeviceState.STARTED.value, time_s=now_s))

    def device_result(self, publication, device, state: DeviceState, *, now_s):
        p = self._owned(publication)
        positive(now_s, "device event time", zero=True)
        state = DeviceState(state)
        if now_s < self._latest_time(p):
            return False
        previous = p.states.get(device)
        if previous == state:
            return False  # Duplicate ACK is not another physical interval.
        allowed = {
            DeviceState.NOT_SENT: (DeviceState.REJECTED,),
            DeviceState.STARTED: (DeviceState.SENT, DeviceState.UNKNOWN, DeviceState.REJECTED),
            DeviceState.SENT: (DeviceState.ACKNOWLEDGED, DeviceState.UNKNOWN),
            DeviceState.UNKNOWN: (DeviceState.SENT, DeviceState.ACKNOWLEDGED),
        }
        if state not in allowed.get(previous, ()):
            return False  # Out-of-order facts never roll a device backward.
        p._facts.states[device] = state
        p._facts.events.append(dict(device=device, state=state.value, time_s=now_s))
        if state in (DeviceState.SENT, DeviceState.ACKNOWLEDGED, DeviceState.UNKNOWN):
            self.held[device] = getattr(p.command, device+"_predicted_tool").copy()
        if state == DeviceState.UNKNOWN or (p.command.stop_epoch != self.stop_epoch and state in (DeviceState.SENT, DeviceState.ACKNOWLEDGED)):
            self.execution_known = False
        if now_s > p.reviewed_s+self.bounds.publish_delay_s and state in (DeviceState.SENT, DeviceState.UNKNOWN):
            self.ledger.invalidate("publication_exceeds_declared_delay")
            self.execution_known = False
        if now_s > p.reviewed_s+self.bounds.horizon_s and state == DeviceState.ACKNOWLEDGED:
            self.ledger.invalidate("late_execution_exceeds_declared_horizon")
        return True

    def commit(self, publication, *, now_s):
        p = self._owned(publication)
        positive(now_s, "commit time", zero=True)
        if p.committed:
            return 0.
        if (p.aborted or p.command.stop_epoch != self.stop_epoch or now_s > p.certificate.valid_until_s
                or now_s > p.reviewed_s+self.bounds.publish_delay_s or p.command.sequence != self.last_sequence
                or now_s < self._latest_time(p) or not self.execution_known
                or (self.ledger.bounds.verified and not self.ledger.certification_valid)
                or not self.ledger.covers(p.command.sequence, now_s, p.reviewed_s+self.bounds.horizon_s)
                or any(s not in (DeviceState.SENT, DeviceState.ACKNOWLEDGED) for s in p.states.values())):
            self.abort(p, "incomplete_or_late_publication")
            return 0.
        p._facts.committed = True
        self.reference_s += p.progress_s
        self.events.append(dict(event="reference_commit", sequence=p.command.sequence, increment_s=p.progress_s))
        return p.progress_s

    def abort(self, publication, reason):
        p = self._owned(publication)
        if p.committed:
            raise ValueError("a completed publication cannot be physically rolled back")
        p._facts.aborted = True
        if any(s in (DeviceState.STARTED, DeviceState.SENT, DeviceState.ACKNOWLEDGED, DeviceState.UNKNOWN) for s in p.states.values()):
            self.execution_known = False
        # Never refund mixed/held liabilities on logical rollback.
        self.events.append(dict(event="publication_aborted", sequence=p.command.sequence, reason=str(reason),
                                states={k: v.value for k, v in p.states.items()}))

    def stop(self, *, now_s, wrench_environment):
        for publication in self.publications.values():
            if not publication.committed and not publication.aborted:
                self.abort(publication, "stop_epoch_retired")
        self.stop_epoch += 1
        self.execution_known = False
        segment = self._exposure(now_s, wrench_environment)
        key = ("stop", self.stop_epoch)
        if not self.ledger.reserve(key, [segment], stop_epoch=self.stop_epoch, stopping=True):
            self.ledger.record_unfunded_exposure(key, [segment], stop_epoch=self.stop_epoch)
        self.events.append(dict(event="stop_requested", epoch=self.stop_epoch, time_s=now_s))
        return self.stop_epoch

    def reconcile_stopped(self, *, now_s, measured_tool, feedback_valid, queues_fenced, no_inflight,
                          arm_stopped=False, rail_stopped=False):
        """Trusted adapter evidence, not a zero command or an SDK return code."""
        positive(now_s, "reconciliation time", zero=True)
        v = vector(measured_tool, (6,))
        if any(type(x) is not bool for x in (feedback_valid, queues_fenced, no_inflight, arm_stopped, rail_stopped)):
            raise ValueError("explicit reconciliation booleans required")
        if not all((feedback_valid, queues_fenced, no_inflight, arm_stopped, rail_stopped)) or np.any(abs(v) > self.tolerance):
            return False
        self.held = {"arm": np.zeros(6), "rail": np.zeros(6)}
        self.execution_known = True
        self.events.append(dict(event="stopped_reconciled", epoch=self.stop_epoch, time_s=now_s))
        return True  # Unsettled energy is deliberately retained.

    def _owned(self, publication):
        if self.publications.get(publication.command.sequence) is not publication:
            raise ValueError("publication belongs to another transaction owner")
        return publication

    @staticmethod
    def _latest_time(publication):
        return max((e["time_s"] for e in publication.events), default=publication.reviewed_s)
