"""Publication lifetime and dual-device transaction, with no energy accounting."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Lease:
    command_id: int
    created_s: float
    expires_s: float
    committed_s: float | None = None


class CommandLease:
    def __init__(self, max_interval_s=.05):
        if isinstance(max_interval_s, bool) or not math.isfinite(max_interval_s) or max_interval_s <= 0:
            raise ValueError('positive finite command.max_interval_s required')
        self.max_command_interval_s = float(max_interval_s)
        self.active = self.pending = None
        self.started = False
        self.last_time_s = None
        self.latched_reason = None

    def advance(self, now_s):
        if not math.isfinite(now_s) or (self.last_time_s is not None and now_s < self.last_time_s):
            self.latched_reason = self.latched_reason or 'lease_clock_invalid'
            return False
        self.last_time_s = float(now_s)
        if self.active is not None and now_s >= self.active.expires_s:
            self.latched_reason = self.latched_reason or 'committed_lease_expired'
        return self.latched_reason is None

    def reserve(self, command_id, *, created_s, now_s):
        if not self.advance(now_s) or self.pending is not None or self.started:
            return False
        expiry = created_s+self.max_command_interval_s
        if not math.isfinite(created_s) or not created_s <= now_s < expiry:
            return False
        self.pending = Lease(command_id, float(created_s), float(expiry))
        return True

    def publication_started(self, command_id, *, now_s):
        if (not self.advance(now_s) or self.pending is None or self.started
                or self.pending.command_id != command_id or now_s >= self.pending.expires_s):
            return False
        self.started = True
        return True

    def commit(self, command_id, *, now_s, dual_success):
        # The old lease must have been live at dispatch. Transport completion
        # can cross that old expiry; the reserved new expiry is never renewed.
        p = self.pending
        valid = (self.latched_reason is None and p is not None and self.started
                 and p.command_id == command_id and dual_success and math.isfinite(now_s)
                 and self.last_time_s <= now_s < p.expires_s)
        if not valid:
            self.latched_reason = self.latched_reason or 'invalid_or_partial_publication'
            raise RuntimeError(self.latched_reason)
        self.active = Lease(p.command_id, p.created_s, p.expires_s, float(now_s))
        self.pending = None
        self.started = False
        self.last_time_s = float(now_s)

    def reject_new_only(self, *, definitely_not_sent, now_s):
        if self.started or not definitely_not_sent:
            self.latched_reason = self.latched_reason or 'publication_outcome_unknown'
        self.pending = None
        self.started = False
        self.advance(now_s)

    def stop(self, *, now_s):
        self.latched_reason = self.latched_reason or 'stopped'
        self.pending = None
        self.started = False
        self.last_time_s = float(now_s)
