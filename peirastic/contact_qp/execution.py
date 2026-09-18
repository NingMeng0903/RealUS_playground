"""Versioned execution policy state; never authorizes a device publication."""
from dataclasses import dataclass
import math
import numpy as np


EXECUTION_POLICY = 'continuous_recovery_v1'


class ProposalDeferred(RuntimeError):
    """A numerical candidate was refused before either transport was started."""


@dataclass
class QualityProgress:
    """Quality owns a positive target, mechanics may further limit delivery.

    Evidence is read once per image. The slew state commits only with a sent
    command. A refused tick creates no progress debt or command state change.
    """
    ramp_s: float
    c_min: float
    alpha: float = 1.0
    frame_key: object = None
    loss: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.ramp_s) or self.ramp_s <= 0:
            raise ValueError('continuous alpha needs the existing positive path ramp')
        if not 0 < self.c_min < 1:
            raise ValueError('invalid quality threshold')

    def preview(self, observation, dt_s):
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError('positive actual control interval required')
        if observation is None:
            # Unavailable evidence cannot be treated as recovered quality.
            self.frame_key = None
            self.loss = 1.0
        else:
            key = (observation.source_id, observation.frame_seq, observation.effective_time_s)
            if key != self.frame_key:
                self.frame_key = key
                q = np.asarray(observation.quality)[[0, 2]]
                self.loss = float(np.max(np.maximum(self.c_min-q, 0.) / self.c_min))
        target = float(np.clip(1.0 - .75*self.loss, .25, 1.))
        value = float(np.clip(target, self.alpha-.75*dt_s/self.ramp_s,
                              self.alpha+.75*dt_s/self.ramp_s))
        return target, value

    def commit(self, value):
        if not math.isfinite(value) or not .25 <= value <= 1.:
            raise ValueError('invalid quality alpha')
        self.alpha = float(value)


class QualityIntervals:
    """Weak-side intervals. A marked no-improvement interval escalates repair."""
    def __init__(self, c_min, improvement_deadband, window_s, emit, *, region_count=None,
                 recover_quality=None, recover_hold_s=0., hold_while_open=False,
                 escalation_omega_rad_s=None, alpha_hold=0.25,
                 hold_omega_rad_s=None, hold_omega_s=None):
        self.c_min = c_min
        self.deadband = improvement_deadband
        self.window_s = window_s
        self.emit = emit
        self.key = None
        self.active = {}
        self.region_count=region_count
        self.last_request_sign = 0.0
        recover = float(c_min if recover_quality is None else recover_quality)
        if not math.isfinite(recover) or recover < c_min or recover > 1.:
            raise ValueError('recover_quality must be in [c_min, 1]')
        hold = float(recover_hold_s)
        if not math.isfinite(hold) or hold < 0.:
            raise ValueError('recover_hold_s must be finite and nonnegative')
        self.recover_quality = recover
        self.recover_hold_s = hold
        self.hold_while_open = bool(hold_while_open)
        if escalation_omega_rad_s is not None:
            floor = float(escalation_omega_rad_s)
            if not math.isfinite(floor) or floor <= 0.:
                raise ValueError('escalation_omega_rad_s must be finite and positive')
            escalation_omega_rad_s = floor
        self.escalation_omega_rad_s = escalation_omega_rad_s
        if hold_omega_rad_s is not None:
            value = float(hold_omega_rad_s)
            if not math.isfinite(value) or value <= 0.:
                raise ValueError('hold_omega_rad_s must be finite and positive')
            if escalation_omega_rad_s is not None and value > escalation_omega_rad_s:
                raise ValueError('hold_omega_rad_s must not exceed escalation_omega_rad_s')
            hold_omega_rad_s = value
        self.hold_omega_rad_s = hold_omega_rad_s
        if hold_omega_s is not None:
            span = float(hold_omega_s)
            if not math.isfinite(span) or span <= 0.:
                raise ValueError('hold_omega_s must be finite and positive')
            hold_omega_s = span
        self.hold_omega_s = hold_omega_s
        if not math.isfinite(alpha_hold) or not 0. < alpha_hold <= 1.:
            raise ValueError('alpha_hold must be in (0,1]')
        self.alpha_hold = float(alpha_hold)

    def observe(self, observation, now_s, reasons):
        if observation is None:
            for item in self.active.values(): item['reasons'].add('image_unavailable')
            return
        key = (observation.source_id, observation.frame_seq, observation.effective_time_s)
        if key == self.key:
            for item in self.active.values(): item['reasons'].update(reasons)
            return
        self.key = key
        entries=(tuple((f'region_{i}',i) for i in range(self.region_count)) if self.region_count is not None
                 else (('left',0),('right',2)))
        values=observation.region_confidence if self.region_count is not None else observation.quality
        validity=observation.region_valid if self.region_count is not None else (True,True,True)
        if values is None or len(values)!=(self.region_count if self.region_count is not None else 3):return
        for name, index in entries:
            if not validity[index]:
                if name in self.active:self.active[name]['reasons'].add('region_unknown')
                continue
            quality = float(values[index])
            item = self.active.get(name)
            if quality >= self.c_min:
                if item is not None:
                    if quality < self.recover_quality:
                        item['above_recover_s'] = None
                        item['last_quality'] = quality
                        item['reasons'].update(reasons)
                        continue
                    if item.get('above_recover_s') is None:
                        item['above_recover_s'] = now_s
                    if now_s - item['above_recover_s'] >= self.recover_hold_s:
                        self._close(name, now_s, 'quality_recovered')
                    else:
                        item['last_quality'] = quality
                        item['reasons'].update(reasons)
                continue
            if item is None:
                item = self.active[name] = dict(start_s=now_s, baseline=quality,
                    best=quality, last_quality=quality, frames=0, reasons=set(), marked=False,
                    last_weak_s=now_s)
                self.emit('quality_interval_start', side=name, start_s=now_s, quality=quality)
            item['last_quality'] = quality
            item['best'] = max(item['best'], quality)
            item['above_recover_s'] = None
            item['last_weak_s'] = now_s
            item['frames'] += 1
            item['reasons'].update(reasons)
            if (not item['marked'] and item['frames'] > 1 and
                    now_s-item['start_s'] >= self.window_s and
                    item['best']-item['baseline'] <= self.deadband):
                item['marked'] = True
                self.emit('quality_no_improvement', side=name, start_s=item['start_s'],
                          time_s=now_s, reasons=sorted(item['reasons']), diagnostic_only=False)

    @property
    def escalation_active(self):
        return any(bool(item.get('marked')) for item in self.active.values())

    def _hold_floor(self, now_s):
        """Small ω floor for a bounded window after a region was last weak.

        Region q25 climbs back over c_min while 2-4 mm of B-mode shadow is still
        on screen: 58% of the >=2 mm frames in 019 read 'all_regions_sufficient'.
        Releasing ω on the metric alone lets the gap reopen before the next image
        arrives, which is the punch-and-release seen in 019. The floor therefore
        survives the metric, but only for hold_omega_s past the last sub-threshold
        frame -- roughly the 0.47 s sensing round trip. It is deliberately far
        below the escalation floor: 011 pinned ω at 0.1875 rad/s through recovered
        dips and pushed the opposite region to 0.68.
        """
        if self.hold_omega_rad_s is None or not (self.hold_while_open and self.active):
            return None
        if self.hold_omega_s is None or now_s is None:
            return self.hold_omega_rad_s
        if any(float(now_s)-float(item.get('last_weak_s', now_s)) <= self.hold_omega_s
               for item in self.active.values()):
            return self.hold_omega_rad_s
        return None

    def apply_escalation(self, request_rad_s, alpha_preferred, *, repair_speed_m_s, now_s=None):
        """Hold path at α_hold while weak; floor ω while the deficit is unsettled.

        Two separate floors. The large escalation floor still needs a marked
        no-improvement interval. The small hold floor only needs an open interval
        and a recent weak frame; see _hold_floor for why the metric alone is not
        a safe release condition.
        """
        request = 0. if request_rad_s is None else float(request_rad_s)
        if abs(request) > 0.:
            self.last_request_sign = float(np.sign(request))
        if not self.escalation_active and not (self.hold_while_open and self.active):
            return request, alpha_preferred
        alpha = (self.alpha_hold if alpha_preferred is None
                 else min(float(alpha_preferred), self.alpha_hold))
        if self.escalation_active:
            floor = (6.25 * float(repair_speed_m_s) if self.escalation_omega_rad_s is None
                     else self.escalation_omega_rad_s)
        else:
            floor = self._hold_floor(now_s)
            if floor is None:
                return request, alpha
        sign = float(np.sign(request)) if request != 0. else float(self.last_request_sign)
        if sign == 0. or not math.isfinite(floor) or floor <= 0.:
            return request, alpha
        return sign * max(abs(request), floor), alpha

    def add_reasons(self, reasons):
        for item in self.active.values(): item['reasons'].update(reasons)

    def _close(self, name, now_s, reason):
        item = self.active.pop(name)
        self.emit('quality_interval_end', side=name, start_s=item['start_s'], end_s=now_s,
                  duration_s=now_s-item['start_s'], unique_frames=item['frames'],
                  last_quality=item['last_quality'], best_quality=item['best'],
                  no_improvement_marked=item['marked'], reasons=sorted(item['reasons']), end_reason=reason)

    def close(self, now_s):
        for name in list(self.active): self._close(name, now_s, 'scan_ended')
