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
    """Diagnostic weak-side intervals; never a repair permission timer."""
    def __init__(self, c_min, improvement_deadband, window_s, emit):
        self.c_min = c_min
        self.deadband = improvement_deadband
        self.window_s = window_s
        self.emit = emit
        self.key = None
        self.active = {}

    def observe(self, observation, now_s, reasons):
        if observation is None:
            for item in self.active.values(): item['reasons'].add('image_unavailable')
            return
        key = (observation.source_id, observation.frame_seq, observation.effective_time_s)
        if key == self.key:
            for item in self.active.values(): item['reasons'].update(reasons)
            return
        self.key = key
        for name, index in (('left', 0), ('right', 2)):
            quality = float(observation.quality[index])
            item = self.active.get(name)
            if quality >= self.c_min:
                if item is not None:
                    self._close(name, now_s, 'quality_recovered')
                continue
            if item is None:
                item = self.active[name] = dict(start_s=now_s, baseline=quality,
                    best=quality, last_quality=quality, frames=0, reasons=set(), marked=False)
                self.emit('quality_interval_start', side=name, start_s=now_s, quality=quality)
            item['last_quality'] = quality
            item['best'] = max(item['best'], quality)
            item['frames'] += 1
            item['reasons'].update(reasons)
            if (not item['marked'] and item['frames'] > 1 and
                    now_s-item['start_s'] >= self.window_s and
                    item['best']-item['baseline'] <= self.deadband):
                item['marked'] = True
                self.emit('quality_no_improvement', side=name, start_s=item['start_s'],
                          time_s=now_s, reasons=sorted(item['reasons']), diagnostic_only=True)

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
