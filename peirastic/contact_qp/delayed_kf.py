"""Fixed-noise, effective-image-time KF with bounded out-of-sequence replay.

Only image confidence is predicted. This module never shifts force timestamps.
Ingest uses ContactObservation.effective_time_s directly: extraction has already
applied the calibrated capture delay, or the recording was already aligned.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import ContactObservation, WEAK_SIDE_FEATURE_VERSION, REGION_FEATURE_VERSION, positive


@dataclass(frozen=True)
class KfConfig:
    # Deliberately required: deployment must supply fixed offline-fitted noise.
    q: float
    r: float
    max_age_s: float = .3
    history_s: float = .3
    initial_rate_variance: float = 1.
    channels: int = 2
    observation_kind: str = 'legacy_lr'

    def __post_init__(self):
        for name in ('q', 'r', 'max_age_s', 'history_s', 'initial_rate_variance'):
            object.__setattr__(self, name, positive(getattr(self, name), name, zero=name == 'q'))
        if self.history_s < self.max_age_s:
            raise ValueError('history_s must cover max_age_s')
        if isinstance(self.channels, bool) or not isinstance(self.channels, (int, np.integer)) or not 2 <= self.channels <= 64:
            raise ValueError('KF channels must be an integer in [2,64]')
        if self.observation_kind not in ('legacy_lr', 'regions') or (self.observation_kind == 'legacy_lr' and self.channels != 2):
            raise ValueError('regional KF requires observation_kind=regions')


def transition(dt, q, channels=2):
    """Continuous white-acceleration covariance, N independent CV blocks."""
    if not math.isfinite(dt) or dt < 0:
        raise ValueError('KF prediction requires finite nonnegative dt')
    f = np.eye(2*channels)
    f[np.arange(channels)*2, np.arange(channels)*2+1] = dt
    block = q * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    process = np.kron(np.eye(channels), block)
    return f, process


def predict_state(state, covariance, dt, q):
    f, process = transition(dt, q, len(state)//2)
    p = f @ covariance @ f.T + process
    return f @ state, (p+p.T)/2


def measurement_update(state, covariance, quality, r):
    """Joseph-form update; exported for identical offline innovation fitting."""
    channels = len(state)//2
    h = np.eye(2*channels)[::2]
    measurement_noise = np.eye(channels)*r
    innovation = np.asarray(quality)-h @ state
    s = h @ covariance @ h.T + measurement_noise
    k = np.linalg.solve(s, h @ covariance).T
    residual = np.eye(2*channels)-k @ h
    p = residual @ covariance @ residual.T + k @ measurement_noise @ k.T
    return state+k @ innovation, (p+p.T)/2, innovation, s


@dataclass(frozen=True)
class ConfidencePrediction:
    state: np.ndarray | None
    covariance: np.ndarray | None
    age_s: float | None
    valid: bool
    reason: str
    effective_time_s: float | None = None
    frame_seq: int | None = None
    generation: int = 0

    @property
    def quality_raw(self):
        return None if self.state is None else self.state[::2].copy()

    @property
    def quality_task(self):
        # Validity is a separate gate. Never disguise an invalid frame as zero.
        return None if not self.valid else np.clip(self.quality_raw, 0., 1.)

    def to_dict(self):
        return dict(state=None if self.state is None else self.state.tolist(),
                    covariance=None if self.covariance is None else self.covariance.tolist(),
                    quality_raw=None if self.quality_raw is None else self.quality_raw.tolist(),
                    quality_task=None if self.quality_task is None else self.quality_task.tolist(),
                    age_s=self.age_s, valid=self.valid, reason=self.reason,
                    effective_time_s=self.effective_time_s, frame_seq=self.frame_seq,
                    generation=self.generation)


class DelayConfidenceKF:
    """Consume each frame once, replay delayed observations at their own time.

    The anchor is the posterior before the retained window. Rebuilding from it
    makes an in-window out-of-order insertion numerically equal to ordered input.
    predict() does not change this posterior or count as an image measurement.
    """
    def __init__(self, config: KfConfig):
        self.config = config
        self.generation = 0
        self._identity = None
        self._retired_sources = set()
        self._retired_identities = set()
        self._now = -math.inf
        self.reset()

    def reset(self):
        self._latest_record = None
        self._entries = []  # (observation, posterior state, posterior covariance)
        self._anchor = None  # (effective time, posterior state, covariance)
        self._seen = {}  # sequence -> effective time, bounded with the window
        self._validity_event = None  # newest (effective time, validity, reason)
        self._identity = None
        self.last_reason = 'empty'
        self.last_innovation = None
        self.last_innovation_covariance = None

    def _prune(self, now):
        cutoff = now-self.config.history_s
        while self._entries and self._entries[0][0].effective_time_s < cutoff:
            obs, x, p = self._entries.pop(0)
            self._anchor = (obs.effective_time_s, x, p)
        self._seen = {seq: stamp for seq, stamp in self._seen.items() if stamp >= cutoff}

    def _rebuild(self):
        previous = self._anchor
        for i, (obs, _, _) in enumerate(self._entries):
            if previous is None:
                x = np.zeros(2*self.config.channels)
                x[::2] = self._quality(obs)
                p = np.diag(np.tile([self.config.r, self.config.initial_rate_variance], self.config.channels))
                innovation = innovation_covariance = None
            else:
                stamp, x, p = previous
                x, p = predict_state(x, p, obs.effective_time_s-stamp, self.config.q)
                x, p, innovation, innovation_covariance = measurement_update(x, p, self._quality(obs), self.config.r)
            self._entries[i] = (obs, x, p)
            previous = (obs.effective_time_s, x, p)
        if self._entries:
            self._latest_record = self._entries[-1]
        self.last_innovation = innovation if self._entries else None
        self.last_innovation_covariance = innovation_covariance if self._entries else None

    def _quality(self, obs):
        return obs.region_confidence if self.config.observation_kind == 'regions' else obs.confidence_lr

    def ingest(self, obs: ContactObservation, now_s: float):
        now = float(now_s)
        def reject(reason):
            self.last_reason = reason
            return False
        if not math.isfinite(now) or now < self._now:
            return reject('bad_or_backwards_now')
        if obs.received_time_s > now+1e-9 or obs.effective_time_s > now:
            return reject('future_timestamp')
        self._now = now
        self._prune(now)
        if now-obs.effective_time_s > self.config.history_s:
            return reject('too_old')
        # Unsupported evidence may pause the visual task, but must never retire
        # the healthy stream: a later compatible frame can resume that stream.
        regional = self.config.observation_kind == 'regions'
        evidence_version = obs.region_feature_version if regional else obs.weakside_feature_version
        required_version = REGION_FEATURE_VERSION if regional else WEAK_SIDE_FEATURE_VERSION
        quality = self._quality(obs)
        if (evidence_version != required_version or obs.timestamp_semantics != 'effective_image_time'
                or quality is None or len(quality) != self.config.channels):
            reason = 'feature_version_or_timestamp_semantics'
            if self._validity_event is None or obs.effective_time_s >= self._validity_event[0]:
                self._validity_event = (obs.effective_time_s, False, reason)
            return reject(reason)
        identity = (obs.source_id, obs.version, evidence_version, obs.timestamp_semantics,
                    (obs.region_layout_version, tuple(obs.region_edges)) if regional else None)
        if obs.source_id in self._retired_sources:
            return reject('retired_source')
        if identity in self._retired_identities:
            return reject('retired_identity')
        changed = self._identity is not None and identity != self._identity
        if changed:
            self._retired_identities.add(self._identity)
            if obs.source_id != self._identity[0]:
                self._retired_sources.add(self._identity[0])
            self.reset()
            self.generation += 1
        self._identity = identity
        if obs.frame_seq in self._seen:
            return reject('duplicate_frame')
        if self._anchor is not None and obs.effective_time_s <= self._anchor[0]:
            return reject('before_history_anchor')
        if any(entry[0].effective_time_s == obs.effective_time_s for entry in self._entries):
            return reject('duplicate_effective_time')
        self._seen[obs.frame_seq] = obs.effective_time_s
        valid = bool(obs.region_valid.all() if regional else
                     obs.confidence_lr_valid.all() and obs.valid[[0, 2]].all())
        if self._validity_event is None or obs.effective_time_s >= self._validity_event[0]:
            self._validity_event = (obs.effective_time_s, valid, 'valid' if valid else 'invalid_windows')
        if not valid:
            return reject('invalid_windows')
        self._entries.append((obs, None, None))
        self._entries.sort(key=lambda entry: entry[0].effective_time_s)
        self._rebuild()
        self.last_reason = 'reset_updated' if changed else 'updated'
        return True

    def predict(self, now_s: float):
        now = float(now_s)
        if not math.isfinite(now) or now < self._now:
            return ConfidencePrediction(None, None, None, False, 'bad_or_backwards_now', generation=self.generation)
        self._now = now
        # Retain the latest record for diagnostics even after the history expires.
        if self._latest_record is None:
            return ConfidencePrediction(None, None, None, False, self.last_reason, generation=self.generation)
        obs, x, p = self._latest_record
        age = now-obs.effective_time_s
        if age < 0:
            return ConfidencePrediction(None, None, age, False, 'future_timestamp', generation=self.generation)
        x, p = predict_state(x, p, age, self.config.q)
        valid = age <= self.config.max_age_s
        reason = 'valid' if valid else 'image_expired'
        if self._validity_event is not None and not self._validity_event[1]:
            valid, reason = False, self._validity_event[2]
        return ConfidencePrediction(x, p, age, bool(valid), reason, obs.effective_time_s,
                                    obs.frame_seq, self.generation)
