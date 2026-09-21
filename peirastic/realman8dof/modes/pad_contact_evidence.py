"""Image evidence for handheld contact repair; no mechanical control state.

Raw near-field observations determine direction and recovery. Confidence KF
predictions are retained as trend telemetry, never as completion evidence.
"""
from dataclasses import dataclass
from collections import deque
import math

import numpy as np

from peirastic.contact_qp.delayed_kf import DelayConfidenceKF, KfConfig
from peirastic.contact_qp.features import FeatureConfig, LatestObservation
from peirastic.contact_qp.types import ECHO_FEATURE_VERSION, REGION_FEATURE_VERSION
from .contact_recording import FeatureReceiver


@dataclass(frozen=True)
class PadEvidenceRequest:
    omega_request_rad_s: float
    alpha_target: float
    valid: bool
    reason: str
    facts: dict


class PadContactEvidence:
    def __init__(self, config, receiver=None):
        self.config = config
        self.pad = config['pad']
        self.settings = self.pad['evidence']
        self.strict_layout = self.pad.get('policy') == 'pad_scan_v1'
        self.feature_config = FeatureConfig(**config['feature']['config'])
        self._validate_settings()
        self.receiver = receiver if receiver is not None else FeatureReceiver(config['feature_endpoint'])
        self.latest = LatestObservation()
        self.kf = DelayConfidenceKF(KfConfig(**config['image_kf']))
        self.baseline = None
        self.noise = None
        self._warming_mean = None
        self._warming_noise = None
        self._stable_since = None
        self._last_force = None
        self._force_history = deque()
        self._force_tripped = False
        self._force_unstable_until = None
        self._last_echo_time = None
        self._direction = 0
        self._candidate_direction = 0
        self._direction_frames = 0
        self._repair_since = None
        self._recovery_since = None
        # Scan permission starts with the same fresh-image confirmation used
        # after a defect. Learning an echo baseline is a separate process.
        self._recovery_required = self.strict_layout
        self._current_kind = 'empty'
        self._current_request = 0.
        self.last_facts = {}
        self._observation_facts = {}
        self._last_now = None

    def _validate_settings(self):
        self.settings.setdefault('echo_moment_compress', 0.)
        self.settings.setdefault('echo_moment_cubic_knee', 0.)
        self.settings.setdefault('force_step_release_n', self.settings['force_step_max_n'])
        self.settings.setdefault('force_unstable_hold_s', 0.)
        for name in ('baseline_hold_s', 'baseline_tau_s', 'noise_sigma', 'echo_ratio', 'absolute_echo_floor',
                     'force_step_max_n', 'force_window_s', 'force_window_range_n'):
            v = self.settings[name]
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'pad evidence {name} must be finite and positive')
        for name in ('minimum_dark_columns', 'direction_confirm_frames'):
            v = self.settings[name]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'pad evidence {name} must be a positive integer')
        if self.settings['echo_ratio'] >= 1 or self.settings['absolute_echo_floor'] >= 1:
            raise ValueError('echo ratios and floor must be below one')
        if not np.array_equal(self.settings['near_depth'], [.04, .22]):
            raise ValueError('nearfield_column_echo_v1 requires depth [.04,.22]')

    def close(self, *, wait=False):
        self.receiver.close(wait=wait)

    def _reset_stream(self):
        self.baseline = self.noise = None
        self._warming_mean = self._warming_noise = None
        self._stable_since = self._last_echo_time = None
        self._direction = self._candidate_direction = self._direction_frames = 0
        self._repair_since = self._recovery_since = None
        self._recovery_required = True
        self._current_kind = 'stream_reset'
        self._current_request = 0.
        self._observation_facts = {}
        if self.strict_layout:
            self._last_force = None
            self._force_history.clear()
            self._force_tripped = False
            self._force_unstable_until = None

    def _pause(self, reason):
        self._stable_since = self._recovery_since = None
        self._warming_mean = self._warming_noise = None
        self._direction = self._candidate_direction = self._direction_frames = 0
        self._recovery_required = True
        self._current_request = 0.
        self._current_kind = reason

    def _compatible(self, obs):
        return self._compatibility_error(obs) is None

    def _compatibility_error(self, obs):
        f = self.config['feature']
        if obs.version != (f['registration_version'], f['window_version'],
                           self.config['geometry']['calibration_version']):
            return 'registration_window_or_calibration_mismatch'
        if obs.echo_feature_version != ECHO_FEATURE_VERSION or obs.nearfield_echo is None:
            return 'nearfield_echo_feature_missing_or_incompatible'
        if obs.timestamp_semantics != 'effective_image_time':
            return 'timestamp_semantics_mismatch'
        if self.strict_layout:
            if obs.region_confidence is None or len(obs.region_confidence) != self.feature_config.region_count:
                return 'region_count_mismatch'
            if obs.region_feature_version != REGION_FEATURE_VERSION:
                return 'region_feature_version_mismatch'
            if obs.region_layout_version != f['region_layout_version']:
                return 'region_layout_mismatch'
            if not np.array_equal(obs.region_edges, self.feature_config.region_edges):
                return 'region_edges_mismatch'
            if not np.all(obs.region_valid):
                return 'invalid_region_columns'
        if len(obs.nearfield_echo) != f['config']['width']:
            return 'echo_width_mismatch'
        if not np.array_equal(obs.nearfield_echo_depth, self.settings['near_depth']):
            return 'echo_depth_mismatch'
        if not np.all(obs.nearfield_echo_valid):
            return 'invalid_echo_columns'
        return None

    @staticmethod
    def _runs(mask, minimum):
        delta = np.diff(np.r_[False, mask, False].astype(int))
        return [(int(a), int(b)) for a, b in zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1))
                if b-a >= minimum]

    def _classify(self, obs):
        echo = np.asarray(obs.nearfield_echo, dtype=float)
        half = len(echo)//2
        reference_level = max(float(np.median(echo[:half])), float(np.median(echo[half:])))
        reference = (self.baseline if self.baseline is not None
                     else np.full_like(echo, reference_level))
        noise = self.noise if self.noise is not None else np.full_like(echo, 1/255.)
        loss = np.maximum(reference-echo-self.settings['noise_sigma']*noise, 0.)
        relative = np.divide(loss, np.maximum(reference, self.settings['absolute_echo_floor']))
        dark = ((echo < self.settings['echo_ratio']*reference) & (loss > 0)) | (
            echo < self.settings['absolute_echo_floor'])
        runs = self._runs(dark, self.settings['minimum_dark_columns'])
        mask = np.zeros_like(dark)
        for a, b in runs: mask[a:b] = True
        x = (2*np.arange(len(echo))+1-len(echo))/len(echo)
        deficit = np.where(mask, np.maximum(relative, np.maximum(
            self.settings['absolute_echo_floor']-echo, 0.)/self.settings['absolute_echo_floor']), 0.)
        left = bool(np.any(mask & (x < -.2)))
        right = bool(np.any(mask & (x > .2)))
        global_low = bool(np.median(echo) < self.settings['absolute_echo_floor'] or mask.mean() >= .65)
        kind = 'healthy'
        direction = 0
        if global_low: kind = 'uniform_low_echo'
        elif left and right: kind = 'bilateral_missing_echo'
        elif runs:
            centroid = float(np.dot(deficit, x)/max(float(deficit.sum()), 1e-12))
            if abs(centroid) <= .2: kind = 'ambiguous_missing_echo'
            else:
                direction = -int(self.config['geometry']['image_x_sign'])*int(np.sign(centroid))
                kind = 'unilateral_missing_echo'
        # Strong confidence evidence on the opposite side creates uncertainty.
        # Deep-only confidence loss cannot create a rocking request by itself.
        confidence_conflict = False
        q = obs.region_confidence
        if direction and q is not None and np.all(obs.region_valid):
            q = np.asarray(q)
            qref = max(np.median(q[:len(q)//2]), np.median(q[len(q)//2:]))
            qloss = np.maximum(qref-q, 0.)
            qx = (2*np.arange(len(q))+1-len(q))/len(q)
            qmoment = float(np.dot(qloss, qx))
            emoment = float(np.dot(deficit, x))
            confidence_conflict = bool(qloss.max() > .15 and qmoment*emoment < 0)
            if confidence_conflict:
                kind, direction = 'conflicting_spatial_evidence', 0
        # Continuous spatial moment: a two-column hole has a small request,
        # with no fixed step or remembered angular peak.
        moment = float(np.dot(deficit, x)/np.dot(x, x))
        knee = float(self.settings.get('echo_moment_cubic_knee', 0.) or 0.)
        compress = float(self.settings.get('echo_moment_compress', 0.) or 0.)
        if knee > 0. and moment != 0.:
            # |m| below the knee shrinks as m^3. A wide edge at or above the
            # knee keeps its linear moment. tanh(m/0.06) did the opposite.
            weight = min(1., abs(moment) / knee)
            moment = math.copysign(abs(moment) * weight * weight, moment)
        elif compress > 0.:
            gain = math.tanh(abs(moment) / compress)
            moment = math.copysign(gain, moment) if moment != 0. else 0.
        omega = -self.config['geometry']['image_x_sign']*self.pad['max_visual_omega_rad_s']*moment
        omega = float(np.clip(omega, -self.pad['max_visual_omega_rad_s'], self.pad['max_visual_omega_rad_s']))
        self._observation_facts = dict(
            nearfield_echo=echo.tolist(), echo_reference=reference.tolist(), echo_noise=noise.tolist(),
            echo_feature_version=obs.echo_feature_version, nearfield_echo_depth=obs.nearfield_echo_depth.tolist(),
            dark_runs=[dict(start=a, stop=b, width_columns=b-a,
                            width_fraction=(b-a)/len(echo)) for a, b in runs],
            dark_columns=np.flatnonzero(mask).tolist(), echo_deficit=deficit.tolist(),
            echo_classification=kind, confidence_conflict=confidence_conflict,
            region_quality=None if obs.region_confidence is None else obs.region_confidence.tolist(),
            quality_raw=obs.quality.tolist(), baseline_ready=self.baseline is not None,
        )
        return kind, direction, omega, echo

    def _update_baseline(self, echo, stamp, previous_stamp):
        dt = 0. if previous_stamp is None else max(0., stamp-previous_stamp)
        if self.baseline is None:
            if self._stable_since is None:
                self._stable_since = stamp
                self._warming_mean = echo.copy()
                self._warming_noise = np.full_like(echo, 1/255.)
            else:
                weight = min(1., dt/max(self.settings['baseline_hold_s'], 1e-6))
                difference = np.abs(echo-self._warming_mean)
                self._warming_noise += weight*(difference-self._warming_noise)
                self._warming_mean += weight*(echo-self._warming_mean)
            if stamp-self._stable_since >= self.settings['baseline_hold_s']-1e-9:
                self.baseline = self._warming_mean.copy()
                self.noise = np.maximum(self._warming_noise, 1/255.)
        else:
            weight = 1.-math.exp(-dt/self.settings['baseline_tau_s'])
            difference = np.abs(echo-self.baseline)
            self.noise = np.maximum(1/255., self.noise+weight*(difference-self.noise))
            self.baseline += weight*(echo-self.baseline)

    def update(self, *, now_s, dt_s, contact_present, force_n, stick_speed_m_s):
        now = float(now_s)
        if (not math.isfinite(now) or not math.isfinite(dt_s) or dt_s <= 0
                or self._last_now is not None and now < self._last_now):
            self._pause('invalid_clock')
            return self._result(now, False, 0., 'invalid_clock', None, False)
        self._last_now = now
        obs = self.receiver.observation
        fresh = False
        if obs is not None:
            previous = self.latest.observation
            if obs is not previous and self.latest.accept(obs):
                fresh = True
                if previous is not None and (obs.source_id != previous.source_id or obs.version != previous.version
                        or self.strict_layout and (obs.region_layout_version != previous.region_layout_version
                            or not np.array_equal(obs.region_edges, previous.region_edges))):
                    self._reset_stream()
                    self.kf.reset()
                # Validate before both KF ingestion and raw echo classification.
                if not self.strict_layout or (self._compatible(obs)
                        and 0 <= now-obs.effective_time_s <= self.pad['max_image_age_s']
                        and obs.received_time_s <= now):
                    self.kf.ingest(obs, now)
            obs = self.latest.observation
        age = None if obs is None else now-obs.effective_time_s
        if (obs is None or not self._compatible(obs) or age is None
                or not 0 <= age <= self.pad['max_image_age_s'] or obs.received_time_s > now):
            reason = 'image_unavailable' if obs is None else 'image_stale' if age is not None and age > self.pad['max_image_age_s'] else 'image_invalid'
            self._pause(reason)
            return self._result(now, False, 0., reason, obs, fresh)
        force_valid = bool(math.isfinite(force_n) and self.pad['cop_min_force_n'] <= force_n
                           < self.pad.get('force_gate_end_n', 4.5))
        step = 0. if self._last_force is None else abs(force_n - self._last_force)
        enter = float(self.settings['force_step_max_n'])
        leave = min(enter, float(self.settings.get('force_step_release_n', enter)))
        hold_s = float(self.settings.get('force_unstable_hold_s', 0.) or 0.)
        self._last_force = float(force_n) if math.isfinite(force_n) else None
        self._force_history.append((now, float(force_n)))
        while self._force_history and self._force_history[0][0] < now-self.settings['force_window_s']:
            self._force_history.popleft()
        force_range = float(np.ptp([f for _, f in self._force_history]))
        window = float(self.settings['force_window_range_n'])
        over_enter = (not force_valid) or step > enter
        over_leave = (not force_valid) or step > leave
        if over_enter:
            self._force_tripped = True
            if hold_s > 0.:
                self._force_unstable_until = now + hold_s
            force_stable = False
        elif self._force_unstable_until is not None and now < self._force_unstable_until:
            force_stable = False
        elif self._force_tripped and over_leave:
            force_stable = False
        else:
            self._force_tripped = False
            self._force_unstable_until = None
            force_stable = True
        baseline_force_stable = force_stable and force_range <= window
        if not baseline_force_stable:
            self._stable_since = None
            self._warming_mean = self._warming_noise = None
        if not contact_present or not force_stable:
            reason = 'contact_unavailable' if not contact_present else 'force_unstable'
            self._pause(reason)
            return self._result(now, False, 0., reason, obs, fresh)
        if fresh:
            previous_stamp = self._last_echo_time
            stamp = obs.effective_time_s
            if previous_stamp is not None and stamp-previous_stamp > self.pad['max_image_age_s']:
                self._pause('image_gap')
            self._last_echo_time = stamp
            kind, direction, omega, echo = self._classify(obs)
            self._current_kind = kind
            if kind != 'healthy':
                self._stable_since = self._recovery_since = None
                self._warming_mean = self._warming_noise = None
                self._recovery_required = True
                if self._repair_since is None: self._repair_since = now
                if direction and direction == self._candidate_direction:
                    self._direction_frames += 1
                else:
                    self._candidate_direction = direction
                    self._direction_frames = 1 if direction else 0
                self._direction = direction if self._direction_frames >= self.settings['direction_confirm_frames'] else 0
                self._current_request = omega if self._direction else 0.
            else:
                self._direction = self._candidate_direction = self._direction_frames = 0
                self._current_request = 0.
                if self.baseline is None and baseline_force_stable:
                    self._update_baseline(echo, stamp, previous_stamp)
                elif self.baseline is not None and not self._recovery_required and baseline_force_stable:
                    self._update_baseline(echo, stamp, previous_stamp)
                if (self.strict_layout or self.baseline is not None) and self._recovery_required:
                    if self._recovery_since is None: self._recovery_since = stamp
                    if stamp-self._recovery_since >= self.pad['recover_hold_s']-1e-9:
                        self._recovery_required = False
                        self._recovery_since = self._repair_since = None
        kind = self._current_kind
        if kind == 'unilateral_missing_echo':
            elapsed = 0. if self._repair_since is None else now-self._repair_since
            speed = abs(float(stick_speed_m_s))
            alpha = min(1., self.pad['repair_speed_cap_m_s']/max(speed, 1e-12))
            if elapsed >= self.pad['repair_pause_after_s']: alpha = 0.
            if self.strict_layout and not self._direction:
                alpha = 0.
            return self._result(now, bool(self._direction), alpha,
                                kind if self._direction else 'direction_unconfirmed', obs, fresh)
        if kind != 'healthy':
            return self._result(now, False, 0., kind, obs, fresh)
        if self.baseline is None and not self.strict_layout:
            return self._result(now, False, 0., 'baseline_warming', obs, fresh)
        if self._recovery_required:
            return self._result(now, True, 0., 'recovery_confirming', obs, fresh)
        return self._result(now, True, 1., 'healthy', obs, fresh)

    def _result(self, now, valid, alpha, reason, obs, fresh):
        prediction = self.kf.predict(now) if math.isfinite(now) else None
        force_gate = 1.
        if self.strict_layout:
            start, end = self.pad['force_gate_start_n'], self.pad['force_gate_end_n']
            force_gate = (0. if self._last_force is None else
                          float(np.clip((end-self._last_force)/(end-start), 0., 1.)))
        facts = dict(self._observation_facts,
            reason=reason, valid=bool(valid), frame_seq=None if obs is None else obs.frame_seq,
            source_id=None if obs is None else obs.source_id,
            effective_time_s=None if obs is None else obs.effective_time_s,
            received_time_s=None if obs is None else obs.received_time_s,
            image_compatibility_reason=None if obs is None else self._compatibility_error(obs),
            echo_feature_version=None if obs is None else obs.echo_feature_version,
            image_age_s=None if obs is None else now-obs.effective_time_s,
            observation_fresh=bool(fresh), baseline_ready=self.baseline is not None,
            direction_confirmed=bool(self._direction), direction_frames=self._direction_frames,
            recovery_required=self._recovery_required,
            repair_elapsed_s=None if self._repair_since is None else now-self._repair_since,
            # Force gamma is applied once by the shared mechanical allocator.
            alpha_target=float(alpha), omega_request_rad_s=self._current_request if valid else 0.,
            force_gate=force_gate,
            kf_reason=None if prediction is None else prediction.reason,
            kf_age_s=None if prediction is None else prediction.age_s,
            quality_predicted=None if prediction is None or prediction.quality_raw is None else prediction.quality_raw.tolist())
        if self.strict_layout:
            facts.update(kf_valid=bool(prediction is not None and prediction.valid),
                region_count=None if obs is None or obs.region_confidence is None else len(obs.region_confidence),
                region_layout_version=None if obs is None else obs.region_layout_version,
                region_feature_version=None if obs is None else obs.region_feature_version,
                region_edges=None if obs is None or obs.region_edges is None else obs.region_edges.tolist(),
                region_quality_raw=None if obs is None or obs.region_confidence is None else obs.region_confidence.tolist(),
                region_valid=None if obs is None or obs.region_valid is None else obs.region_valid.tolist(),
                timestamp_semantics=None if obs is None else obs.timestamp_semantics,
                quality_auxiliary='weighted_q25', nearfield_echo_width=self.feature_config.width,
                nearfield_echo=None if obs is None or obs.nearfield_echo is None else obs.nearfield_echo.tolist(),
                nearfield_echo_valid=None if obs is None or obs.nearfield_echo_valid is None else obs.nearfield_echo_valid.tolist(),
                nearfield_echo_depth=None if obs is None or obs.nearfield_echo_depth is None else obs.nearfield_echo_depth.tolist())
        self.last_facts = facts
        return PadEvidenceRequest(facts['omega_request_rad_s'], float(alpha), bool(valid), reason, facts)
