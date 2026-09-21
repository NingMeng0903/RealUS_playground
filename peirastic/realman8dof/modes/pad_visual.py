"""Pad-mode visual ωy guide, plus UltraPoC α and scan-style contact fusion.

The pad never owns inner publication. ``pad_visual.progress`` supplies the
region first-moment request and preferred-scan α; HybridTffOuter runs the
same delay-KF COP QP as the phantom Lissajous scan.
"""
from __future__ import annotations

import time
from functools import wraps
from pathlib import Path

import numpy as np

from peirastic.contact_qp.confidence_fusion import ConfidenceAngularTask, FEATURE_VERSION
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.region_visual import region_visual_request
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config
from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
from .pad_scan_config import default_pad_scan_config_path, load_pad_scan_config

DEFAULT_PAD_VISUAL_CONFIG = default_pad_scan_config_path()
PAD_HYBRID_LAWS = ("legacy", "ac2d", "ultrapoc")


def pad_hybrid_payload(
    law="legacy",
    *,
    desired_z=None,
    label=None,
    log_path=None,
    force_profile="icra",
    contact_config=None,
) -> dict:
    """L3 TRACK_HYBRID payload for pad speed-mode scanning.

    ``legacy`` is the old pad hybrid: TFF tool-Z plus optional icra
    torque_tilt My→ωy. ``ac2d`` is the comparison Z+My law.
    ``ultrapoc`` keeps that 994742bb split for install, then replaces the
    additive My⊕image sum with the scan delay-KF COP / α / visual QP.
    ``log_path`` is a jsonl path or a dict with at least ``path``.
    """

    name = str(law or "legacy").lower()
    if name not in PAD_HYBRID_LAWS:
        raise ValueError(f"pad hybrid law must be one of {PAD_HYBRID_LAWS}, got {law!r}")
    from peirastic.scan_path import SCAN_FORCE_AXES, force_profile as scan_force_profile

    payload = {
        "reference": "pad",
        "use_tff_split": True,
        "filter": False,
        "label": label or ("track_hybrid_pad_ultrapoc" if name == "ultrapoc" else "track_hybrid_pad"),
    }
    if desired_z is not None:
        payload["desired_z"] = float(desired_z)
    if log_path is not None:
        payload["pad_log"] = log_path
    if name == "legacy":
        payload.update(scan_force_profile(force_profile))
        return payload
    payload["force_axes"] = list(SCAN_FORCE_AXES)
    if name == "ac2d":
        payload["law"] = "ac2d"
        return payload
    payload.update(scan_force_profile("icra"))
    config = load_pad_scan_config(contact_config)
    payload["pad_visual"] = dict(config, progress=True)
    payload["pad_scan_config"] = config
    return payload


def default_pad_log_path(repo=None, *, now=None) -> Path:
    root = Path(repo) if repo is not None else Path.cwd()
    stamp = time.strftime("%Y%m%d_%H%M%S") if now is None else time.strftime(
        "%Y%m%d_%H%M%S", time.localtime(now)
    )
    return (root / "apps" / "logs" / "pad_hybrid" / f"{stamp}.jsonl")


def allocate_pad_log_path(requested, *, now=None) -> Path:
    """Pick a not-yet-existing jsonl path. Does not create the file."""

    stamp = time.strftime("%Y%m%d_%H%M%S") if now is None else time.strftime(
        "%Y%m%d_%H%M%S", time.localtime(now)
    )
    text = "" if requested is None else str(requested).strip()
    if text.lower() in ("", "auto", "1", "true"):
        raw = Path("apps/logs/pad_hybrid") / f"{stamp}.jsonl"
    else:
        raw = Path(text).expanduser()
        if raw.suffix.lower() != ".jsonl":
            raw = raw / f"{stamp}.jsonl"
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    raw = raw.resolve()
    if not raw.exists():
        return raw
    for index in range(1, 10000):
        candidate = raw.with_name(f"{raw.stem}_{index:03d}{raw.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"could not allocate pad log path from {raw}")


def _pad_log_request(payload) -> dict | None:
    raw = dict(payload or {}).get("pad_log")
    if raw is None or raw is False or raw == "":
        return None
    if raw is True or (isinstance(raw, str) and raw.lower() in ("auto", "1", "true")):
        return {"path": "auto"}
    if isinstance(raw, str):
        return {"path": raw}
    if isinstance(raw, dict):
        path = raw.get("path")
        if path is None or path is False or path == "":
            return None
        return dict(raw)
    return None


def wrap_pad_hybrid_log(phase, payload):
    """Record pad-hybrid samples off the control thread. No-op without pad_log."""

    request = _pad_log_request(payload)
    if request is None:
        return phase
    from peirastic.realman8dof.modes.contact_recording import ContactRecordSink, clock_metadata

    path = allocate_pad_log_path(request.get("path"))
    sink = ContactRecordSink(path)
    sink.emit(
        "study_start",
        record_kind="pad_hybrid",
        law=dict(payload or {}).get("law"),
        label=dict(payload or {}).get("label"),
        pad_visual=dict(payload or {}).get("pad_visual"),
        pad_scan_config=dict(payload or {}).get("pad_scan_config"),
        config_version=(dict(payload or {}).get("pad_scan_config") or {}).get("config_version"),
        pad_log=request,
        resolved_path=str(path),
        **clock_metadata(),
    )
    baseline = phase.outer
    original_sample = baseline.sample
    control_id = 0
    evidence_key = None
    previous_exit = phase.on_exit

    @wraps(original_sample)
    def sample(t_s, current_pose, f_ext, **kwargs):
        nonlocal control_id, evidence_key
        twist = original_sample(t_s, current_pose, f_ext, **kwargs)
        control_id += 1
        guide = getattr(baseline, "pad_visual", None)
        facts = dict(getattr(guide, "last_facts", None) or {})
        facts.update(getattr(baseline, "last_facts", None) or {})
        key = (facts.get('source_id'), facts.get('frame_seq'), facts.get('effective_time_s'))
        if key != evidence_key and key[1] is not None:
            evidence_key = key
            sink.emit('pad_image_evidence', record_kind='pad_hybrid', control_id=control_id,
                      **dict(getattr(guide, 'last_facts', None) or {}))
        observation = None
        receiver = getattr(guide, "_receiver", None) if guide is not None else None
        if receiver is not None:
            observation = getattr(receiver, "observation", None)
        image_age = facts.get("image_age_s")
        if image_age is None and observation is not None:
            try:
                image_age = time.monotonic() - float(observation.effective_time_s)
            except (TypeError, ValueError):
                image_age = None
        region_quality = facts.get("region_quality")
        if region_quality is None and observation is not None:
            quality = getattr(observation, "region_confidence", None)
            if quality is not None:
                region_quality = np.asarray(quality, dtype=float)
        wrench = np.asarray(f_ext, dtype=float).reshape(-1)
        stick = np.asarray(getattr(baseline, "last_path_twist_raw", np.zeros(6)), dtype=float).reshape(6)
        scaled = np.asarray(getattr(baseline, "last_path_twist", np.zeros(6)), dtype=float).reshape(6)
        command = np.asarray(twist, dtype=float).reshape(6)
        sink.emit(
            "control_sample",
            record_kind="pad_hybrid",
            control_id=control_id,
            reference_s=t_s,
            contact=getattr(baseline, "last_contact_present", kwargs.get("contact")),
            fz=float(wrench[2]) if wrench.size > 2 else None,
            my=float(wrench[4]) if wrench.size > 4 else None,
            wrench_tool=wrench[:6] if wrench.size >= 6 else wrench,
            v_stick=stick,
            v_scaled=scaled,
            v_cmd=command,
            command_stage='outer_candidate_before_ik_and_publication',
            stick_speed_m_s=float(np.linalg.norm(stick[:3])),
            alpha=getattr(baseline, "last_pad_alpha", None),
            alpha_target=facts.get("alpha_target"),
            omega_moment=getattr(baseline, "last_omega_y", None),
            omega_visual=facts.get("visual_omega_rad_s"),
            omega_visual_target=facts.get("visual_request_rad_s"),
            omega_cmd=float(command[4]),
            command_semantics=(
                "delay_kf_cop_v1"
                if (facts.get("allocation_diagnostics") or {}).get("allocation_policy") == "delay_kf_cop_v1"
                else "legacy_additive"
            ),
            frame_seq=facts.get("frame_seq"),
            effective_image_time_s=facts.get("effective_time_s"),
            cop_m=facts.get("cop_m"),
            cop_valid=facts.get("cop_valid"),
            cop_reason=facts.get("cop_reason"),
            tau_y=getattr(baseline, "last_tau_y", None),
            tau_error_y=getattr(baseline, "last_tau_error_y", None),
            visual_reason=facts.get("visual_reason"),
            visual_policy=facts.get("visual_policy"),
            echo_reason=facts.get("echo_reason"),
            echo_omega_rad_s=facts.get("echo_omega_rad_s"),
            kf_valid=facts.get("kf_valid"),
            kf_reason=facts.get("kf_reason"),
            kf_age_s=facts.get("kf_age_s"),
            image_age_s=image_age,
            region_quality=region_quality,
            region_count=facts.get("region_count"),
            region_edges=facts.get("region_edges"),
            region_layout_version=facts.get("region_layout_version"),
            region_feature_version=facts.get("region_feature_version"),
            region_quality_raw=facts.get("region_quality_raw"),
            quality_predicted=facts.get("quality_predicted"),
            source_id=facts.get("source_id"),
            received_image_time_s=facts.get("received_time_s"),
            timestamp_semantics=facts.get("timestamp_semantics"),
            evidence_valid=facts.get("evidence_valid", facts.get("valid")),
            echo_classification=facts.get("echo_classification"),
            image_compatibility_reason=facts.get("image_compatibility_reason"),
            allocation_status=facts.get("allocation_status", (facts.get('allocation_diagnostics') or {}).get('solver_status')),
            allocation_success=facts.get('allocation_success'),
            allocation_diagnostics=facts.get("allocation_diagnostics"),
            normal_shortfall=facts.get("normal_shortfall", (facts.get('allocation_diagnostics') or {}).get('normal_shortfall_m_s')),
            config_version=facts.get("config_version"),
            max_deficit=facts.get("max_deficit"),
            quality_reference=facts.get("quality_reference"),
            deficit_mode=facts.get("deficit_mode"),
            bad_region_indices=facts.get("bad_region_indices"),
            pose_tcp_base=current_pose,
        )
        return twist

    baseline.sample = sample
    # The loop caches explicit keyword names for feedback injection.
    baseline.__dict__.pop("_sample_params", None)
    baseline.pad_log_sink = sink
    lifecycle = getattr(baseline, "command_lifecycle", None)
    if lifecycle is not None:
        lifecycle.record_sink = sink
        sink.emit("pad_configuration", **lifecycle.configuration_snapshot())

    def on_exit():
        try:
            sink.emit("stop", record_kind="pad_hybrid", control_id=control_id, reason="phase_exit")
            if previous_exit is not None:
                previous_exit()
        finally:
            sink.close(wait=False)

    phase.on_exit = on_exit
    return phase


def _explicit_visual_request(payload: dict) -> bool:
    visual = payload.get("pad_visual")
    return bool(
        visual is True
        or (isinstance(visual, dict) and visual.get("enabled") is not False)
    )


def _progress_requested(payload: dict) -> bool:
    visual = payload.get("pad_visual")
    return bool(isinstance(visual, dict) and visual.get("progress") is True)


def resolve_pad_visual_config(payload: dict | None) -> dict | None:
    pay = dict(payload or {})
    visual = pay.get("pad_visual")
    if visual is False or visual is None:
        return None
    if isinstance(visual, dict) and visual.get("enabled") is False:
        return None
    if isinstance(visual, dict) and (visual.get("geometry") or visual.get("feature_endpoint")
                                     or visual.get("feature") or visual.get("qp")):
        return dict(visual)
    if visual is True or isinstance(visual, dict):
        if DEFAULT_PAD_VISUAL_CONFIG.is_file():
            return load_pad_scan_config(DEFAULT_PAD_VISUAL_CONFIG)
    return None


def attach_pad_visual(payload: dict | None, tilt_cfg, *, force_law=None,
                      receiver=None) -> "PadVisualGuide | None":
    """Return a live guide, or None when pad should stay moment-only."""
    pay = dict(payload or {})
    config = resolve_pad_visual_config(pay)
    tilt_enabled = bool(getattr(tilt_cfg, "enabled", False))
    if not tilt_enabled:
        if _explicit_visual_request(pay):
            raise ValueError(
                "pad visual ωy requires tilt-owned tool-Y; selection[4]>=1 disables the injection point"
            )
        return None
    if config is None:
        return None
    try:
        return PadVisualGuide(
            config, tilt_cfg, receiver=receiver, progress=_progress_requested(pay),
        )
    except ValueError:
        if _explicit_visual_request(pay):
            raise
        return None


class PadVisualGuide:
    """Inject image ωy when a fresh frame is present; optionally publish α."""

    def __init__(self, config, tilt_cfg, *, receiver=None, progress=False):
        self.config = dict(config)
        self.progress = bool(progress)
        self.geometry, _ = calibrated_geometry(self.config)
        qp = dict(self.config.get("qp") or {})
        feature = dict(self.config.get("feature") or {})
        raw_feature = feature.get("config")
        windows = FeatureConfig(**raw_feature).lateral_windows if raw_feature else FeatureConfig().lateral_windows
        rows = window_rows(self.geometry, windows)
        repair = qp.get("differential_repair") or {}
        deadband = float(repair.get("balance_deadband", 0.03)) if isinstance(repair, dict) else 0.03
        c_min = float(feature.get("c_min", qp.get("c_min", 0.5)))
        self.c_min = c_min
        self.repair_speed_m_s = float(qp.get("repair_speed_m_s", 0.002))
        self.progress_loss_gain = float(qp.get("progress_loss_gain", 0.75))
        self.alpha_min = float(qp.get("alpha_min", 0.25))
        self.alpha_min_at_quality = qp.get("alpha_min_at_quality")
        scan_max = qp.get("max_visual_omega_rad_s")
        self.visual_hold_s = 0.0
        self.visual_hold_floor = 0.0
        self._hold_omega = 0.0
        self._hold_until_s = None
        if progress:
            # Relative good-side deficit: ω and α still use the first-moment
            # plant, but ref is this frame's better half, not phantom c_min.
            # Uniform tissue/gel stays on My*=0. No peak hold (181143 bounced).
            raw_ref = qp.get("pad_rocking_reference")
            self.rocking_reference = None if raw_ref is None else float(raw_ref)
            self.deficit_mode = str(qp.get("pad_deficit_mode", "relative")).lower()
            self.relative_alpha_span = float(qp.get("pad_relative_alpha_span", 0.15))
            self.balance_deadband = deadband
            self.omega_step_deficit = float(qp.get("pad_omega_step_deficit", 0.08))
            self.repair_speed_m_s = float(qp.get("pad_repair_speed_m_s", self.repair_speed_m_s))
            self.alpha_min = float(qp.get("pad_alpha_min", 0.25))
            self.alpha_min_at_quality = qp.get(
                "pad_alpha_min_at_quality", qp.get("alpha_min_at_quality")
            )
            self.ramp_s = float(qp.get("pad_alpha_ramp_s", 0.12))
            self.ramp_up_s = float(qp.get("pad_alpha_ramp_up_s", 0.30))
            self.omega_step_rad_s = float(qp.get("pad_omega_step_rad_s", 0.06))
            self.brake_on_sided = bool(qp.get("pad_brake_on_sided", False))
            pad_max = qp.get("pad_max_visual_omega_rad_s")
            self.max_visual_omega_rad_s = float(pad_max) if pad_max is not None else 0.16
            self.visual_hold_s = float(qp.get("pad_visual_hold_s", 0.0))
            self.visual_hold_floor = float(qp.get("pad_visual_hold_omega_rad_s", 0.03))
        else:
            self.rocking_reference = None
            self.deficit_mode = "absolute"
            self.relative_alpha_span = 0.15
            self.balance_deadband = 0.0
            self.omega_step_deficit = 0.0
            self.max_visual_omega_rad_s = scan_max
            self.ramp_s = float(qp.get("pad_alpha_ramp_s", qp.get("path_ramp_s", 0.4)))
            self.ramp_up_s = self.ramp_s
            self.omega_step_rad_s = 0.0
            self.brake_on_sided = False
        self.max_image_age_s = float(qp.get("max_image_age_s", 0.3))
        if not np.isfinite(self.ramp_s) or self.ramp_s <= 0.0:
            raise ValueError("pad alpha ramp_s must be finite and positive")
        self.omega_accel = float(tilt_cfg.a_max)
        self.omega_vmax = float(tilt_cfg.vmax_rad_s)
        self.alpha_cmd = 1.0
        self.omega_cmd = 0.0
        self.alpha_target = 1.0
        self.omega_target = 0.0
        self._image_s = None
        self.task = ConfidenceAngularTask(
            mass=float(tilt_cfg.mass),
            damping=float(tilt_cfg.damping),
            repair_speed_m_s=self.repair_speed_m_s,
            lever_m=abs(float(rows[0, 4] - rows[2, 4])),
            image_x_sign=int(self.geometry.image_x_sign),
            deadband=deadband,
            c_min=c_min,
            max_velocity=float(tilt_cfg.vmax_rad_s),
            max_acceleration=float(tilt_cfg.a_max),
        )
        self._receiver = receiver
        self._owns_receiver = receiver is None
        self._endpoint = self.config.get("feature_endpoint")
        self._seeded = False
        self._image_kf = None
        self._kf_last_key = None
        self._last_region_obs = None
        if progress:
            kf_block = dict(self.config.get("image_kf") or {})
            if kf_block:
                from peirastic.contact_qp.delayed_kf import DelayConfidenceKF, KfConfig
                if raw_feature:
                    kf_block["channels"] = int(FeatureConfig(**raw_feature).region_count)
                self._image_kf = DelayConfidenceKF(KfConfig(**kf_block))
        self.last_facts = {}
        self.last_alpha = 1.0 if self.progress else None
        self.last_visual_omega = 0.0 if self.progress else None
        self.evidence_valid = False
        self._pad_evidence = None
        if progress and (self.config.get('pad') or {}).get('policy') == 'pad_scan_v1':
            # Handheld stick is the path. α starts at 1; image may slow it.
            # Echo evidence must not park the sticks at zero waiting for a
            # baseline, and it does not own the region KF.
            self.ramp_s = self.config['pad']['alpha_ramp_down_s']
            self.ramp_up_s = self.config['pad']['alpha_ramp_up_s']

    def _ensure_receiver(self):
        if self._receiver is not None or not self._owns_receiver or not self._endpoint:
            return
        from .contact_recording import FeatureReceiver
        self._receiver = FeatureReceiver(self._endpoint)

    def _fresh_observation(self, now_s: float):
        self._ensure_receiver()
        observation = getattr(self._receiver, "observation", None) if self._receiver is not None else None
        if observation is None:
            return None
        try:
            if not observation.fresh(now_s, self.max_image_age_s):
                return None
        except (TypeError, ValueError):
            return None
        return observation

    def _region_observation(self, now_s: float):
        """Age-gate regional evidence without the LCR fresh() flags.

        ``ContactObservation.fresh`` requires left/right LCR validity. Pad
        UltraPoC only needs the 10-region map; a flickering LCR bit used to
        drop the image and snap α back to 1.
        """

        self._ensure_receiver()
        observation = getattr(self._receiver, "observation", None) if self._receiver is not None else None
        if observation is None or getattr(observation, "region_confidence", None) is None:
            return None
        try:
            age = float(now_s) - float(observation.effective_time_s)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(age) or age < 0.0 or age > self.max_image_age_s:
            return None
        return observation

    def _progress_evidence(self, now_s, snapshot):
        """Ingest region frames and replay quality to now. No CAM-QP."""

        kf_facts = dict(kf_valid=False, kf_reason=None, kf_age_s=None, kf_quality=None)
        if snapshot is not None:
            self._last_region_obs = snapshot
        if self._image_kf is not None:
            batch = []
            drain = getattr(self._receiver, "drain_observations", None)
            if callable(drain):
                try:
                    batch.extend(drain() or ())
                except TypeError:
                    pass
            if snapshot is not None:
                batch.append(snapshot)
            for obs in batch:
                if obs is None or getattr(obs, "region_confidence", None) is None:
                    continue
                try:
                    key = (getattr(obs, "frame_seq", None), float(obs.effective_time_s))
                except (TypeError, ValueError):
                    continue
                if key == self._kf_last_key:
                    continue
                if self._image_kf.ingest(obs, now_s):
                    self._kf_last_key = key
                    self._last_region_obs = obs
            prediction = self._image_kf.predict(now_s)
            kf_facts.update(
                kf_valid=bool(prediction.valid),
                kf_reason=prediction.reason,
                kf_age_s=prediction.age_s,
                kf_quality=(
                    None if prediction.quality_raw is None
                    else np.clip(prediction.quality_raw, 0.0, 1.0).tolist()
                ),
            )
            meta = snapshot if snapshot is not None else self._last_region_obs
            if prediction.valid and prediction.quality_task is not None and meta is not None:
                return prediction.quality_task, meta, kf_facts
        if snapshot is not None:
            return snapshot.region_confidence, snapshot, kf_facts
        return None, None, kf_facts

    @staticmethod
    def _slew(value, target, rate, dt_s):
        step = float(rate) * float(dt_s)
        return float(value) + float(np.clip(float(target) - float(value), -step, step))

    def _clear_visual_hold(self):
        self._hold_omega = 0.0
        self._hold_until_s = None

    def _apply_visual_hold(self, now_s, reason):
        """Keep rocking through the ~0.23 s image lag so a recovered frame cannot punch-and-release."""

        if self.visual_hold_s <= 0.0:
            return reason
        target = float(self.omega_target)
        floor = float(self.visual_hold_floor)
        if abs(target) >= floor:
            if self._hold_omega * target < 0.0:
                self._hold_omega = target
            elif abs(target) >= abs(self._hold_omega):
                self._hold_omega = target
            self._hold_until_s = float(now_s) + self.visual_hold_s
            return reason
        if (self._hold_until_s is not None and float(now_s) < float(self._hold_until_s)
                and self._hold_omega * target >= 0.0 and abs(self._hold_omega) >= floor):
            self.omega_target = float(self._hold_omega)
            return "visual_hold"
        self._clear_visual_hold()
        return reason

    def preview(self, *, pose, dt_s, contact_present, tilt, now_s=None,
                force_n=float('nan'), stick_speed_m_s=0.0) -> float | None:
        if tilt is None:
            if self.progress:
                self.last_alpha = 1.0
                self.last_visual_omega = 0.0
            else:
                self.last_alpha = None
            return None
        now = float(now_s if now_s is not None else time.monotonic())
        if self.progress:
            visual = self._preview_progress(
                self._region_observation(now),
                contact_present=bool(contact_present),
                now_s=now,
                dt_s=float(dt_s),
            )
            if (self.config.get("pad") or {}).get("policy") == "pad_scan_v1":
                self._overlay_echo_visual(
                    now_s=now, dt_s=float(dt_s), contact_present=bool(contact_present),
                    force_n=force_n, stick_speed_m_s=stick_speed_m_s)
            return visual
        observation = self._fresh_observation(now)
        self.last_alpha = None
        if observation is None:
            self._seeded = False
            self.last_facts = dict(visual_reason="image_absent", visual_policy="pad_moment_fallback")
            return None
        rotation = rotation_from_pose(pose)
        if not self._seeded:
            self.task.omega_base = rotation @ np.array([0.0, float(tilt._w), 0.0])
            self._seeded = True
        try:
            omega, facts = self.task.preview(
                observation,
                rotation_base_tcp=rotation,
                dt_s=float(dt_s),
                force_gate=1.0 if contact_present else 0.0,
                enabled=bool(contact_present),
            )
        except ValueError:
            self._seeded = False
            self.last_facts = dict(visual_reason="version_mismatch", visual_policy="pad_moment_fallback")
            return None
        self.last_facts = facts
        return float(omega)

    def _preview_scan_evidence(self, *, now_s, dt_s, contact_present, force_n, stick_speed_m_s):
        from .pad_contact_evidence import PadContactEvidence
        self._ensure_receiver()
        if self._pad_evidence is None:
            self._pad_evidence = PadContactEvidence(self.config, self._receiver)
            self._image_kf = self._pad_evidence.kf
        result = self._pad_evidence.update(
            now_s=now_s, dt_s=dt_s, contact_present=bool(contact_present),
            force_n=float(force_n), stick_speed_m_s=float(stick_speed_m_s))
        self.evidence_valid = result.valid
        self.alpha_target = result.alpha_target
        self.omega_target = result.omega_request_rad_s
        # Safety pauses are immediate. Confirmed repair ramps toward the live
        # speed cap; recovery increases smoothly, with no saved displacement.
        if self.alpha_target == 0. or not np.isfinite(dt_s) or dt_s <= 0.:
            self.alpha_cmd = 0.
        else:
            ramp = self.ramp_s if self.alpha_target < self.alpha_cmd else self.ramp_up_s
            self.alpha_cmd = self._slew(self.alpha_cmd, self.alpha_target, 1. / ramp, dt_s)
        ramped_alpha = self.alpha_cmd
        # A faster current stick request must not defeat a confirmed repair's
        # physical speed cap while the preferred alpha ramps downward. Keep
        # the accepted alpha as the recovery origin, not the unclipped ramp.
        if result.valid and result.reason == 'unilateral_missing_echo':
            cap = self.config['pad']['repair_speed_cap_m_s']
            self.alpha_cmd = min(self.alpha_cmd, cap/max(abs(float(stick_speed_m_s)), 1e-12))
        self.last_alpha = self.alpha_cmd
        # The outer QP owns angular acceleration and total angular limiting.
        self.omega_cmd = self.last_visual_omega = self.omega_target
        self.last_facts = dict(result.facts, evidence_valid=result.valid,
            visual_reason=result.reason, visual_policy='pad_scan_nearfield_v1',
            alpha=self.last_alpha, visual_request_rad_s=self.omega_target,
            alpha_before_speed_cap=ramped_alpha,
            repair_speed_cap_limited=self.alpha_cmd < ramped_alpha,
            visual_omega_rad_s=self.last_visual_omega,
            stick_speed_m_s=stick_speed_m_s,
            path_speed_target_m_s=abs(stick_speed_m_s)*self.alpha_target,
            path_speed_requested_m_s=abs(stick_speed_m_s)*self.last_alpha,
            config_version=self.config.get('config_version', 'pad_scan_v1'))
        return None

    def _overlay_echo_visual(self, *, now_s, dt_s, contact_present, force_n, stick_speed_m_s):
        """Use 145-column near-field echo for ω. Do not park the stick at α=0.

        Residual edge wedges stay above c_min in 20-region Q25. Echo sees them.
        """
        from .pad_contact_evidence import PadContactEvidence
        self._ensure_receiver()
        if self._pad_evidence is None:
            self._pad_evidence = PadContactEvidence(self.config, self._receiver)
        result = self._pad_evidence.update(
            now_s=now_s, dt_s=dt_s, contact_present=bool(contact_present),
            force_n=float(force_n), stick_speed_m_s=float(stick_speed_m_s))
        facts = dict(self.last_facts)
        facts.update(
            echo_reason=result.reason, echo_valid=result.valid,
            echo_omega_rad_s=result.omega_request_rad_s, evidence_valid=result.valid)
        if result.reason == 'unilateral_missing_echo' and abs(float(result.omega_request_rad_s)) > 0.:
            self.omega_target = float(result.omega_request_rad_s)
            self.omega_cmd = float(np.clip(
                self._slew(self.omega_cmd, self.omega_target, self.omega_accel, dt_s),
                -self.omega_vmax, self.omega_vmax))
            self.last_visual_omega = self.omega_cmd
            if float(result.alpha_target) > 0.:
                self.alpha_target = min(float(self.alpha_target), float(result.alpha_target))
                if self.alpha_target < self.alpha_cmd:
                    alpha_rate = 0.75 / self.ramp_s
                else:
                    alpha_rate = 0.75 / max(self.ramp_up_s, 1.0e-6)
                self.alpha_cmd = float(np.clip(
                    self._slew(self.alpha_cmd, self.alpha_target, alpha_rate, dt_s),
                    self.alpha_min, 1.0))
                cap = self.config['pad']['repair_speed_cap_m_s']
                self.alpha_cmd = min(self.alpha_cmd, cap / max(abs(float(stick_speed_m_s)), 1e-12))
                self.last_alpha = self.alpha_cmd
            facts.update(
                visual_reason=result.reason, visual_policy='pad_scan_nearfield_v1',
                visual_request_rad_s=self.omega_target, visual_omega_rad_s=self.omega_cmd,
                alpha=self.last_alpha, alpha_target=self.alpha_target)
        self.last_facts = facts

    def _preview_progress(self, observation, *, contact_present, now_s, dt_s):
        """Slew region α and an additive ωy. Moment keeps the continuous plant."""

        quality, meta, kf_facts = self._progress_evidence(now_s, observation)
        reason = "no_contact"
        region_quality = None if quality is None else np.asarray(quality, dtype=float)
        max_deficit = None
        quality_reference = None
        task = None
        bad_indices = ()
        image_age = kf_facts.get("kf_age_s")
        if image_age is None and observation is not None:
            try:
                image_age = float(now_s) - float(observation.effective_time_s)
            except (TypeError, ValueError):
                image_age = None
        if not contact_present:
            self.alpha_target = 1.0
            self.omega_target = 0.0
            self._clear_visual_hold()
        elif quality is not None and meta is not None:
            try:
                task = region_visual_request(
                    quality,
                    meta.region_valid,
                    meta.region_edges,
                    self.geometry,
                    c_min=self.c_min,
                    repair_speed_m_s=self.repair_speed_m_s,
                    evidence_valid=True,
                    progress_loss_gain=self.progress_loss_gain,
                    alpha_min=self.alpha_min,
                    alpha_min_at_quality=self.alpha_min_at_quality,
                    max_request_rad_s=self.max_visual_omega_rad_s,
                    rocking_reference=None if self.deficit_mode == "relative" else self.rocking_reference,
                    deficit_mode=self.deficit_mode,
                    balance_deadband=self.balance_deadband if self.deficit_mode == "relative" else 0.0,
                    relative_alpha_span=self.relative_alpha_span,
                )
            except ValueError:
                reason = "version_mismatch"
                self.alpha_target = 1.0
                self.omega_target = 0.0
                self._clear_visual_hold()
            else:
                self._image_s = float(now_s)
                if task.region_quality is not None:
                    region_quality = np.asarray(task.region_quality, dtype=float)
                if task.region_deficits is not None:
                    max_deficit = float(np.max(task.region_deficits))
                quality_reference = task.quality_reference
                bad_indices = tuple(task.bad_region_indices)
                if task.valid:
                    self.alpha_target = float(task.alpha_preferred)
                    self.omega_target = float(task.request_rad_s)
                    sided = task.reason == "regional_deficit_moment"
                    deep = max_deficit is not None and max_deficit >= self.omega_step_deficit
                    if (self.omega_step_rad_s > 0.0 and sided and deep
                            and 0.0 < abs(self.omega_target) < self.omega_step_rad_s):
                        self.omega_target = float(np.copysign(self.omega_step_rad_s, self.omega_target))
                    if self.brake_on_sided and sided and deep:
                        self.alpha_target = self.alpha_min
                    reason = self._apply_visual_hold(now_s, task.reason)
                else:
                    self.alpha_target = 1.0
                    self.omega_target = 0.0
                    reason = task.reason
                    self._clear_visual_hold()
        elif self._image_s is None or (float(now_s) - float(self._image_s)) > self.max_image_age_s:
            self.alpha_target = 1.0
            self.omega_target = 0.0
            reason = "image_absent"
            self._clear_visual_hold()
        else:
            reason = self._apply_visual_hold(now_s, "image_hold")
        if self.alpha_target < self.alpha_cmd:
            alpha_rate = 0.75 / self.ramp_s
        else:
            alpha_rate = 0.75 / max(self.ramp_up_s, 1.0e-6)
        self.alpha_cmd = float(np.clip(
            self._slew(self.alpha_cmd, self.alpha_target, alpha_rate, dt_s),
            self.alpha_min, 1.0,
        ))
        self.omega_cmd = float(np.clip(
            self._slew(self.omega_cmd, self.omega_target, self.omega_accel, dt_s),
            -self.omega_vmax, self.omega_vmax,
        ))
        self.last_alpha = self.alpha_cmd
        self.last_visual_omega = self.omega_cmd
        self.last_facts = dict(
            visual_reason=reason,
            visual_policy="pad_ultrapoc_delay_kf" if self._image_kf is not None else "pad_ultrapoc_gated_step",
            rocking_reference=self.rocking_reference,
            deficit_mode=self.deficit_mode,
            quality_reference=quality_reference,
            omega_step_rad_s=self.omega_step_rad_s,
            alpha=self.alpha_cmd, alpha_target=self.alpha_target,
            visual_request_rad_s=self.omega_target, visual_omega_rad_s=self.omega_cmd,
            image_age_s=image_age, max_deficit=max_deficit,
            bad_region_indices=list(bad_indices),
            region_quality=None if region_quality is None else region_quality.tolist(),
            kf_valid=kf_facts.get("kf_valid"),
            kf_reason=kf_facts.get("kf_reason"),
            kf_age_s=kf_facts.get("kf_age_s"),
            kf_quality=kf_facts.get("kf_quality"),
        )
        return None

    def commit(self, final_tool, rotation):
        if self._seeded:
            self.task.commit(final_tool, rotation)

    def close(self, *, wait=False):
        receiver = self._receiver
        self._receiver = None
        if self._owns_receiver and receiver is not None:
            close = getattr(receiver, "close", None)
            if callable(close):
                close(wait=wait)
