"""Pad-mode visual ωy guide. No outer QP, path, α, or delay KF."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from peirastic.contact_qp.confidence_fusion import ConfidenceAngularTask, FEATURE_VERSION
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.geometry import window_rows
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config
from peirastic.realman8dof.force.torque_tilt import rotation_from_pose

DEFAULT_PAD_VISUAL_CONFIG = (
    Path(__file__).resolve().parents[2] / "config/contact_qp/active_probe50_delay_kf_cop.yaml"
)


def _explicit_visual_request(payload: dict) -> bool:
    visual = payload.get("pad_visual")
    return bool(
        visual is True
        or (isinstance(visual, dict) and visual.get("enabled") is not False)
    )


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
            return load_study_config(DEFAULT_PAD_VISUAL_CONFIG)
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
        return PadVisualGuide(config, tilt_cfg, receiver=receiver)
    except ValueError:
        if _explicit_visual_request(pay):
            raise
        return None


class PadVisualGuide:
    """Inject ConfidenceAngularTask ωy when a fresh image is present."""

    def __init__(self, config, tilt_cfg, *, receiver=None):
        self.config = dict(config)
        self.geometry, _ = calibrated_geometry(self.config)
        qp = dict(self.config.get("qp") or {})
        feature = dict(self.config.get("feature") or {})
        raw_feature = feature.get("config")
        windows = FeatureConfig(**raw_feature).lateral_windows if raw_feature else FeatureConfig().lateral_windows
        rows = window_rows(self.geometry, windows)
        repair = qp.get("differential_repair") or {}
        deadband = float(repair.get("balance_deadband", 0.03)) if isinstance(repair, dict) else 0.03
        c_min = float(feature.get("c_min", qp.get("c_min", 0.5)))
        self.max_image_age_s = float(qp.get("max_image_age_s", 0.3))
        self.task = ConfidenceAngularTask(
            mass=float(tilt_cfg.mass),
            damping=float(tilt_cfg.damping),
            repair_speed_m_s=float(qp.get("repair_speed_m_s", 0.002)),
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
        self.last_facts = {}

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

    def preview(self, *, pose, dt_s, contact_present, tilt, now_s=None) -> float | None:
        if tilt is None:
            return None
        import time
        now = float(now_s if now_s is not None else time.monotonic())
        observation = self._fresh_observation(now)
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
