"""Load peirastic/configs/force.yaml — the force-axis law for every force mode."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from peirastic.configs import DEFAULT_FORCE_YAML

_SKIP_PAYLOAD = {
    "reference",
    "use_tff_split",
    "duration_s",
    "label",
    "v_cmd",
    "desired_z",
}


def load_force_raw(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_FORCE_YAML
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"force yaml is empty: {p}")
    raw["_path"] = str(p)
    return raw


def desired_z_n(raw: dict | None = None, payload: dict | None = None) -> float:
    pay = dict(payload or {})
    if pay.get("desired_z") is not None:
        return float(pay["desired_z"])
    src = raw if raw is not None else load_force_raw()
    return float(src.get("force", {}).get("desired_z_n", 1.0))


def _deep_merge(dst: dict, src: dict) -> dict:
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def apply_force_payload(raw: dict, payload: dict | None) -> dict:
    """Overlay a mode-request payload onto a copy of the force yaml."""

    out = copy.deepcopy(raw)
    pay = dict(payload or {})
    if pay.get("desired_z") is not None:
        out.setdefault("force", {})["desired_z_n"] = float(pay["desired_z"])
    if isinstance(pay.get("force"), dict):
        out.setdefault("force", {})
        _deep_merge(out["force"], pay["force"])
    hm = dict(out.get("hybrid_motion") or {})
    if isinstance(pay.get("hybrid_motion"), dict):
        _deep_merge(hm, pay["hybrid_motion"])
    if pay.get("v_seek_free_m_s") is not None:
        hm.setdefault("force_barrier", {})["v_seek_free_m_s"] = float(pay["v_seek_free_m_s"])
    if pay.get("max_az_tool_m_s2") is not None:
        acc = list(hm.get("max_acceleration") or [1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
        if len(acc) >= 3:
            acc[2] = float(pay["max_az_tool_m_s2"])
        hm["max_acceleration"] = acc
    if pay.get("max_vz_tool_m_s") is not None:
        vz = float(pay["max_vz_tool_m_s"])
        hm["max_vz_tool_m_s"] = vz
        vel = list(hm.get("max_velocity") or [0.22, 0.22, 0.1, 0.6, 0.6, 0.6])
        if len(vel) >= 3:
            vel[2] = vz
        hm["max_velocity"] = vel
    for key, val in pay.items():
        if key in _SKIP_PAYLOAD or key in ("force", "hybrid_motion") or val is None:
            continue
        if key in hm or key in (
            "desired_force_ramp_s",
            "force_axis_slew_press_m_s2",
            "force_axis_slew_retract_m_s2",
            "force_axis_slew_reverse_m_s2",
        ):
            hm[key] = val
    out["hybrid_motion"] = hm
    return out


def build_force_controller(
    dt: float,
    *,
    payload: dict | None = None,
    path: str | Path | None = None,
) -> tuple[AdmittanceController, dict, float]:
    """Reload force.yaml, apply payload overlays, return controller + raw + Fz*."""

    raw = apply_force_payload(load_force_raw(path), payload)
    fz = desired_z_n(raw, payload)
    raw.setdefault("force", {})["desired_z_n"] = float(fz)
    cfg = AdmittanceConfig.from_dict(raw)
    return AdmittanceController(float(dt), cfg), raw, float(fz)
