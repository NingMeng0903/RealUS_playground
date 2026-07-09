"""Scale admittance config for runtime ``desired_z`` CLI overrides."""

from __future__ import annotations

from rm75_control.control.admittance_common.controller import AdmittanceConfig


def scale_admittance_for_desired_z(raw: dict, desired_z_n: float) -> AdmittanceConfig:
    """Adapt setpoint-relative gates to the runtime ``desired_z``.

    The force deadband is deliberately NOT scaled: it rejects sensor noise as
    a fixed physical quantity. At high setpoints (e.g. 12 N) we also scale
    Dimeas ``f_max_n``, virtual mass, and ``bd_max`` so the same
    yaml stays stable from 1–12 N.
    """
    cfg = AdmittanceConfig.from_dict(raw)
    base_z = float(raw.get("force", {}).get("desired_z_n", 1.0))
    if base_z <= 0.0 or desired_z_n <= 0.0:
        return cfg

    ratio = desired_z_n / base_z
    if ratio == 1.0:
        return cfg

    if cfg.adaptive_ke.enabled:
        ak = cfg.adaptive_ke
        ak.contact_force_n = min(float(ak.contact_force_n) * ratio, 0.35 * desired_z_n)
        ak.bd_max = min(400.0, float(ak.bd_max) * max(1.0, ratio ** 0.35))

    cfg.var_damping_f_max_n = max(
        float(cfg.var_damping_f_max_n) * ratio,
        0.55 * desired_z_n,
    )

    if ratio > 1.0:
        cfg.admittance_mass_z = float(cfg.admittance_mass_z) * (ratio ** 0.35)
    return cfg
