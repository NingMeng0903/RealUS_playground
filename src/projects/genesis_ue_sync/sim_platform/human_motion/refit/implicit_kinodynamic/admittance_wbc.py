"""ARCHIVED: velocity WBC lives in ``bak/human_dynamics_tracking_archive_20260520/``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmittanceWbcOptions:
    """Stub kept for CLI/demo wiring; tracking QP disabled."""

    dt: float = 1.0 / 30.0
    floating_base_dofs: int = 6
    bed_margin_m: float = 0.005
    contact_band_m: float = 0.05
    max_contacts: int = 12
    marker_kp: float = 420.0
    max_marker_speed_m_s: float = 3.0
    max_qdot_rad_s: float = 25.0
    phantom_sink_velocity_m_s: float = -0.3
    weight_track: float = 200.0
    weight_track_z_ratio: float = 0.05
    weight_grav: float = 80.0
    weight_mass: float = 2.0e-3
    weight_slack: float = 2.0e5
    cbf_alpha: float = 8.0
    vposer_eta: float = 0.02
    vposer_max_step_rad: float = 0.05
    qp_eps_abs: float = 1.0e-4
    qp_eps_rel: float = 1.0e-4
    qp_max_iter: int = 4000


__all__ = ("AdmittanceWbcOptions",)
