"""ARCHIVED: offline collocation lives in ``bak/human_dynamics_tracking_archive_20260520/``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollocationOptions:
    """Stub kept for CLI/demo wiring; offline collocation disabled."""

    horizon: int = 9
    refresh_stride: int = 3
    lbfgs_max_iter: int = 12
    lbfgs_history_size: int = 10
    continuation_stages: tuple[float, ...] = (0.05, 0.01, 0.001)
    penetration_caps_m: tuple[float, ...] = (0.05, 0.01, 0.0)
    contact_band_m: float = 0.05
    w_vertex_track: float = 80.0
    w_kin: float = 0.0
    w_dyn: float = 5.0e1
    w_vposer: float = 0.0
    w_tau_smooth: float = 1.0e2
    w_tau_effort: float = 0.0
    w_velocity_smooth: float = 0.0
    w_pd_consistency: float = 0.0
    pd_kp: float = 0.0
    pd_kd: float = 0.0
    w_penetration: float = 1.5e6
    w_complementarity: float = 3.0e2
    w_landing: float = 0.0
    w_contact_force: float = 0.0
    bed_contact_stiffness_n_per_m: float = 1.5e6
    contact_plane_eps_m: float = 0.001
    delta_root_z_max_m: float = 0.04
    delta_root_z_up_max_m: float = 0.02


__all__ = ("CollocationOptions",)
