"""Serialize JointIkConfig + URDF paths for wbc_rt."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import DEFAULT_URDF


def _f(x) -> str:
    return f"{float(x):.17g}"


def _arr(name: str, value) -> str:
    a = np.asarray(value, dtype=float).reshape(-1)
    return name + " " + " ".join(_f(v) for v in a)


def _locked_style_u(style) -> int:
    name = str(getattr(style, "name", style)).split(".")[-1].lower()
    if name == "hold":
        return 0
    if "rail" in name:
        return 1
    return 2


def dump_wbc_config(
    cfg,
    path: str | Path,
    *,
    urdf_path: str | Path | None = None,
    collision_urdf: str | Path | None = None,
    pair_config: str | Path | None = None,
    kin=None,
) -> Path:
    """Write a line-oriented config that C++ ``Config::load`` understands."""
    from rm75_control.control.joint_admittance_8dof.collision_model import (
        DEFAULT_COLLISION_URDF,
        DEFAULT_PAIR_CONFIG,
    )

    out = Path(path)
    qp = cfg.qp
    coll = cfg.collision
    ns = cfg.nullspace
    ra = cfg.rail_allocator
    re = cfg.rail_extension
    rail = cfg.rail
    aa = cfg.arm_angle
    pr = cfg.psi_retarget
    sat = cfg.saturation
    ird = cfg.ird
    urdf = Path(urdf_path) if urdf_path is not None else DEFAULT_URDF
    c_urdf = Path(collision_urdf) if collision_urdf is not None else DEFAULT_COLLISION_URDF
    pairs = Path(pair_config) if pair_config is not None else DEFAULT_PAIR_CONFIG
    lines = [
        f"urdf {urdf}",
        f"collision_urdf {c_urdf}",
        f"pair_config {pairs}",
        f"dt {_f(cfg.dt)}",
        f"feedback_timeout_s {_f(cfg.feedback_timeout_s)}",
        f"control_frame {cfg.control_frame}",
        f"euler_order {cfg.euler_order}",
        f"v_scale {_f(cfg.v_scale)}",
        f"a_max_arm_rad_s2 {_f(cfg.a_max_arm_rad_s2)}",
        f"a_max_rail_m_s2 {_f(cfg.a_max_rail_m_s2)}",
        f"position_margin_rad {_f(cfg.position_margin_rad)}",
        f"position_margin_rail_m {_f(cfg.position_margin_rail_m)}",
        f"resync_err_rad {_f(cfg.resync_err_rad)}",
        f"resync_err_rail_m {_f(cfg.resync_err_rail_m)}",
        f"nullspace_d_null {_f(cfg.nullspace_d_null)}",
        f"nullspace_d_null_adaptive {_f(cfg.nullspace_d_null_adaptive)}",
        f"nullspace_max_qdot_frac {_f(cfg.nullspace_max_qdot_frac)}",
        f"sec_target_hz {_f(getattr(cfg, 'sec_target_hz', 15.0))}",
        f"sec_input_lpf_hz {_f(getattr(cfg, 'sec_input_lpf_hz', 0.0))}",
        f"ns_grad_lpf_hz {_f(getattr(cfg, 'ns_grad_lpf_hz', 0.0))}",
        f"ns_hold_fade_v {_f(getattr(cfg, 'ns_hold_fade_v', 0.0))}",
        f"ns_hold_fade_v0 {_f(getattr(cfg, 'ns_hold_fade_v0', 0.0))}",
        f"sec_filter_zeta {_f(getattr(cfg, 'sec_filter_zeta', 1.0))}",
        _arr("qp.task_weight", qp.task_weight),
        _arr("qp.reg", qp.reg),
        f"qp.eps_abs {_f(qp.eps_abs)}",
        f"qp.max_iter {int(qp.max_iter)}",
        f"qp.max_iter_cap {int(qp.max_iter_cap)}",
        f"qp.max_solve_ms {_f(qp.max_solve_ms)}",
        f"qp.fail_qdot_decay {_f(qp.fail_qdot_decay)}",
        f"qp.twist_sigma_floor {_f(qp.twist_sigma_floor)}",
        f"qp.task_weight_min_frac {_f(qp.task_weight_min_frac)}",
        f"qp.task_weight_lpf_tau_s {_f(qp.task_weight_lpf_tau_s)}",
        f"qp.aniso_task_damping {int(bool(qp.aniso_task_damping))}",
        f"qp.use_mass_weighted_reg {int(bool(qp.use_mass_weighted_reg))}",
        f"qp.mass_reg_floor {_f(qp.mass_reg_floor)}",
        f"qp.mass_weight_exempt_rail {int(bool(qp.mass_weight_exempt_rail))}",
        f"qp.mass_reg_lpf_tau_s {_f(qp.mass_reg_lpf_tau_s)}",
        f"qp.limit_damper_band_rad {_f(qp.limit_damper_band_rad)}",
        f"qp.limit_damper_band_rail_m {_f(qp.limit_damper_band_rail_m)}",
        f"qp.limit_damper_rail_reaction_s {_f(qp.limit_damper_rail_reaction_s)}",
        f"qp.near_arm_margin_rad {_f(qp.near_arm_margin_rad)}",
        f"qp.j_max_arm_rad_s3 {_f(qp.j_max_arm_rad_s3)}",
        f"qp.j_max_rail_m_s3 {_f(qp.j_max_rail_m_s3)}",
        _arr("qp.smoothness_weight", qp.smoothness_weight),
        f"qp.sr_lam0 {_f(qp.sr_damping.lam0)}",
        f"qp.sr_sigma_ref {_f(qp.sr_damping.sigma_ref)}",
        f"qp.sr_sigma_floor {_f(qp.sr_damping.sigma_floor)}",
        f"qp.sigma_setbased.enabled {int(bool(qp.sigma_setbased.enabled))}",
        f"qp.sigma_setbased.activate {_f(qp.sigma_setbased.activate)}",
        f"qp.sigma_setbased.safe {_f(qp.sigma_setbased.safe)}",
        f"qp.sigma_setbased.exit {_f(qp.sigma_setbased.exit)}",
        f"qp.sigma_setbased.gamma {_f(qp.sigma_setbased.gamma)}",
        f"qp.sigma_setbased.slack_weight {_f(qp.sigma_setbased.slack_weight)}",
        f"qp.sigma_setbased.grad_eps {_f(qp.sigma_setbased.grad_eps)}",
        f"qp.sigma_setbased.grad_period_ticks {int(qp.sigma_setbased.grad_period_ticks)}",
        f"qp.branch_barrier.enabled {int(bool(qp.branch_barrier.enabled))}",
        f"qp.branch_barrier.activate_rad {_f(qp.branch_barrier.activate_rad)}",
        f"qp.branch_barrier.box_activate_rad {_f(qp.branch_barrier.box_activate_rad)}",
        f"qp.branch_barrier.eps_rad {_f(qp.branch_barrier.eps_rad)}",
        f"qp.branch_barrier.j4_limit_eps_rad {_f(qp.branch_barrier.j4_limit_eps_rad)}",
        f"qp.branch_barrier.j4_limit_activate_rad {_f(qp.branch_barrier.j4_limit_activate_rad)}",
        f"qp.branch_barrier.j1_overfold_abs_rad {_f(qp.branch_barrier.j1_overfold_abs_rad)}",
        f"qp.branch_barrier.j1_overfold_activate_rad {_f(qp.branch_barrier.j1_overfold_activate_rad)}",
        f"qp.branch_barrier.j1_overfold_eps_rad {_f(qp.branch_barrier.j1_overfold_eps_rad)}",
        f"qp.branch_barrier.gamma {_f(qp.branch_barrier.gamma)}",
        f"qp.branch_barrier.slack_weight {_f(qp.branch_barrier.slack_weight)}",
        f"qp.branch_barrier.target_eps_rad {_f(qp.branch_barrier.target_eps_rad)}",
        f"qp.branch_barrier.dwell_free_s {_f(qp.branch_barrier.dwell_free_s)}",
        f"qp.branch_barrier.dwell_ramp_s {_f(qp.branch_barrier.dwell_ramp_s)}",
        f"qp.branch_barrier.dwell_scale_max {_f(qp.branch_barrier.dwell_scale_max)}",
        f"qp.joint_comfort.enabled {int(bool(qp.joint_comfort.enabled))}",
        f"qp.joint_comfort.m_comfort_rad {_f(qp.joint_comfort.m_comfort_rad)}",
        f"qp.joint_comfort.activate_rad {_f(qp.joint_comfort.activate_rad)}",
        f"qp.joint_comfort.gamma {_f(qp.joint_comfort.gamma)}",
        f"qp.joint_comfort.slack_weight {_f(qp.joint_comfort.slack_weight)}",
        f"qp.j4_design_comfort.enabled {int(bool(qp.j4_design_comfort.enabled))}",
        f"qp.j4_design_comfort.lower_rad {_f(qp.j4_design_comfort.lower_rad)}",
        f"qp.j4_design_comfort.upper_rad {_f(qp.j4_design_comfort.upper_rad)}",
        f"qp.j4_design_comfort.gamma {_f(qp.j4_design_comfort.gamma)}",
        f"qp.j4_design_comfort.slack_weight {_f(qp.j4_design_comfort.slack_weight)}",
        f"collision.enabled {int(bool(coll.enabled))}",
        f"collision.d_safe {_f(coll.d_safe)}",
        f"collision.d_activate {_f(coll.d_activate)}",
        f"collision.gamma {_f(coll.gamma)}",
        f"collision.max_pairs {int(coll.max_pairs)}",
        f"nullspace.k_center {_f(ns.k_center)}",
        f"nullspace.k_limit {_f(ns.k_limit)}",
        f"nullspace.activation {_f(ns.activation)}",
        _arr(
            "nullspace.q_nominal_rad",
            ns.q_nominal_rad if ns.q_nominal_rad is not None else np.zeros(8),
        ),
        f"arm_angle.enabled {int(bool(aa.enabled))}",
        f"arm_angle.k_psi {_f(aa.k_psi)}",
        f"arm_angle.fd_eps_rad {_f(aa.fd_eps_rad)}",
        f"arm_angle.safe_denom_eps {_f(aa.safe_denom_eps)}",
        f"arm_angle.obs_decay_gain {_f(aa.obs_decay_gain)}",
        f"arm_angle.obs_smooth_floor {_f(aa.obs_smooth_floor)}",
        f"arm_angle.max_qdot_frac {_f(aa.max_qdot_frac)}",
        f"psi_retarget.enabled {int(bool(pr.enabled))}",
        f"psi_retarget.psi_attr_rad {_f(pr.psi_attr_rad)}",
        f"psi_retarget.d_attr_m {_f(pr.d_attr_m)}",
        f"psi_retarget.psi_envelope_lo_rad {_f(pr.psi_envelope_lo_rad)}",
        f"psi_retarget.psi_envelope_hi_rad {_f(pr.psi_envelope_hi_rad)}",
        f"psi_retarget.d_center_rate_m_s {_f(getattr(pr, 'd_center_rate_m_s', 0.04))}",
        f"psi_retarget.psi_rate_rad_s {_f(getattr(pr, 'psi_rate_rad_s', 0.4))}",
        f"psi_retarget.rail_margin_m {_f(getattr(pr, 'rail_margin_m', 0.01))}",
        f"psi_retarget.elbow_lo_rad {_f(getattr(pr, 'elbow_lo_rad', 1.2217304763960306))}",
        f"psi_retarget.elbow_center_rad {_f(getattr(pr, 'elbow_center_rad', 1.6580627893946132))}",
        f"psi_retarget.elbow_hi_rad {_f(getattr(pr, 'elbow_hi_rad', 2.007128639793479))}",
        f"psi_retarget.elbow_hi_illegal_rad {_f(getattr(pr, 'elbow_hi_illegal_rad', 2.2689280275926285))}",
        f"psi_retarget.psi_cmd_lead_rad {_f(getattr(pr, 'psi_cmd_lead_rad', 0.3141592653589793))}",
        f"psi_retarget.psi_return_dwell_s {_f(getattr(pr, 'psi_return_dwell_s', 1.0))}",
        f"psi_retarget.psi_replan_period_s {_f(getattr(pr, 'psi_replan_period_s', 0.1))}",
        f"psi_retarget.psi_search_half_span_rad {_f(getattr(pr, 'psi_search_half_span_rad', 0.7853981633974483))}",
        f"psi_retarget.psi_search_n {int(getattr(pr, 'psi_search_n', 9))}",
        f"psi_retarget.psi_wrist_ok_rad {_f(getattr(pr, 'psi_wrist_ok_rad', 0.6981317007977318))}",
        f"psi_retarget.wrist_min_rad {_f(getattr(pr, 'wrist_min_rad', 0.5235987755982988))}",
        f"ird.enabled {int(bool(getattr(ird, 'enabled', False)))}",
        f"manipulability.k_mu {_f(cfg.manipulability.k_mu)}",
        f"manipulability.sigma_fade_ref {_f(getattr(cfg.manipulability, 'sigma_fade_ref', 0.08))}",
        f"rail.mode {0 if str(getattr(rail.mode, 'name', rail.mode)).split('.')[-1].lower() == 'coupled' else 1}",
        f"rail.locked_style {_locked_style_u(rail.locked_style)}",
        f"rail.lock_vel_eps_m_s {_f(rail.lock_vel_eps_m_s)}",
        f"rail.v_max_m_s {_f(rail.v_max_m_s if rail.v_max_m_s is not None else 0.15)}",
        f"rail.soft_min_m {_f(rail.soft_min_m)}",
        f"rail.soft_max_m {_f(rail.soft_max_m)}",
        f"rail.hard_min_m {_f(rail.hard_min_m)}",
        f"rail.hard_max_m {_f(rail.hard_max_m)}",
        f"rail.lock_hard_pin {int(bool(getattr(rail, 'lock_hard_pin', True)))}",
        f"rail.lock_reg_scale {_f(getattr(rail, 'lock_reg_scale', 1.0))}",
        f"rail_allocator.v0_m_s {_f(ra.v0_m_s)}",
        f"rail_allocator.w0_rad_s {_f(ra.w0_rad_s)}",
        f"rail_allocator.k_margin {_f(ra.k_margin)}",
        f"rail_allocator.kp_mid {_f(ra.kp_mid)}",
        f"rail_allocator.ki_mid {_f(ra.ki_mid)}",
        f"rail_allocator.u_mid_max_m_s {_f(ra.u_mid_max_m_s)}",
        f"rail_allocator.k_err_rail {_f(ra.k_err_rail)}",
        f"rail_allocator.e_ref_m {_f(ra.e_ref_m)}",
        f"rail_allocator.f_c_hz {_f(ra.f_c_hz)}",
        f"rail_allocator.kaw_mid {_f(ra.kaw_mid)}",
        f"rail_allocator.rho_mirror_a {_f(ra.rho_mirror_a)}",
        f"rail_allocator.rho_mirror_j {_f(ra.rho_mirror_j)}",
        f"rail_allocator.reaction_s {_f(ra.reaction_s)}",
        f"rail_allocator.observer_pos_gain {_f(ra.observer_pos_gain)}",
        f"rail_allocator.observer_vel_gain {_f(ra.observer_vel_gain)}",
        f"rail_allocator.observer_vel_lpf_hz {_f(ra.observer_vel_lpf_hz)}",
        f"rail_extension.enabled {int(bool(re.enabled))}",
        f"rail_extension.k_ext {_f(re.k_ext)}",
        f"rail_extension.k_ff {_f(re.k_ff)}",
        f"rail_extension.v_ff_thr_m_s {_f(re.v_ff_thr_m_s)}",
        f"rail_extension.v_ff_span_m_s {_f(re.v_ff_span_m_s)}",
        f"rail_extension.e0_m {_f(re.e0_m)}",
        f"rail_extension.e1_m {_f(re.e1_m)}",
        f"rail_extension.w_max {_f(re.w_max)}",
        f"rail_extension.v_max_m_s {_f(re.v_max_m_s)}",
        f"rail_extension.limit_margin_m {_f(re.limit_margin_m)}",
        f"rail_extension.pin_margin_m {_f(re.pin_margin_m)}",
        f"rail_extension.escape_leave_m {_f(re.escape_leave_m)}",
        f"rail_extension.soft_min_m {_f(re.soft_min_m)}",
        f"rail_extension.soft_max_m {_f(re.soft_max_m)}",
        f"rail_extension.healthy_sigma_mute {_f(re.healthy_sigma_mute)}",
        f"rail_extension.d_band_m {_f(re.d_band_m)}",
        f"rail_extension.k_sigma_boost {_f(re.k_sigma_boost)}",
        f"rail_extension.k_esc {_f(re.k_esc)}",
        f"rail_extension.w_sigma_floor {_f(re.w_sigma_floor)}",
        f"rail_extension.k_pose {_f(re.k_pose)}",
        f"rail_extension.pose_e0_m {_f(re.pose_e0_m)}",
        f"rail_extension.pose_e1_m {_f(re.pose_e1_m)}",
        f"rail_extension.pose_w_max {_f(re.pose_w_max)}",
        f"rail_extension.k_escape_boost {_f(re.k_escape_boost)}",
        f"rail_extension.k_margin_boost {_f(re.k_margin_boost)}",
        f"rail_extension.w_ext_cap {_f(re.w_ext_cap)}",
        f"rail_extension.d_star_err0_m {_f(re.d_star_err0_m)}",
        f"rail_extension.d_star_err1_m {_f(re.d_star_err1_m)}",
        f"rail_extension.d_star_w_mult {_f(re.d_star_w_mult)}",
        f"rail_extension.press_v_force_min_m_s {_f(re.press_v_force_min_m_s)}",
        f"rail_extension.press_dz_max_m {_f(re.press_dz_max_m)}",
        f"rail_extension.press_y_err_m {_f(re.press_y_err_m)}",
        f"rail_extension.press_stall_s {_f(re.press_stall_s)}",
        f"rail_extension.d_star_nudge_m {_f(re.d_star_nudge_m)}",
        f"rail_extension.open_travel_min_m {_f(re.open_travel_min_m)}",
        f"rail_extension.escape_sign_policy {re.escape_sign_policy}",
        f"saturation.slack_enter {_f(sat.slack_enter)}",
        f"saturation.slack_exit {_f(sat.slack_exit)}",
        f"saturation.secondary_scale {_f(sat.secondary_scale)}",
        f"saturation.secondary_scale_tau_s {_f(getattr(sat, 'secondary_scale_tau_s', 0.10))}",
    ]
    if kin is not None:
        from rm75_control.kinematics.srs_ik import flange_tcp_from_kin

        R_ft, t_ft = flange_tcp_from_kin(kin)
        lines.append(_arr("srs.R_flange_tcp", np.asarray(R_ft, dtype=float).reshape(-1)))
        lines.append(_arr("srs.t_flange_tcp", np.asarray(t_ft, dtype=float).reshape(-1)))
        M = kin.model.frames[kin.tcp_id].placement
        lines.append(_arr("tcp_placement_R", np.asarray(M.rotation, dtype=float).reshape(3, 3).reshape(-1)))
        lines.append(_arr("tcp_placement_t", np.asarray(M.translation, dtype=float).reshape(-1)))
    out.write_text("\n".join(lines) + "\n")
    return out
