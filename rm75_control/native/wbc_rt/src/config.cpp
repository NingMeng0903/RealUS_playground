#include "wbc_rt/config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace wbc_rt {
namespace {

void fill_vec(const std::vector<double>& v, double* dst, int n) {
  for (int i = 0; i < n && i < static_cast<int>(v.size()); ++i) dst[i] = v[i];
}

}  // namespace

Config Config::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path);
  Config c;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string key;
    ss >> key;
    if (key.empty()) continue;
    std::vector<double> nums;
    std::string tok;
    std::string rest;
    while (ss >> tok) {
      char* end = nullptr;
      const double x = std::strtod(tok.c_str(), &end);
      if (end != tok.c_str() && *end == '\0') {
        nums.push_back(x);
      } else {
        if (!rest.empty()) rest += " ";
        rest += tok;
      }
    }
    auto n0 = [&]() { return nums.empty() ? 0.0 : nums[0]; };
    auto i0 = [&]() { return static_cast<int>(n0() + (n0() >= 0 ? 0.5 : -0.5)); };
    if (key == "urdf") c.urdf = rest;
    else if (key == "collision_urdf") c.collision_urdf = rest;
    else if (key == "pair_config") c.pair_config = rest;
    else if (key == "control_frame") c.control_frame = rest;
    else if (key == "euler_order") c.euler_order = rest;
    else if (key == "rail_extension.escape_sign_policy") c.escape_sign_policy = rest;
    else if (key == "dt") c.dt = n0();
    else if (key == "feedback_timeout_s") c.feedback_timeout_s = n0();
    else if (key == "v_scale") c.v_scale = n0();
    else if (key == "a_max_arm_rad_s2") c.a_max_arm = n0();
    else if (key == "a_max_rail_m_s2") c.a_max_rail = n0();
    else if (key == "position_margin_rad") c.position_margin_rad = n0();
    else if (key == "position_margin_rail_m") c.position_margin_rail_m = n0();
    else if (key == "resync_err_rad") c.resync_err_rad = n0();
    else if (key == "resync_err_rail_m") c.resync_err_rail_m = n0();
    else if (key == "nullspace_d_null") c.d_null = n0();
    else if (key == "nullspace_d_null_adaptive") c.d_null_adaptive = n0();
    else if (key == "nullspace_max_qdot_frac") c.max_qdot_frac = n0();
    else if (key == "sec_target_hz") c.sec_target_hz = n0();
    else if (key == "sec_input_lpf_hz") c.sec_input_lpf_hz = n0();
    else if (key == "ns_grad_lpf_hz") c.ns_grad_lpf_hz = n0();
    else if (key == "ns_hold_fade_v") c.ns_hold_fade_v = n0();
    else if (key == "ns_hold_fade_v0") c.ns_hold_fade_v0 = n0();
    else if (key == "sec_filter_zeta") c.sec_filter_zeta = n0();
    else if (key == "qp.task_weight") fill_vec(nums, c.task_weight.data(), 6);
    else if (key == "qp.reg") fill_vec(nums, c.reg.data(), 8);
    else if (key == "qp.smoothness_weight") fill_vec(nums, c.smoothness.data(), 8);
    else if (key == "qp.eps_abs") c.eps_abs = n0();
    else if (key == "qp.max_iter") c.max_iter = i0();
    else if (key == "qp.max_iter_cap") c.max_iter_cap = i0();
    else if (key == "qp.max_solve_ms") c.max_solve_ms = n0();
    else if (key == "qp.fail_qdot_decay") c.fail_qdot_decay = n0();
    else if (key == "qp.twist_sigma_floor") c.twist_sigma_floor = n0();
    else if (key == "qp.task_weight_min_frac") c.task_weight_min_frac = n0();
    else if (key == "qp.task_weight_lpf_tau_s") c.task_weight_lpf_tau_s = n0();
    else if (key == "qp.aniso_task_damping") c.aniso_task_damping = i0() != 0;
    else if (key == "qp.use_mass_weighted_reg") c.use_mass_weighted_reg = i0() != 0;
    else if (key == "qp.mass_reg_floor") c.mass_reg_floor = n0();
    else if (key == "qp.mass_weight_exempt_rail") c.mass_weight_exempt_rail = i0() != 0;
    else if (key == "qp.mass_reg_lpf_tau_s") c.mass_reg_lpf_tau_s = n0();
    else if (key == "qp.limit_damper_band_rad") c.damper_band_rad = n0();
    else if (key == "qp.limit_damper_band_rail_m") c.damper_band_rail = n0();
    else if (key == "qp.limit_damper_rail_reaction_s") c.rail_reaction_s = n0();
    else if (key == "qp.near_arm_margin_rad") c.near_arm_margin_rad = n0();
    else if (key == "qp.j_max_arm_rad_s3") c.j_max_arm = n0();
    else if (key == "qp.j_max_rail_m_s3") c.j_max_rail = n0();
    else if (key == "qp.sr_lam0") c.sr_lam0 = n0();
    else if (key == "qp.sr_sigma_ref") c.sr_sigma_ref = n0();
    else if (key == "qp.sr_sigma_floor") c.sr_sigma_floor = n0();
    else if (key == "qp.sigma_setbased.enabled") c.sigma_enabled = i0() != 0;
    else if (key == "qp.sigma_setbased.activate") c.sigma_activate = n0();
    else if (key == "qp.sigma_setbased.safe") c.sigma_safe = n0();
    else if (key == "qp.sigma_setbased.exit") c.sigma_exit = n0();
    else if (key == "qp.sigma_setbased.gamma") c.sigma_gamma = n0();
    else if (key == "qp.sigma_setbased.slack_weight") c.sigma_slack_w = n0();
    else if (key == "qp.sigma_setbased.grad_eps") c.sigma_grad_eps = n0();
    else if (key == "qp.sigma_setbased.grad_period_ticks") c.sigma_grad_period = i0();
    else if (key == "qp.branch_barrier.enabled") c.branch_enabled = i0() != 0;
    else if (key == "qp.branch_barrier.activate_rad") c.branch_activate = n0();
    else if (key == "qp.branch_barrier.box_activate_rad") c.branch_box_activate = n0();
    else if (key == "qp.branch_barrier.eps_rad") c.branch_eps = n0();
    else if (key == "qp.branch_barrier.j4_limit_eps_rad") c.j4_limit_eps = n0();
    else if (key == "qp.branch_barrier.j4_limit_activate_rad") c.j4_limit_activate = n0();
    else if (key == "qp.branch_barrier.j1_overfold_abs_rad") c.j1_overfold_abs = n0();
    else if (key == "qp.branch_barrier.j1_overfold_activate_rad") c.j1_overfold_activate = n0();
    else if (key == "qp.branch_barrier.j1_overfold_eps_rad") c.j1_overfold_eps = n0();
    else if (key == "qp.branch_barrier.gamma") c.branch_gamma = n0();
    else if (key == "qp.branch_barrier.slack_weight") c.branch_slack_w = n0();
    else if (key == "qp.branch_barrier.target_eps_rad") c.branch_target_eps = n0();
    else if (key == "qp.branch_barrier.dwell_free_s") c.dwell_free_s = n0();
    else if (key == "qp.branch_barrier.dwell_ramp_s") c.dwell_ramp_s = n0();
    else if (key == "qp.branch_barrier.dwell_scale_max") c.dwell_scale_max = n0();
    else if (key == "qp.joint_comfort.enabled") c.comfort_enabled = i0() != 0;
    else if (key == "qp.joint_comfort.m_comfort_rad") c.comfort_m = n0();
    else if (key == "qp.joint_comfort.activate_rad") c.comfort_activate = n0();
    else if (key == "qp.joint_comfort.gamma") c.comfort_gamma = n0();
    else if (key == "qp.joint_comfort.slack_weight") c.comfort_slack_w = n0();
    else if (key == "qp.j4_design_comfort.enabled") c.j4_design_enabled = i0() != 0;
    else if (key == "qp.j4_design_comfort.lower_rad") c.j4_design_lo = n0();
    else if (key == "qp.j4_design_comfort.upper_rad") c.j4_design_hi = n0();
    else if (key == "qp.j4_design_comfort.gamma") c.j4_design_gamma = n0();
    else if (key == "qp.j4_design_comfort.slack_weight") c.j4_design_slack_w = n0();
    else if (key == "collision.enabled") c.collision_enabled = i0() != 0;
    else if (key == "collision.d_safe") c.d_safe = n0();
    else if (key == "collision.d_activate") c.d_activate = n0();
    else if (key == "collision.gamma") c.cbf_gamma = n0();
    else if (key == "collision.max_pairs") c.max_pairs = i0();
    else if (key == "nullspace.k_center") c.k_center = n0();
    else if (key == "nullspace.k_limit") c.k_limit = n0();
    else if (key == "nullspace.activation") c.ns_activation = n0();
    else if (key == "nullspace.q_nominal_rad") fill_vec(nums, c.q_nominal.data(), 8);
    else if (key == "arm_angle.enabled") c.arm_enabled = i0() != 0;
    else if (key == "arm_angle.k_psi") c.k_psi = n0();
    else if (key == "arm_angle.fd_eps_rad") c.fd_eps = n0();
    else if (key == "arm_angle.safe_denom_eps") c.safe_denom_eps = n0();
    else if (key == "arm_angle.obs_decay_gain") c.obs_decay_gain = n0();
    else if (key == "arm_angle.obs_smooth_floor") c.obs_smooth_floor = n0();
    else if (key == "arm_angle.max_qdot_frac") c.arm_max_qdot_frac = n0();
    else if (key == "psi_retarget.enabled") c.psi_enabled = i0() != 0;
    else if (key == "psi_retarget.psi_attr_rad") c.psi_attr = n0();
    else if (key == "psi_retarget.d_attr_m") c.d_attr = n0();
    else if (key == "psi_retarget.psi_envelope_lo_rad") c.psi_env_lo = n0();
    else if (key == "psi_retarget.psi_envelope_hi_rad") c.psi_env_hi = n0();
    else if (key == "psi_retarget.d_center_rate_m_s") c.d_center_rate = n0();
    else if (key == "psi_retarget.psi_rate_rad_s") c.psi_rate = n0();
    else if (key == "psi_retarget.rail_margin_m") c.rail_margin = n0();
    else if (key == "psi_retarget.elbow_hi_rad") c.elbow_hi = n0();
    else if (key == "psi_retarget.elbow_hi_illegal_rad") c.elbow_illegal = n0();
    else if (key == "psi_retarget.elbow_lo_rad") c.elbow_lo = n0();
    else if (key == "psi_retarget.elbow_center_rad") c.elbow_center = n0();
    else if (key == "psi_retarget.psi_cmd_lead_rad") c.psi_cmd_lead = n0();
    else if (key == "psi_retarget.psi_return_dwell_s") c.psi_return_dwell = n0();
    else if (key == "psi_retarget.psi_replan_period_s") c.psi_replan_period = n0();
    else if (key == "psi_retarget.psi_search_half_span_rad") c.psi_search_half = n0();
    else if (key == "psi_retarget.psi_search_n") c.psi_search_n = i0();
    else if (key == "psi_retarget.psi_wrist_ok_rad") c.psi_wrist_ok = n0();
    else if (key == "psi_retarget.wrist_min_rad") c.wrist_min = n0();
    else if (key == "srs.R_flange_tcp") {
      if (nums.size() >= 9) {
        for (int r = 0; r < 3; ++r)
          for (int col = 0; col < 3; ++col) c.R_flange_tcp(r, col) = nums[r * 3 + col];
        c.have_flange_tcp = true;
      }
    } else if (key == "srs.t_flange_tcp") {
      fill_vec(nums, c.t_flange_tcp.data(), 3);
      c.have_flange_tcp = true;
    } else if (key == "ird.enabled") c.ird_enabled = i0() != 0;
    else if (key == "manipulability.k_mu") c.k_mu = n0();
    else if (key == "manipulability.sigma_fade_ref") c.sigma_fade_ref = n0();
    else if (key == "rail.mode") c.rail_mode = i0();
    else if (key == "rail.locked_style") c.locked_style = i0();
    else if (key == "rail.lock_vel_eps_m_s") c.lock_vel_eps = n0();
    else if (key == "rail.v_max_m_s") c.rail_v_max = n0();
    else if (key == "rail.soft_min_m") c.soft_min = n0();
    else if (key == "rail.soft_max_m") c.soft_max = n0();
    else if (key == "rail.hard_min_m") c.hard_min = n0();
    else if (key == "rail.hard_max_m") c.hard_max = n0();
    else if (key == "rail.lock_hard_pin") c.lock_hard_pin = i0() != 0;
    else if (key == "rail.lock_reg_scale") c.lock_reg_scale = n0();
    else if (key == "rail_allocator.v0_m_s") c.v0 = n0();
    else if (key == "rail_allocator.w0_rad_s") c.w0 = n0();
    else if (key == "rail_allocator.k_margin") c.k_margin = n0();
    else if (key == "rail_allocator.kp_mid") c.kp_mid = n0();
    else if (key == "rail_allocator.ki_mid") c.ki_mid = n0();
    else if (key == "rail_allocator.u_mid_max_m_s") c.u_mid_max = n0();
    else if (key == "rail_allocator.k_err_rail") c.k_err_rail = n0();
    else if (key == "rail_allocator.e_ref_m") c.e_ref = n0();
    else if (key == "rail_allocator.f_c_hz") c.f_c_hz = n0();
    else if (key == "rail_allocator.kaw_mid") c.kaw_mid = n0();
    else if (key == "rail_allocator.rho_mirror_a") c.rho_a = n0();
    else if (key == "rail_allocator.rho_mirror_j") c.rho_j = n0();
    else if (key == "rail_allocator.reaction_s") c.rail_reaction_s = n0();
    else if (key == "rail_allocator.observer_pos_gain") c.observer_pos_gain = n0();
    else if (key == "rail_allocator.observer_vel_gain") c.observer_vel_gain = n0();
    else if (key == "rail_allocator.observer_vel_lpf_hz") c.observer_vel_lpf_hz = n0();
    else if (key == "rail_extension.enabled") c.rail_ext_enabled = i0() != 0;
    else if (key == "rail_extension.k_ext") c.k_ext = n0();
    else if (key == "rail_extension.k_ff") c.k_ff = n0();
    else if (key == "rail_extension.v_ff_thr_m_s") c.v_ff_thr = n0();
    else if (key == "rail_extension.v_ff_span_m_s") c.v_ff_span = n0();
    else if (key == "rail_extension.e0_m") c.e0_m = n0();
    else if (key == "rail_extension.e1_m") c.e1_m = n0();
    else if (key == "rail_extension.w_max") c.w_max_ext = n0();
    else if (key == "rail_extension.v_max_m_s") c.v_max_ext = n0();
    else if (key == "rail_extension.limit_margin_m") c.limit_margin = n0();
    else if (key == "rail_extension.pin_margin_m") c.pin_margin = n0();
    else if (key == "rail_extension.escape_leave_m") c.escape_leave = n0();
    else if (key == "rail_extension.soft_min_m") c.soft_min = n0();
    else if (key == "rail_extension.soft_max_m") c.soft_max = n0();
    else if (key == "rail_extension.healthy_sigma_mute") c.healthy_sigma_mute = n0();
    else if (key == "rail_extension.d_band_m") c.d_band = n0();
    else if (key == "rail_extension.k_sigma_boost") c.k_sigma_boost = n0();
    else if (key == "rail_extension.k_esc") c.k_esc = n0();
    else if (key == "rail_extension.w_sigma_floor") c.w_sigma_floor = n0();
    else if (key == "rail_extension.k_pose") c.k_pose = n0();
    else if (key == "rail_extension.pose_e0_m") c.pose_e0 = n0();
    else if (key == "rail_extension.pose_e1_m") c.pose_e1 = n0();
    else if (key == "rail_extension.pose_w_max") c.pose_w_max = n0();
    else if (key == "rail_extension.k_escape_boost") c.k_escape_boost = n0();
    else if (key == "rail_extension.d_star_err0_m") c.d_star_err0 = n0();
    else if (key == "rail_extension.d_star_err1_m") c.d_star_err1 = n0();
    else if (key == "rail_extension.d_star_w_mult") c.d_star_w_mult = n0();
    else if (key == "rail_extension.k_margin_boost") c.k_margin_boost = n0();
    else if (key == "rail_extension.w_ext_cap") c.w_ext_cap = n0();
    else if (key == "rail_extension.press_v_force_min_m_s") c.press_v_force_min = n0();
    else if (key == "rail_extension.press_dz_max_m") c.press_dz_max = n0();
    else if (key == "rail_extension.press_y_err_m") c.press_y_err = n0();
    else if (key == "rail_extension.press_stall_s") c.press_stall_s = n0();
    else if (key == "rail_extension.d_star_nudge_m") c.d_star_nudge = n0();
    else if (key == "rail_extension.open_travel_min_m") c.open_travel_min = n0();
    else if (key == "saturation.slack_enter") c.slack_enter = n0();
    else if (key == "saturation.slack_exit") c.slack_exit = n0();
    else if (key == "saturation.secondary_scale") c.secondary_scale = n0();
    else if (key == "saturation.secondary_scale_tau_s") c.secondary_scale_tau_s = n0();
  }
  return c;
}

}  // namespace wbc_rt
