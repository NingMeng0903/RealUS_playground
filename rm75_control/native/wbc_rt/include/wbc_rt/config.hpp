#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_rt/types.hpp"

namespace wbc_rt {

struct Config {
  std::string urdf;
  std::string collision_urdf;
  std::string pair_config;
  std::string control_frame = "tool";
  std::string euler_order = "xyz";
  std::string escape_sign_policy = "auto";

  double dt = 0.005;
  double feedback_timeout_s = 0.08;
  double v_scale = 0.8;
  double a_max_arm = 3.0;
  double a_max_rail = 0.60;
  double position_margin_rad = 0.005236;
  double position_margin_rail_m = 0.0;
  double resync_err_rad = 0.10472;
  double resync_err_rail_m = 0.020;
  double d_null = 0.5;
  double d_null_adaptive = 1.0;
  double max_qdot_frac = 0.2;
  double sec_target_hz = 15.0;
  double sec_input_lpf_hz = 0.0;
  double ns_grad_lpf_hz = 0.0;
  double ns_hold_fade_v = 0.0;
  double ns_hold_fade_v0 = 0.0;
  double sec_filter_zeta = 1.0;

  Vec6 task_weight = (Vec6() << 100, 100, 100, 50, 50, 50).finished();
  Vec8 reg = (Vec8() << 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1.2e-2, 1.2e-2, 1.2e-2).finished();
  Vec8 smoothness = (Vec8() << 0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15).finished();
  double eps_abs = 1e-6;
  int max_iter = 400;
  int max_iter_cap = 400;
  double max_solve_ms = 5.0;
  double fail_qdot_decay = 0.85;
  double twist_sigma_floor = 0.02;
  double task_weight_min_frac = 0.05;
  double task_weight_lpf_tau_s = 0.25;
  bool aniso_task_damping = true;
  bool use_mass_weighted_reg = true;
  double mass_reg_floor = 0.05;
  bool mass_weight_exempt_rail = true;
  double mass_reg_lpf_tau_s = 0.2;
  double damper_band_rad = 0.25;
  double damper_band_rail = 0.025;
  double rail_reaction_s = 0.06;
  double near_arm_margin_rad = 0.08;
  double j_max_arm = 300.0;
  double j_max_rail = 60.0;
  double sr_lam0 = 0.05;
  double sr_sigma_ref = 0.08;
  double sr_sigma_floor = 1e-6;

  bool sigma_enabled = true;
  double sigma_activate = 0.09;
  double sigma_safe = 0.045;
  double sigma_exit = 0.13;
  double sigma_gamma = 8.0;
  double sigma_slack_w = 200.0;
  double sigma_grad_eps = 1e-4;
  int sigma_grad_period = 10;

  bool branch_enabled = true;
  double branch_activate = 0.52;
  double branch_box_activate = 0.87;
  double branch_eps = 0.35;
  double j4_limit_eps = 0.087266;
  double j4_limit_activate = 0.436332;
  double j1_overfold_abs = 2.44346;
  double j1_overfold_activate = 0.436332;
  double j1_overfold_eps = 0.0;
  double branch_gamma = 6.0;
  double branch_slack_w = 80.0;
  double branch_target_eps = 1e-3;
  double dwell_free_s = 0.3;
  double dwell_ramp_s = 1.0;
  double dwell_scale_max = 5.0;

  bool comfort_enabled = true;
  double comfort_m = 0.261799;
  double comfort_activate = 0.436332;
  double comfort_gamma = 6.0;
  double comfort_slack_w = 80.0;

  bool j4_design_enabled = true;
  double j4_design_lo = 1.2217304763960306;   // 70°
  double j4_design_hi = 2.007128639793479;    // 115°
  double j4_design_gamma = 4.0;
  double j4_design_slack_w = 60.0;

  bool collision_enabled = true;
  double d_safe = 0.01;
  double d_activate = 0.04;
  double cbf_gamma = 5.0;
  int max_pairs = 8;

  double k_center = 1.0;
  double k_limit = 2.0;
  double ns_activation = 0.8;
  Vec8 q_nominal = Vec8::Zero();

  bool arm_enabled = true;
  double k_psi = 1.0;
  double fd_eps = 1e-4;
  double safe_denom_eps = 1e-4;
  double obs_decay_gain = 400.0;
  double obs_smooth_floor = 0.3;
  double arm_max_qdot_frac = 0.15;

  bool psi_enabled = true;
  double psi_attr = 1.1868238913561442;          // 68°
  double d_attr = -0.185;
  double psi_env_lo = 0.6981317007977318;        // 40°
  double psi_env_hi = 1.9198621771937625;        // 110°
  double d_center_rate = 0.02;
  double psi_rate = 0.4363323129985824;          // 25°/s
  double rail_margin = 0.02;
  double elbow_hi = 2.007128639793479;       // 115°
  double elbow_illegal = 2.2689280275926285;  // 130°
  double elbow_lo = 1.2217304763960306;      // 70°
  double elbow_center = 1.6580627893946132;  // 95°
  double psi_cmd_lead = 0.3141592653589793;  // 18°
  double psi_return_dwell = 1.0;
  double psi_replan_period = 0.1;
  double psi_search_half = 0.7853981633974483;  // 45°
  int psi_search_n = 9;
  double psi_wrist_ok = 0.6981317007977318;  // 40°
  double wrist_min = 0.5235987755982988;     // 30°
  Eigen::Matrix3d R_flange_tcp = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_flange_tcp = Eigen::Vector3d(0.0, 0.0, 0.22);
  bool have_flange_tcp = false;
  Eigen::Matrix3d R_tcp_placement = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_tcp_placement = Eigen::Vector3d::Zero();
  bool have_tcp_placement = false;
  bool ird_enabled = false;

  const Eigen::Matrix3d* tcp_placement_R() const {
    return have_tcp_placement ? &R_tcp_placement : nullptr;
  }
  const Eigen::Vector3d* tcp_placement_t() const {
    return have_tcp_placement ? &t_tcp_placement : nullptr;
  }

  double k_mu = 0.0;
  double sigma_fade_ref = 0.08;

  int rail_mode = 0;
  int locked_style = 0;
  double lock_vel_eps = 0.0;
  double rail_v_max = 0.15;
  double soft_min = 0.030;
  double soft_max = 0.755;
  double hard_min = 0.005;
  double hard_max = 0.78;
  bool lock_hard_pin = true;
  double lock_reg_scale = 100.0;

  double v0 = 0.05;
  double w0 = 0.30;
  double k_margin = 4.0;
  double kp_mid = 1.2;
  double ki_mid = 0.80;
  double u_mid_max = 0.12;
  double k_err_rail = 4.0;
  double e_ref = 0.08;
  double f_c_hz = 1.0;
  double kaw_mid = 8.0;
  double rho_a = 0.50;
  double rho_j = 0.30;
  double observer_pos_gain = 0.35;
  double observer_vel_gain = 2.0;
  double observer_vel_lpf_hz = 8.0;

  bool rail_ext_enabled = true;
  double k_ext = 1.0;
  double k_ff = 1.0;
  double v_ff_thr = 0.01;
  double v_ff_span = 0.03;
  double e0_m = 0.05;
  double e1_m = 0.15;
  double w_max_ext = 1.5;
  double v_max_ext = 0.08;
  double limit_margin = 0.15;
  double pin_margin = 0.008;
  double escape_leave = 0.04;
  double healthy_sigma_mute = 0.08;
  double d_band = 0.005;
  double k_sigma_boost = 2.0;
  double k_esc = 0.5;
  double w_sigma_floor = 1.0;
  double k_pose = 2.0;
  double pose_e0 = 0.005;
  double pose_e1 = 0.04;
  double pose_w_max = 4.0;
  double k_escape_boost = 1.2;
  double k_margin_boost = 4.0;
  double w_ext_cap = 24.0;
  double d_star_err0 = 0.01;
  double d_star_err1 = 0.04;
  double d_star_w_mult = 6.0;
  double press_v_force_min = 0.02;
  double press_dz_max = 0.002;
  double press_y_err = 0.005;
  double press_stall_s = 0.5;
  double d_star_nudge = 0.01;
  double open_travel_min = 0.01;

  double slack_enter = 0.15;
  double slack_exit = 0.03;
  double secondary_scale = 0.15;
  double secondary_scale_tau_s = 0.10;

  static Config load(const std::string& path);
};

}  // namespace wbc_rt
