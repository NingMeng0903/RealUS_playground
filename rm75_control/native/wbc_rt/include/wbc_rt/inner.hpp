#pragma once

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/multibody/geometry.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

#include "wbc_rt/config.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/posture.hpp"
#include "wbc_rt/protocol.hpp"
#include "wbc_rt/task_weight.hpp"

namespace wbc_rt {

struct TickIn {
  Vec6 v_cmd = Vec6::Zero();
  Vec8 q_meas = Vec8::Zero();
  Vec8 qdot_ff = Vec8::Zero();
  Vec6 pose_d = Vec6::Zero();
  Vec6 vel_ff = Vec6::Zero();
  Vec6 path_twist = Vec6::Zero();
  Vec6 feedback_twist = Vec6::Zero();
  double dt_nom = 0.005;
  double dt_wall = 0.005;
  double t_mono = 0.0;
  double rail_v = 0.0;
  double v_force_z = 0.0;
  double posture_d = std::numeric_limits<double>::quiet_NaN();
  double posture_psi = std::numeric_limits<double>::quiet_NaN();
  Vec8 posture_q = Vec8::Zero();
  uint32_t flags = 0;
};

struct TickOut {
  Vec8 q_cmd = Vec8::Zero();
  Vec8 qdot = Vec8::Zero();
  Vec6 v_recv = Vec6::Zero();
  Vec6 v_feas = Vec6::Zero();
  Vec6 v_tcp = Vec6::Zero();
  Vec6 residual = Vec6::Zero();
  double slack = 0.0;
  double e_qp = 0.0;
  double u_alloc = 0.0;
  double u_mid = 0.0;
  double v_r_ref = 0.0;
  double psi = 0.0;
  double d_star = 0.0;
  double d_pref = 0.0;
  double solve_ms = 0.0;
  double sigma_min = 0.0;
  double sigma_arm = 0.0;
  uint32_t flags = 0;
  uint32_t joint_limited = 0;
  uint32_t rail_limited = 0;
  uint32_t wall_active = 0;
  uint32_t secondary_suppressed = 0;
  uint32_t status = kStatusOk;
  double ns_norm = 0.0;
  double ns_centering = 0.0;
  double ns_manip = 0.0;
  double ns_arm_angle = 0.0;
  double ns_damping = 0.0;
  double ns_rail_lock = 0.0;
  double sat_scale = 1.0;
  double sec_target_norm = 0.0;
  double homotopy_s = 0.0;
  double psi_star = 0.0;
  double rail_motion_share = std::numeric_limits<double>::quiet_NaN();
  // Mixer telemetry (SHM v4). V_d_proxy is a configuration-error storage
  // proxy: 0.5 * kp_mid * e_d^2. kp_mid is s^-1, not stiffness, not joules.
  double u_task_raw = 0.0;
  double u_task_feasible = 0.0;
  double u_pi_raw = 0.0;
  double u_mid_cmd = 0.0;
  double u_post_raw = 0.0;
  double u_post_feasible = 0.0;
  double u_mid_applied = 0.0;
  double d_star_dot_cmd = 0.0;
  double u_escape_raw = 0.0;
  double u_escape_feasible = 0.0;
  double escape_active = 0.0;
  double escape_dir = 0.0;
  double u_base = 0.0;
  double u_feasible = 0.0;
  double v_r_lpf = 0.0;
  double e_d = 0.0;
  double V_d_proxy = 0.0;
  double j4_design_slack = 0.0;
  double sigma_slack = 0.0;
  double rail_box_lo = 0.0;
  double rail_box_hi = 0.0;
  uint32_t rail_bind_lo = 0;
  uint32_t rail_bind_hi = 0;
  double rail_task_vel_used = 0.0;
  double rail_h1 = 0.0;
  double rail_h2 = 0.0;
  double rail_qdot_prev = 0.0;
  double rail_qdot_prev2 = 0.0;
  uint32_t qp1_status = kQpNotRun;
  uint32_t qp2_status = kQpNotRun;
  uint32_t fallback_level = kFallbackNone;
  uint32_t failure_code = kFailureNone;
  double qp1_hard_violation = 0.0;
  double final_hard_violation = 0.0;
  double task_lock_violation = 0.0;
  double final_box_violation = 0.0;
  uint32_t qp_overrun = 0;
  double posture_gate = 1.0;
};

class Collision {
 public:
  Collision(pinocchio::Model& model, const Config& cfg);
  void update(const Vec8& q, pinocchio::Data& data);
  int build_rows(pinocchio::Data& data, MatX* jac, VecX* lower, std::vector<int>* slots);

 private:
  pinocchio::Model* model_ = nullptr;
  pinocchio::GeometryModel geom_model_;
  pinocchio::GeometryData geom_data_;
  Config cfg_;
  std::vector<int> slots_;
};

class InnerLoop {
 public:
  explicit InnerLoop(const Config& cfg);

  void enable();
  void stop();
  void reset(const Vec8& q0);
  void begin_hybrid(const Vec8& q_meas, const Vec8& qdot_applied);
  void set_rail_mode(uint32_t mode, uint32_t style, double q_ref, bool has_ref);
  void set_flags(uint32_t bits);
  void set_stroke(double d_star, double psi_star);
  std::pair<double, double> plan_stroke(const Vec8& q, double y_center, double amp);
  void set_rail_pose_target(double y, bool valid);
  void capture_rail_ext_ref(const Vec8& q);
  void set_rail_ext_mode(int pose_attract);

  TickOut step(const TickIn& in);

  const Vec8& q_cmd() const { return q_cmd_; }

 private:
  bool apply_velocity_box(const Vec8& q_geom, const Vec8& q_cmd, const Vec8& q_meas,
                          double dt, double h1, double h2, bool rail_locked,
                          double rail_pin, bool has_pin, bool lead_exempt,
                          Vec8* lo, Vec8* hi);
  void tighten_branch(const Vec8& q, bool rail_open, Vec8* lo, Vec8* hi);
  void clear_rail_box_tel();
  void note_rail_bind(double old_lo, double old_hi, const Vec8& lo, const Vec8& hi,
                      uint32_t stage);
  bool solve_hqp(const Mat6x8& J, const Vec6& v_cmd, const Vec8& q_geom,
                 const Vec8& q_prev, const Vec8& qdot_nom, double rail_exec,
                 bool has_rail_exec, double rail_task_vel, double rail_w,
                 bool rail_locked, double dt, double h1, double h2,
                 bool rail_open, double rail_pin, bool has_pin, bool lead_exempt,
                 double sigma_arm, bool direct_pin, Vec8* qdot, Vec6* residual, double* slack);

  Config cfg_;
  Kinematics kin_;
  PostureRetarget posture_;
  std::unique_ptr<Collision> collision_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp1_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp2_;
  bool qp1_inited_ = false;
  bool qp2_inited_ = false;
  bool qp1_last_ok_ = false;
  bool qp2_last_ok_ = false;

  Vec8 q_cmd_ = Vec8::Zero();
  Vec8 qdot_prev_ = Vec8::Zero();
  Vec8 qdot_seen_ = Vec8::Zero();
  Vec8 qdot_prev2_ = Vec8::Zero();
  Vec8 dq_prev_ = Vec8::Zero();
  bool have_dq_prev_ = false;
  Vec8 q_lo_ = Vec8::Zero();
  Vec8 q_hi_ = Vec8::Zero();
  Vec8 v_max_ = Vec8::Ones();
  Vec8 a_max_ = Vec8::Ones();
  Vec8 j_max_ = Vec8::Ones();
  Vec8 q_mid_ = Vec8::Zero();
  Vec8 half_ = Vec8::Ones();
  Vec8 q_star_ = Vec8::Zero();
  Vec8 q_star_signs_ = Vec8::Zero();
  Vec8 q_nominal_ = Vec8::Zero();
  Vec8 last_valid_q_star_ = Vec8::Zero();
  bool have_valid_q_star_ = false;
  Vec8 m_diag_lpf_ = Vec8::Ones();
  bool m_diag_init_ = false;

  Vec8 sec_qdot_ = Vec8::Zero();
  Vec8 sec_acc_ = Vec8::Zero();
  Vec8 sec_target_ = Vec8::Zero();
  Vec8 sec_lpf_ = Vec8::Zero();
  Vec8 gN_lpf_ = Vec8::Zero();
  bool gN_lpf_init_ = false;
  double sec_age_ = 1e9;

  double v_r_ref_ = 0.0;
  double v_r_a_ = 0.0;
  double v_r_lpf_ = 0.0;
  bool v_r_init_ = false;
  bool wall_pi_frozen_ = false;
  double u_alloc_ = 0.0;
  double u_mid_ = 0.0;
  double u_mid_committed_ = 0.0;
  double mid_integ_ = 0.0;
  double u_task_raw_ = 0.0;
  double u_task_feasible_ = 0.0;
  double u_pi_raw_ = 0.0;
  double u_mid_cmd_ = 0.0;
  double u_post_raw_ = 0.0;
  double u_post_feasible_ = 0.0;
  double u_mid_applied_ = 0.0;
  double d_star_dot_cmd_ = 0.0;
  double u_escape_raw_ = 0.0;
  double u_escape_feasible_ = 0.0;
  double u_base_ = 0.0;
  double u_feasible_ = 0.0;
  double e_d_ = 0.0;
  double V_d_proxy_ = 0.0;
  double j4_design_slack_ = 0.0;
  double sigma_slack_ = 0.0;
  double rail_box_lo_ = 0.0;
  double rail_box_hi_ = 0.0;
  uint32_t rail_bind_lo_ = 0;
  uint32_t rail_bind_hi_ = 0;
  double rail_task_vel_used_ = 0.0;
  double rail_h1_ = 0.0;
  double rail_h2_ = 0.0;
  double rail_qdot_prev_tel_ = 0.0;
  double rail_qdot_prev2_tel_ = 0.0;
  bool posture_gate_active_ = false;
  double posture_gate_scale_ = 0.0;
  double posture_gate_enter_s_ = 0.0;
  double posture_gate_exit_s_ = 0.0;
  uint32_t qp1_status_ = kQpNotRun;
  uint32_t qp2_status_ = kQpNotRun;
  uint32_t fallback_level_ = kFallbackNone;
  uint32_t failure_code_ = kFailureNone;
  double qp1_hard_violation_ = 0.0;
  double final_hard_violation_ = 0.0;
  double task_lock_violation_ = 0.0;
  double final_box_violation_ = 0.0;
  bool qp_overrun_ = false;
  double q_hat_ = 0.0;
  double v_hat_ = 0.0;
  bool obs_init_ = false;
  double last_sample_t_ = -1.0;

  double last_slack_ = 0.0;
  bool slack_hold_latched_ = false;
  double sat_scale_ = 1.0;
  double last_sigma_ = 0.08;
  double quiet_s_ = 0.0;
  double cmd_quiet_s_ = 0.0;
  bool quiescent_ = false;
  bool hold_d_prev_ = false;
  bool enabled_ = true;
  Vec6 last_tcp_est_ = Vec6::Zero();

  int rail_mode_ = 0;
  int locked_style_ = 0;
  double rail_q_ref_ = 0.0;
  bool has_rail_ref_ = false;
  bool plan_drives_rail_ = false;
  bool direct_ptp_ = false;
  bool arm_suppress_ = false;
  bool center_suppress_ = false;
  bool manip_active_ = false;
  bool rail_ext_active_ = true;
  int rail_ext_mode_ = 0;
  double y_rail_target_ = 0.0;
  bool has_y_target_ = false;

  double d_star_ = 0.0;
  double d_pref_ = 0.0;
  double d_star_ref_ = 0.0;
  bool d_star_ref_init_ = false;
  double psi_cmd_ = 0.0;
  double psi_star_ = 0.0;
  double homotopy_s_ = 0.0;
  bool planned_ = false;
  double d0_ = 0.0;
  double psi0_ = 0.0;

  double press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
  double press_stall_s_ = 0.0;
  double nudge_cool_s_ = 0.0;

  bool escape_active_ = false;
  int escape_dir_ = 0;
  double escape_sign_ = 0.0;
  double last_e_mid_ = 0.0;
  double last_v_escape_ = 0.0;
  double last_v_ff_ = 0.0;
  double last_ext_w_ = 0.0;
  bool last_limit_sat_ = false;
  double last_d_star_reg_ = 1.0;

  double dwell_s_ = 0.0;
  double dwell_scale_ = 1.0;
  bool sigma_row_active_ = false;
  Vec8 sigma_grad_ = Vec8::Zero();
  int sigma_tick_ = 0;
  bool box_t_init_ = false;
  double box_last_t_ = 0.0;
  double box_h1_ = 0.005;

  Mat6x8 last_lock_J_ = Mat6x8::Zero();
  Vec6 last_lock_v_ = Vec6::Zero();
  MatX last_C_;
  VecX last_lo_;
  VecX last_hi_;
  VecX last_x_;
  TaskWeightState task_weight_;
};

}  // namespace wbc_rt
