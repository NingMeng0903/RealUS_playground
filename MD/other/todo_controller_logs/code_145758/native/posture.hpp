#pragma once

#include <optional>

#include "wbc_rt/config.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/srs_ik.hpp"
#include "wbc_rt/types.hpp"

namespace wbc_rt {

class PostureRetarget {
 public:
  explicit PostureRetarget(const Config& cfg);

  void reset(const Vec8& q, const Vec6& pose);
  void set_planned_stroke(double d_star, double psi_star);
  void begin_unplanned(const Vec8& q, const Vec6& pose);
  double nudge_d_star(double delta_m, double y_des_m, double rail_lo, double rail_hi,
                      double dt);

  // Matches Python PostureRetarget.step.  Updates d*, ψ*, q*.
  void step(const Vec8& q, const Vec6& pose, double dt, double rail_lo, double rail_hi,
            bool hold_setpoint);

  bool planned() const { return planned_; }
  double d_star() const { return d_star_; }
  double psi_cmd() const { return psi_cmd_; }
  double psi_star() const { return psi_star_; }
  double homotopy_s() const { return s_; }
  const Vec8& q_star() const { return q_star_; }

 private:
  struct Pack {
    Eigen::Matrix<double, 7, 1> q_arm;
    Vec8 q_full;
  };

  std::optional<Pack> eval_at(const Vec6& pose, double psi, double d) const;
  std::optional<Pack> eval_at(const Vec6& pose, double psi, double d, int branch) const;
  std::optional<double> select_d_for_elbow(const Vec8& q, const Vec6& pose, double psi,
                                           double rail_lo, double rail_hi) const;
  void advance_homotopy(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                        double rail_hi, double live_psi);
  void maybe_retarget_psi(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                          double rail_hi);
  std::optional<double> search_psi(const Vec8& q, const Vec6& pose, double rail_lo,
                                   double rail_hi) const;
  double rate_limit_psi(double dt, double live_psi);
  double rate_limit_d(double dt);
  std::optional<std::pair<double, double>> rail_window(double y_tcp, double rail_lo,
                                                       double rail_hi) const;
  std::optional<double> clip_d(double d, double y_tcp, double rail_lo, double rail_hi,
                               double d_live) const;
  bool j4_in_design(double j4, bool loose) const;
  bool j4_illegal(double j4, bool has_travel) const;
  bool q_star_ok(const Eigen::Matrix<double, 7, 1>& q_arm, const Vec8& q_live, const Vec6& pose,
                 double rail_lo, double rail_hi) const;
  double homotopy_T(double d0, double d_goal, double psi0, double psi_goal) const;

  const Eigen::Matrix3d* flange_R() const;
  const Eigen::Vector3d* flange_t() const;

  Config cfg_;
  int branch_ = 0;
  bool planned_ = false;
  bool held_prev_ = false;
  double d_star_ = 0.0;
  double d0_ = 0.0;
  double d_center_target_ = 0.0;
  double psi_cmd_ = 0.0;
  double psi0_ = 0.0;
  double psi_star_ = 0.0;
  double s_ = 0.0;
  double search_age_s_ = 0.0;
  double healthy_dwell_s_ = 0.0;
  Vec8 q_star_ = Vec8::Zero();
};

}  // namespace wbc_rt
