#include "wbc_rt/posture.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace wbc_rt {

PostureRetarget::PostureRetarget(const Config& cfg) : cfg_(cfg) {}

const Eigen::Matrix3d* PostureRetarget::flange_R() const {
  return cfg_.have_flange_tcp ? &cfg_.R_flange_tcp : nullptr;
}
const Eigen::Vector3d* PostureRetarget::flange_t() const {
  return cfg_.have_flange_tcp ? &cfg_.t_flange_tcp : nullptr;
}

void PostureRetarget::reset(const Vec8& q, const Vec6& pose) {
  const double psi = fold_psi_to_positive(srs::psi_from_q(q));
  psi_cmd_ = psi;
  psi0_ = psi;
  psi_star_ = clamp_psi_to_envelope(cfg_.psi_attr, cfg_.psi_env_lo, cfg_.psi_env_hi);
  d_star_ = pose[1] - q[0];
  d0_ = d_star_;
  d_center_target_ = cfg_.d_attr;
  s_ = 0.0;
  branch_ = srs::branch_from_q(q);
  q_star_ = q;
  planned_ = false;
  held_prev_ = false;
  psi_dot_ = 0.0;
  d_dot_ = 0.0;
  s_dot_ = 0.0;
  search_age_s_ = 0.0;
  healthy_dwell_s_ = 0.0;
}

void PostureRetarget::set_planned_stroke(double d_star, double psi_star) {
  d_star_ = d_star;
  d_center_target_ = d_star;
  psi_star_ = psi_star;
  planned_ = true;
}

void PostureRetarget::begin_unplanned(const Vec8& q, const Vec6& pose) {
  reset(q, pose);
}

void PostureRetarget::follow_live(const Vec8& q, const Vec6& pose, double dt) {
  const double live_psi = fold_psi_to_positive(srs::psi_from_q(q));
  const double d_live = pose[1] - q[0];
  dt = std::max(dt, 0.0);
  const double psi_cur = fold_psi_to_positive(psi_cmd_);
  auto [nxt, v_psi] = track_bounded(psi_cur, psi_dot_, live_psi, dt, cfg_.psi_rate, cfg_.psi_accel,
                                    psi_err_avoiding_zero(psi_cur, live_psi));
  psi_cmd_ = fold_psi_to_positive(nxt);
  psi_dot_ = v_psi;
  auto [d_nxt, v_d] = track_bounded(d_star_, d_dot_, d_live, dt, cfg_.d_center_rate, cfg_.d_center_accel);
  d_star_ = d_nxt;
  d_dot_ = v_d;
  d0_ = d_star_;
  psi0_ = psi_cmd_;
  s_ = 0.0;
  s_dot_ = 0.0;
}

std::optional<PostureRetarget::Pack> PostureRetarget::eval_at(const Vec6& pose, double psi,
                                                             double d) const {
  return eval_at(pose, psi, d, branch_);
}

std::optional<PostureRetarget::Pack> PostureRetarget::eval_at(const Vec6& pose, double psi,
                                                             double d, int branch) const {
  const double y_urdf = pose[1] - d;
  const double y_s = srs::shoulder_y_from_q_rail(y_urdf);
  const auto q_arm = srs::srs_ik(pose, psi, branch, y_s, flange_R(), flange_t());
  if (!q_arm) return std::nullopt;
  Pack p;
  p.q_arm = *q_arm;
  p.q_full[0] = y_urdf;
  p.q_full.tail<7>() = *q_arm;
  return p;
}

std::optional<std::pair<double, double>> PostureRetarget::rail_window(double y_tcp, double rail_lo,
                                                                     double rail_hi) const {
  const double margin = std::max(cfg_.rail_margin, 0.0);
  const double y_lo = rail_lo + margin;
  const double y_hi = rail_hi - margin;
  if (y_lo > y_hi + 1e-12) return std::nullopt;
  const double d_lo = y_tcp - y_hi;
  const double d_hi = y_tcp - y_lo;
  if (d_lo > d_hi + 1e-12) return std::nullopt;
  return std::make_pair(d_lo, d_hi);
}

std::optional<double> PostureRetarget::clip_d(double d, double y_tcp, double rail_lo, double rail_hi,
                                             double d_live) const {
  const auto w = rail_window(y_tcp, rail_lo, rail_hi);
  if (!w) {
    if (std::isfinite(d_live)) return d_live;
    return std::nullopt;
  }
  return clip(d, w->first, w->second);
}

bool PostureRetarget::j4_in_design(double j4, bool loose) const {
  double lo = cfg_.elbow_lo;
  double hi = cfg_.elbow_hi;
  if (loose) {
    lo -= 5.0 * M_PI / 180.0;
    hi += 7.0 * M_PI / 180.0;
  }
  return lo - 1e-9 <= j4 && j4 <= hi + 1e-9;
}

bool PostureRetarget::j4_illegal(double j4, bool has_travel) const {
  if (!has_travel) return false;
  return std::abs(j4) >= cfg_.elbow_illegal - 1e-9;
}

bool PostureRetarget::q_star_ok(const Eigen::Matrix<double, 7, 1>& q_arm, const Vec8& /*q_live*/,
                               const Vec6& pose, double rail_lo, double rail_hi) const {
  const auto w = rail_window(pose[1], rail_lo, rail_hi);
  const bool has_travel = w && (w->second - w->first) > 0.01;
  return !j4_illegal(q_arm[3], has_travel);
}

double PostureRetarget::homotopy_T(double d0, double d_goal, double psi0, double psi_goal) const {
  const double d_rate = std::max(cfg_.d_center_rate, 1e-9);
  const double psi_rate = std::max(cfg_.psi_rate, 1e-9);
  const double t_d = std::abs(d_goal - d0) / d_rate;
  const double t_psi = std::abs(psi_err_avoiding_zero(psi0, psi_goal)) / psi_rate;
  return std::max({t_d, t_psi, 1e-6});
}

std::optional<double> PostureRetarget::select_d_for_elbow(const Vec8& q, const Vec6& pose,
                                                         double psi, double rail_lo,
                                                         double rail_hi) const {
  const auto w = rail_window(pose[1], rail_lo, rail_hi);
  if (!w) return std::nullopt;
  const double d_lo = w->first;
  const double d_hi = w->second;
  const double d_pref = std::isfinite(d_center_target_) ? d_center_target_ : cfg_.d_attr;
  const bool has_travel = (d_hi - d_lo) > 0.01;
  std::vector<double> samples;
  for (int i = 0; i < 11; ++i) {
    const double a = static_cast<double>(i) / 10.0;
    samples.push_back(d_lo + a * (d_hi - d_lo));
  }
  for (double extra : {d_pref, d_star_, d0_}) {
    if (std::isfinite(extra) && d_lo - 1e-9 <= extra && extra <= d_hi + 1e-9) samples.push_back(extra);
  }
  std::sort(samples.begin(), samples.end());
  samples.erase(std::unique(samples.begin(), samples.end(),
                            [](double a, double b) { return std::abs(a - b) < 1e-12; }),
                samples.end());
  const double sign_pref = -1.0;
  const double j4_c = cfg_.elbow_center;
  double best_d = std::numeric_limits<double>::quiet_NaN();
  double best_cost = 1e300;
  double fallback_d = std::numeric_limits<double>::quiet_NaN();
  double fallback_cost = 1e300;
  for (double d : samples) {
    const auto pack = eval_at(pose, psi, d);
    if (!pack) continue;
    const double j4 = pack->q_arm[3];
    const double j1 = pack->q_arm[0];
    if (j4_illegal(j4, has_travel)) continue;
    double sign_pen = 0.0;
    if (std::abs(j1) > 10.0 * M_PI / 180.0 && j1 * sign_pref < 0.0) sign_pen = 10.0;
    const double cost = std::abs(d - d_pref) + 0.15 * std::abs(j4 - j4_c) + sign_pen;
    if (cost < fallback_cost) {
      fallback_cost = cost;
      fallback_d = d;
    }
    if (!j4_in_design(j4, false)) continue;
    if (cost < best_cost) {
      best_cost = cost;
      best_d = d;
    }
  }
  (void)q;
  if (std::isfinite(best_d)) return best_d;
  if (std::isfinite(fallback_d)) return fallback_d;
  return std::nullopt;
}

double PostureRetarget::rate_limit_d(double dt) {
  const double target = std::isfinite(d_center_target_) ? d_center_target_ : d_star_;
  auto [d_nxt, v_d] =
      track_bounded(d_star_, d_dot_, target, dt, cfg_.d_center_rate, cfg_.d_center_accel);
  d_star_ = d_nxt;
  d_dot_ = v_d;
  return d_star_;
}

double PostureRetarget::nudge_d_star(double delta_m, double y_des_m, double rail_lo,
                                    double rail_hi, double dt) {
  double d_lo = y_des_m - rail_hi;
  double d_hi = y_des_m - rail_lo;
  if (d_lo > d_hi) std::swap(d_lo, d_hi);
  d_center_target_ = clip(d_star_ + delta_m, d_lo, d_hi);
  return rate_limit_d(dt);
}

double PostureRetarget::rate_limit_psi(double dt, double live_psi) {
  const double target = fold_psi_to_positive(psi_star_);
  const double cur = fold_psi_to_positive(psi_cmd_);
  auto [nxt0, v_psi] = track_bounded(cur, psi_dot_, target, dt, cfg_.psi_rate, cfg_.psi_accel,
                                     psi_err_avoiding_zero(cur, target));
  double nxt = nxt0;
  psi_dot_ = v_psi;
  if (cur * nxt < 0.0 && std::abs(cur) > 1e-6) {
    nxt = std::copysign(1e-6, cur);
    psi_dot_ = 0.0;
  }
  nxt = fold_psi_to_positive(nxt);
  const double lead = std::max(cfg_.psi_cmd_lead, 0.0);
  if (lead > 0.0 && std::isfinite(live_psi)) {
    const double live = fold_psi_to_positive(live_psi);
    const double lead_nxt = std::abs(psi_err_avoiding_zero(live, nxt));
    const double lead_cur = std::abs(psi_err_avoiding_zero(live, cur));
    if (lead_nxt > lead + 1e-12 && lead_nxt > lead_cur + 1e-12) {
      nxt = cur;
      psi_dot_ = 0.0;
    }
  }
  psi_cmd_ = nxt;
  return psi_cmd_;
}

void PostureRetarget::advance_homotopy(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                                      double rail_hi, double live_psi) {
  const double psi_goal = fold_psi_to_positive(psi_star_);
  const auto d_goal = select_d_for_elbow(q, pose, psi_goal, rail_lo, rail_hi);
  if (!d_goal) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  const double T = homotopy_T(d0_, *d_goal, psi0_, psi_goal);
  const double s_dot_nom = 1.0 / T;
  const double ramp = std::max(cfg_.homotopy_ramp_s, 1e-6);
  const double a_s = s_dot_nom / ramp;
  const double s_dot_tgt = (s_ >= 1.0 - 1e-12) ? 0.0 : s_dot_nom;
  if (s_dot_ < s_dot_tgt) s_dot_ = std::min(s_dot_tgt, s_dot_ + a_s * dt);
  else if (s_dot_ > s_dot_tgt) s_dot_ = std::max(s_dot_tgt, s_dot_ - a_s * dt);
  const double s_try = std::min(1.0, s_ + s_dot_ * dt);
  const double d_live = pose[1] - q[0];
  auto d_try = clip_d(d0_ + s_try * (*d_goal - d0_), pose[1], rail_lo, rail_hi, d_live);
  if (!d_try) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  auto [d_nxt, v_d] =
      track_bounded(d_star_, d_dot_, *d_try, dt, cfg_.d_center_rate, cfg_.d_center_accel);
  *d_try = d_nxt;
  d_dot_ = v_d;
  const double psi_s = fold_psi_to_positive(psi0_ + s_try * psi_err_avoiding_zero(psi0_, psi_goal));
  const auto pack = eval_at(pose, psi_s, *d_try);
  if (!pack || !q_star_ok(pack->q_arm, q, pose, rail_lo, rail_hi)) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  s_ = s_try;
  d_star_ = *d_try;
  q_star_ = pack->q_full;
  rate_limit_psi(dt, live_psi);
}

void PostureRetarget::maybe_retarget_psi(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                                        double rail_hi) {
  search_age_s_ += std::max(dt, 0.0);
  const double period = std::max(cfg_.psi_replan_period, 0.0);
  const bool due = search_age_s_ + 1e-12 >= period;
  const double j4 = std::abs(q[4]);
  const double j6 = std::abs(q[6]);
  const double attr = clamp_psi_to_envelope(cfg_.psi_attr, cfg_.psi_env_lo, cfg_.psi_env_hi);
  if (j4 < cfg_.psi_env_lo) return;
  if (j6 < cfg_.psi_wrist_ok) {
    healthy_dwell_s_ = 0.0;
    if (!due) return;
    search_age_s_ = 0.0;
    const auto found = search_psi(q, pose, rail_lo, rail_hi);
    if (found) psi_star_ = *found;
    return;
  }
  healthy_dwell_s_ += dt;
  if (due) search_age_s_ = 0.0;
  if (healthy_dwell_s_ + 1e-12 >= std::max(cfg_.psi_return_dwell, 0.0)) psi_star_ = attr;
}

std::optional<double> PostureRetarget::search_psi(const Vec8& q, const Vec6& pose, double rail_lo,
                                                 double rail_hi) const {
  const int branch = srs::branch_from_q(q);
  const double d_c = d_star_;
  const double y_rail = pose[1] - d_c;
  const double margin = std::max(cfg_.rail_margin, 0.0);
  if (y_rail < rail_lo + margin || y_rail > rail_hi - margin) return std::nullopt;
  const double lo = cfg_.psi_env_lo;
  const double hi = cfg_.psi_env_hi;
  double center = clamp_psi_to_envelope(psi_star_, lo, hi);
  const double half = std::max(cfg_.psi_search_half, 0.0);
  const int n = std::max(cfg_.psi_search_n, 3);
  auto score = [&](const std::vector<double>& samples) -> std::pair<double, double> {
    double best_s = -1e300;
    double best_psi = std::numeric_limits<double>::quiet_NaN();
    double best_j6 = std::numeric_limits<double>::quiet_NaN();
    for (double psi : samples) {
      const auto pack = eval_at(pose, psi, d_c, branch);
      if (!pack) continue;
      const double j6 = std::abs(pack->q_arm[5]);
      if (j6 < cfg_.wrist_min - 1e-9) continue;
      double marg = 1e9;
      for (int i = 0; i < 7; ++i) {
        marg = std::min(marg, std::min(pack->q_arm[i] - srs::q_lo()[i],
                                       srs::q_hi()[i] - pack->q_arm[i]));
      }
      const double score_v =
          std::min(j6 / (60.0 * M_PI / 180.0), 1.0) + 0.8 * std::min(marg / (30.0 * M_PI / 180.0), 1.0);
      if (score_v > best_s + 1e-9) {
        best_s = score_v;
        best_psi = psi;
        best_j6 = j6;
      }
    }
    return {best_psi, best_j6};
  };
  std::vector<double> local;
  for (int i = 0; i < n; ++i) {
    const double a = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
    local.push_back(clamp_psi_to_envelope(center - half + a * 2.0 * half, lo, hi));
  }
  std::sort(local.begin(), local.end());
  local.erase(std::unique(local.begin(), local.end(),
                          [](double a, double b) { return std::abs(a - b) < 1e-12; }),
              local.end());
  auto [best_psi, best_j6] = score(local);
  if (!std::isfinite(best_psi) || !std::isfinite(best_j6) || best_j6 < cfg_.psi_wrist_ok) {
    std::vector<double> full;
    for (int i = 0; i < n; ++i) {
      const double a = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
      full.push_back(lo + a * (hi - lo));
    }
    auto [bp, bj] = score(full);
    if (std::isfinite(bp) && (!std::isfinite(best_psi) || bj > best_j6 + 1e-9)) {
      best_psi = bp;
      best_j6 = bj;
    }
  }
  if (std::isfinite(best_psi)) return best_psi;
  return std::nullopt;
}

void PostureRetarget::step(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                           double rail_hi, bool hold_setpoint, double rate_scale) {
  (void)hold_setpoint;
  dt = std::max(dt, 0.0) * std::min(1.0, std::max(0.0, rate_scale));
  const double live_psi = fold_psi_to_positive(srs::psi_from_q(q));
  if (planned_) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  held_prev_ = false;
  maybe_retarget_psi(q, pose, dt, rail_lo, rail_hi);
  advance_homotopy(q, pose, dt, rail_lo, rail_hi, live_psi);
}

}  // namespace wbc_rt
