#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <Eigen/Dense>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace wbc_rt {

constexpr int kNv = 8;
constexpr int kTask = 6;
constexpr int kNPref = 9;
constexpr int kMaxCbf = 8;
constexpr int kMaxPrefRows = 16;
constexpr int kNTaskSlack = 6;

// Last stage that strictly tightened the rail velocity bound.  0 means the
// box is still the initial ±v_max (or unset after reset).
enum RailBindStage : uint32_t {
  kRailBindNone = 0,
  kRailBindVMaxDamper = 1,
  kRailBindCmdDamper = 2,
  kRailBindWallCap = 3,
  kRailBindPosBound = 4,
  kRailBindAccel = 5,
  kRailBindJerk = 6,
  kRailBindLead = 7,
  kRailBindPin = 8,
  kRailBindLocked = 9,
  kRailBindBranch = 10,
  kRailBindCollapse = 11,
};

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using VecX = Eigen::VectorXd;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat6x8 = Eigen::Matrix<double, 6, 8>;
using Mat8 = Eigen::Matrix<double, 8, 8>;
using MatX = Eigen::MatrixXd;

inline double clip(double x, double lo, double hi) {
  return std::min(hi, std::max(lo, x));
}

inline double soft_saturate(double x, double lim) {
  if (!(lim > 0.0) || !std::isfinite(x)) return 0.0;
  return lim * std::tanh(x / lim);
}

inline Vec8 inbox_brake(const Vec8& qdot_prev, const Vec8& lo, const Vec8& hi,
                        const Vec8& a_max, double h1) {
  Vec8 out = Vec8::Zero();
  const double h = std::max(h1, 0.0);
  for (int i = 0; i < kNv; ++i) {
    const double step = std::max(a_max[i] * h, 0.0);
    double brake = qdot_prev[i];
    if (qdot_prev[i] > 0.0) brake = qdot_prev[i] - step;
    else if (qdot_prev[i] < 0.0) brake = qdot_prev[i] + step;
    if (std::isfinite(lo[i]) && std::isfinite(hi[i]) && lo[i] <= hi[i]) {
      out[i] = clip(brake, lo[i], hi[i]);
    } else if (std::isfinite(lo[i]) && std::isfinite(hi[i])) {
      out[i] = 0.5 * (lo[i] + hi[i]);
    } else {
      out[i] = brake;
    }
  }
  return out;
}

inline void measure_qdot_box(const Vec8& qdot, const Vec8& lo, const Vec8& hi,
                             double* excess_max, uint32_t* degenerate,
                             uint32_t* infeasible, bool* substantial_out) {
  *excess_max = 0.0;
  *degenerate = 0;
  *infeasible = 0;
  *substantial_out = false;
  for (int i = 0; i < kNv; ++i) {
    if (!(std::isfinite(lo[i]) && std::isfinite(hi[i]))) continue;
    const double w = hi[i] - lo[i];
    if (w < -1.0e-12) {
      *infeasible = 1;
      *degenerate = 1;
    } else if (w <= 1.0e-9) {
      *degenerate = 1;
    }
    double excess = 0.0;
    if (qdot[i] < lo[i]) excess = lo[i] - qdot[i];
    if (qdot[i] > hi[i]) excess = std::max(excess, qdot[i] - hi[i]);
    *excess_max = std::max(*excess_max, excess);
    if (excess > 1.0e-6 && (w <= 1.0e-9 || excess > 0.10 * w)) *substantial_out = true;
  }
}

inline double raised_cosine_alpha(double slack, double slack_exit, double slack_enter,
                                  double sigma, double sigma_ref) {
  const double span = std::max(slack_enter - slack_exit, 1.0e-9);
  double x = clip((slack - slack_exit) / span, 0.0, 1.0);
  const double a_slack = 0.5 * (1.0 + std::cos(M_PI * x));
  const double a_sigma = clip(sigma / std::max(sigma_ref, 1.0e-9), 0.0, 1.0);
  return a_slack * a_sigma;
}

inline double dual_cancel_frac(double u_task, double u_post, double active = 0.002) {
  if (std::abs(u_task) <= active || std::abs(u_post) <= active) return 0.0;
  if (u_task * u_post >= 0.0) return 0.0;
  const double den = std::abs(u_task) + std::abs(u_post);
  if (den <= 1.0e-12) return 0.0;
  return 1.0 - std::abs(u_task + u_post) / den;
}

inline double smoothstep01(double x) {
  x = clip(x, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

inline double lpf_tau_from_fc(double fc) {
  if (fc <= 1.0e-9) return 0.0;
  return 1.0 / (2.0 * M_PI * fc);
}

inline double first_order_lpf(double prev, double target, double dt, double tau) {
  if (tau <= 1.0e-9 || dt <= 0.0) return target;
  const double a = dt / (tau + dt);
  return (1.0 - a) * prev + a * target;
}

inline Vec8 first_order_lpf_vec(const Vec8& prev, const Vec8& target, double dt, double tau) {
  if (tau <= 1.0e-9 || dt <= 0.0) return target;
  const double a = dt / (tau + dt);
  return (1.0 - a) * prev + a * target;
}

inline Vec6 first_order_lpf_vec6(const Vec6& prev, const Vec6& target, double dt, double tau) {
  if (tau <= 1.0e-9 || dt <= 0.0) return target;
  const double a = dt / (tau + dt);
  return (1.0 - a) * prev + a * target;
}

inline double wrap_pi(double a) {
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a < 0.0) a += 2.0 * M_PI;
  return a - M_PI;
}

inline double fold_psi_to_positive(double psi) {
  return std::min(std::abs(wrap_pi(psi)), M_PI);
}

inline double psi_err_avoiding_zero(double cur, double target) {
  cur = wrap_pi(cur);
  target = wrap_pi(target);
  double err = wrap_pi(target - cur);
  const double nxt = cur + err;
  if (cur * nxt < 0.0 && std::abs(cur) < 0.5 * M_PI && std::abs(target) < 0.5 * M_PI) {
    if (err > 0.0) err -= 2.0 * M_PI;
    else err += 2.0 * M_PI;
  }
  return err;
}

inline double clamp_psi_to_envelope(double psi, double lo, double hi) {
  lo = std::max(lo, 1e-6);
  hi = std::min(hi, M_PI - 1e-6);
  if (lo > hi) std::swap(lo, hi);
  return clip(fold_psi_to_positive(psi), lo, hi);
}

inline double integration_period(double dt_nom, double dt_wall) {
  if (!(std::isfinite(dt_wall) && dt_wall > 0.0)) return dt_nom;
  return clip(dt_wall, dt_nom, 1.25 * dt_nom);
}

inline double stopping_velocity(double distance, double accel, double reaction) {
  const double d = std::max(distance, 0.0);
  const double a = std::max(accel, 1.0e-9);
  const double r = std::max(reaction, 0.0);
  return std::sqrt(a * a * r * r + 2.0 * a * d) - a * r;
}

inline void wall_cap(double x, double lo, double hi, double a_max, double reaction,
                    double* lo_cap, double* hi_cap) {
  *lo_cap = -stopping_velocity(x - lo, a_max, reaction);
  *hi_cap = stopping_velocity(hi - x, a_max, reaction);
}

inline double sr_damping_lambda(double sigma_min, double lam0, double sigma_ref,
                                double sigma_floor) {
  const double sigma = std::max(sigma_min, sigma_floor);
  if (sigma >= sigma_ref) return lam0;
  const double r = sigma_ref / sigma;
  return lam0 * r * r;
}

inline double wall_leave_only_sign(double x, double hard_min, double hard_max, double band) {
  band = std::max(band, 0.0);
  if (x >= hard_max - band) return 1.0;
  if (x <= hard_min + band) return -1.0;
  return 0.0;
}

inline void collapse_interval(Vec8* lo, Vec8* hi, const Vec8* qdot_prev, const Vec8* a_max,
                              double dt) {
  for (int i = 0; i < kNv; ++i) {
    if ((*lo)[i] <= (*hi)[i]) continue;
    // Crossed: lo > hi.  The conflicting interval is [hi, lo].
    const double gap_lo = (*hi)[i];
    const double gap_hi = (*lo)[i];
    double target = 0.0;
    if (qdot_prev != nullptr) {
      target = (*qdot_prev)[i];
      if (a_max != nullptr && dt > 0.0) {
        const double step = (*a_max)[i] * dt;
        if (target > 0.0) target = std::max(0.0, target - step);
        else if (target < 0.0) target = std::min(0.0, target + step);
      } else {
        target = 0.0;
      }
    }
    double collapsed = clip(target, gap_lo, gap_hi);
    if (qdot_prev != nullptr && a_max != nullptr && dt > 0.0) {
      const double step = (*a_max)[i] * dt;
      collapsed = clip(collapsed, (*qdot_prev)[i] - step, (*qdot_prev)[i] + step);
    }
    (*lo)[i] = collapsed;
    (*hi)[i] = collapsed;
  }
}

inline double max_limit_activation(const Vec8& q, const Vec8& mid, const Vec8& half,
                                   double activation) {
  double u_max = 0.0;
  const double span = std::max(1.0 - activation, 1e-6);
  for (int i = 0; i < kNv; ++i) {
    const double u = (q[i] - mid[i]) / std::max(half[i], 1e-9);
    const double over = clip((std::abs(u) - activation) / span, 0.0, 1.0);
    u_max = std::max(u_max, over);
  }
  return u_max;
}

inline Vec8 margin_weight_from_activation(const Vec8& q, const Vec8& mid, const Vec8& half,
                                          double k_margin, double activation) {
  Vec8 mw = Vec8::Ones();
  const double span = std::max(1.0 - activation, 1e-6);
  for (int i = 0; i < kNv; ++i) {
    const double u = clip(std::abs(q[i] - mid[i]) / std::max(half[i], 1e-9), 0.0, 1.0);
    const double over = clip((u - activation) / span, 0.0, 1.0);
    mw[i] = 1.0 + k_margin * over * over;
  }
  return mw;
}

inline std::pair<double, Vec8> allocate_rail(const Mat6x8& J, const Vec6& v_d,
                                             const Vec8& qdot_scale, const Vec8& mw,
                                             double lam, double v0, double w0, double e_mid,
                                             double k_err, double e_ref) {
  Vec6 scale;
  scale << v0, v0, v0, w0, w0, w0;
  scale = scale.cwiseMax(1.0e-9);
  const Vec6 v_n = v_d.cwiseQuotient(scale);
  Mat6x8 Jn;
  for (int r = 0; r < 6; ++r) Jn.row(r) = J.row(r) / scale[r];
  Vec8 Winv = qdot_scale.cwiseProduct(qdot_scale).cwiseQuotient(mw.cwiseMax(1.0e-9));
  if (k_err > 0.0) {
    const double gain = 1.0 + k_err * std::min(std::abs(e_mid) / std::max(e_ref, 1e-9), 1.0);
    Winv[0] *= gain * gain;
  }
  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  for (int i = 0; i < kNv; ++i) {
    A += Winv[i] * Jn.col(i) * Jn.col(i).transpose();
  }
  A.diagonal().array() += lam * lam;
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
      A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Vec6 y = svd.solve(v_n);
  const Vec8 JtY = Jn.transpose() * y;
  Vec8 qdot = Winv.cwiseProduct(JtY);
  return {qdot[0], qdot};
}

inline std::pair<Vec6, double> project_arm_compensation(const Mat6x8& J, const Vec6& req,
                                                        const Vec8& q, const Vec8& q_lo,
                                                        const Vec8& q_hi, double activation,
                                                        double alpha) {
  const auto Ja = J.rightCols<7>();
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(
      Ja, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 1> qdot_a = svd.solve(req);
  Eigen::Matrix<double, 7, 1> qdot_p = qdot_a;
  for (int i = 0; i < 7; ++i) {
    const double lo = q_lo[i + 1];
    const double hi = q_hi[i + 1];
    const double half = std::max(0.5 * (hi - lo), 1e-9);
    const double mid = 0.5 * (hi + lo);
    const double u = (q[i + 1] - mid) / half;
    if ((u * qdot_a[i]) > 0.0 && std::abs(u) >= activation) {
      qdot_p[i] *= 1.0 - clip(alpha, 0.0, 1.0);
    }
  }
  const Vec6 cmp = Ja * qdot_p;
  const double nreq = req.norm();
  double frac = (nreq < 1e-12) ? 0.0 : clip(1.0 - cmp.norm() / nreq, 0.0, 1.0);
  return {cmp, frac};
}

inline std::pair<double, double> arm_mirror_rail_limits(const Mat6x8& J, const Vec8& a_arm,
                                                        const Vec8& j_arm, double rho_a,
                                                        double rho_j) {
  const auto Ja = J.rightCols<7>();
  const Vec6 jr = J.col(0);
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd(
      Ja, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, 7, 1> p = svd.solve(jr).cwiseAbs();
  double a_lim = std::numeric_limits<double>::infinity();
  double j_lim = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 7; ++i) {
    if (p[i] <= 1e-6) continue;
    a_lim = std::min(a_lim, rho_a * std::abs(a_arm[i + 1]) / p[i]);
    j_lim = std::min(j_lim, rho_j * std::abs(j_arm[i + 1]) / p[i]);
  }
  if (!std::isfinite(a_lim)) a_lim = std::numeric_limits<double>::infinity();
  if (!std::isfinite(j_lim)) j_lim = std::numeric_limits<double>::infinity();
  return {std::max(a_lim, 0.0), std::max(j_lim, 0.0)};
}

inline Vec8 project_nullspace(const Mat6x8& J, const Vec8& qdot0, double damping) {
  // N = I - J^T (J J^T + λ²I)^{-1} J
  Eigen::Matrix<double, 6, 6> A = J * J.transpose();
  A.diagonal().array() += damping * damping;
  Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(A);
  const Vec6 y = ldlt.solve(J * qdot0);
  return qdot0 - J.transpose() * y;
}

inline bool clamp_command_step(const Vec8& q_prev, const Vec8& q_des, const Vec8* dq_prev,
                               const Vec8& a_max, double dt, Vec8* q_safe, Vec8* dq,
                               bool* clamped) {
  *dq = q_des - q_prev;
  *clamped = false;
  if (dq_prev != nullptr && dt > 0.0) {
    const double dt2 = dt * dt;
    for (int i = 0; i < kNv; ++i) {
      const double ddq_max = a_max[i] * dt2;
      const double dq_new = (*dq_prev)[i] + clip((*dq)[i] - (*dq_prev)[i], -ddq_max, ddq_max);
      if (std::abs(dq_new - (*dq)[i]) > 1e-15) *clamped = true;
      (*dq)[i] = dq_new;
    }
  }
  *q_safe = q_prev + *dq;
  return *clamped;
}

}  // namespace wbc_rt
