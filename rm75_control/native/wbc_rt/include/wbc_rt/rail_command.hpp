#pragma once

#include <algorithm>
#include <cmath>

#include "wbc_rt/types.hpp"

namespace wbc_rt {

constexpr double kEscapeEnter = 2.0e-4;
constexpr double kEscapeExit = 1.0e-4;

inline double project_interval(double x, double lo, double hi) {
  if (lo > hi) return 0.5 * (lo + hi);
  return clip(x, lo, hi);
}

inline int j4_index(int nv) {
  if (nv >= 8) return 4;
  if (nv == 7) return 3;
  return std::min(4, std::max(0, nv - 1));
}

inline int update_escape_dir(bool explicit_active, double u_escape_raw, int prev_dir,
                             double eps_enter = kEscapeEnter, double eps_exit = kEscapeExit) {
  if (!explicit_active) return 0;
  const double mag = std::abs(u_escape_raw);
  if (prev_dir == 0) {
    if (mag > eps_enter) return (u_escape_raw > 0.0) ? 1 : -1;
    return 0;
  }
  if (mag < eps_exit) return 0;
  return prev_dir;
}

inline void wall_velocity_bounds(double u_max, double leave_sign, double* lo, double* hi) {
  const double cap = std::abs(u_max);
  *lo = -cap;
  *hi = cap;
  if (leave_sign > 0.0) *hi = std::min(*hi, 0.0);
  else if (leave_sign < 0.0) *lo = std::max(*lo, 0.0);
}

struct RailShares {
  double u_escape_feasible = 0.0;
  double u_task_feasible = 0.0;
  double u_base = 0.0;
  double u_post_feasible = 0.0;
  double u_feasible = 0.0;
};

inline RailShares allocate_rail_shares(double u_task_raw, double u_post_raw, double u_escape_raw,
                                       int escape_dir, double u_lo, double u_hi) {
  RailShares s;
  s.u_escape_feasible = project_interval(u_escape_raw, u_lo, u_hi);
  double t_lo = u_lo;
  double t_hi = u_hi;
  if (escape_dir > 0) {
    t_lo = 0.0;
    t_hi = u_hi - s.u_escape_feasible;
  } else if (escape_dir < 0) {
    t_lo = u_lo - s.u_escape_feasible;
    t_hi = 0.0;
  }
  s.u_task_feasible = project_interval(u_task_raw, t_lo, t_hi);
  s.u_base = s.u_escape_feasible + s.u_task_feasible;
  double p_lo = u_lo - s.u_base;
  double p_hi = u_hi - s.u_base;
  if (escape_dir > 0) {
    p_lo = s.u_escape_feasible - s.u_base;
    p_hi = u_hi - s.u_base;
  } else if (escape_dir < 0) {
    p_lo = u_lo - s.u_base;
    p_hi = s.u_escape_feasible - s.u_base;
  }
  s.u_post_feasible = project_interval(u_post_raw, p_lo, p_hi);
  s.u_feasible = s.u_base + s.u_post_feasible;
  return s;
}

inline double project_lpf_into_wall(double v, double leave_sign) {
  if (leave_sign > 0.0 && v > 0.0) return 0.0;
  if (leave_sign < 0.0 && v < 0.0) return 0.0;
  return v;
}

inline bool q_finite_in_limits(const Vec8& q, const Vec8& lo, const Vec8& hi) {
  for (int i = 0; i < kNv; ++i) {
    if (!std::isfinite(q[i])) return false;
    if (q[i] < lo[i] - 1e-9 || q[i] > hi[i] + 1e-9) return false;
  }
  return true;
}

}  // namespace wbc_rt
