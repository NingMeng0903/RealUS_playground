#pragma once

#include <cmath>
#include <limits>
#include "wbc_rt/protocol.hpp"
#include "wbc_rt/types.hpp"

namespace wbc_rt {
// Generic TCP/base velocity inequalities. No force, image, or task policy here.
constexpr double kCartesianTolerance = 1.0e-8;
struct CartesianConstraints {
  bool enabled = false;
  uint32_t count = 0;
  uint64_t sequence = 0;
  uint64_t stop_epoch = 0;
  double valid_until = 0.0;
  Eigen::Matrix<double, kMaxCartesianRows, 6, Eigen::RowMajor> A =
      Eigen::Matrix<double, kMaxCartesianRows, 6, Eigen::RowMajor>::Zero();
  Eigen::Matrix<double, kMaxCartesianRows, 1> lower =
      Eigen::Matrix<double, kMaxCartesianRows, 1>::Zero();
  Eigen::Matrix<double, kMaxCartesianRows, 1> upper =
      Eigen::Matrix<double, kMaxCartesianRows, 1>::Zero();

  bool valid(double now) const {
    if (!enabled) return true;
    if (count > kMaxCartesianRows || !std::isfinite(now) ||
        !std::isfinite(valid_until) || now >= valid_until) return false;
    for (uint32_t i = 0; i < count; ++i) {
      if (!A.row(i).allFinite() || std::isnan(lower[i]) || std::isnan(upper[i]) ||
          lower[i] > upper[i] || lower[i] == std::numeric_limits<double>::infinity() ||
          upper[i] == -std::numeric_limits<double>::infinity()) return false;
    }
    return true;
  }

  double violation(const Vec6& velocity) const {
    if (!enabled) return 0.;
    if (count > kMaxCartesianRows || !velocity.allFinite())
      return std::numeric_limits<double>::infinity();
    double residual = 0.;
    for (uint32_t i = 0; i < count; ++i) {
      const double value = A.row(i).dot(velocity);
      if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
      residual = std::max(residual, std::max(lower[i] - value, value - upper[i]));
    }
    return residual;
  }
};

inline void append_cartesian_rows(const CartesianConstraints& certificate,
                                  const Mat6x8& J, double rail_execution,
                                  MatX* C, VecX* lower, VecX* upper, int offset) {
  if (!certificate.enabled) return;
  Mat6x8 predicted = J;
  predicted.col(0).setZero();
  const Vec6 rail = J.col(0) * rail_execution;
  for (uint32_t i = 0; i < certificate.count; ++i) {
    const auto row = certificate.A.row(i);
    C->block(offset + i, 0, 1, kNv) = row * J;
    (*lower)[offset + i] = certificate.lower[i];
    (*upper)[offset + i] = certificate.upper[i];
    const int other = offset + certificate.count + i;
    C->block(other, 0, 1, kNv) = row * predicted;
    const double fixed = row.dot(rail);
    (*lower)[other] = certificate.lower[i] - fixed;
    (*upper)[other] = certificate.upper[i] - fixed;
  }
}
}  // namespace wbc_rt
