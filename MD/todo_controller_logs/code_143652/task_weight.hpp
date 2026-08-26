#pragma once

#include <algorithm>
#include <Eigen/SVD>

#include "wbc_rt/types.hpp"

namespace wbc_rt {

struct TaskWeightState {
  Mat6 U_prev = Mat6::Identity();
  Vec6 s_lpf = Vec6::Ones();
  double task_scale_lpf = 1.0;
  bool init = false;

  void reset() {
    U_prev.setIdentity();
    s_lpf.setOnes();
    task_scale_lpf = 1.0;
    init = false;
  }

  Mat6 step(const Mat6x8& J_task, const Vec6& w, double dt, double tau, double sigma_ref,
            double min_frac, bool aniso) {
    if (!aniso) {
      const Eigen::JacobiSVD<Mat6x8> svd(J_task, Eigen::ComputeThinU);
      const double sigma_min = svd.singularValues().minCoeff();
      double raw = 1.0;
      if (sigma_ref > 1.0e-9 && sigma_min < sigma_ref) {
        const double frac = sigma_min / sigma_ref;
        raw = std::max(frac * frac, min_frac);
      }
      if (!init) {
        task_scale_lpf = raw;
        init = true;
      } else {
        task_scale_lpf = first_order_lpf(task_scale_lpf, raw, dt, tau);
      }
      return (w * task_scale_lpf).asDiagonal();
    }

    Vec6 w_sqrt = w.cwiseMax(1.0e-12).cwiseSqrt();
    Mat6x8 Jw = w_sqrt.asDiagonal() * J_task;
    Eigen::JacobiSVD<Mat6x8> svd(Jw, Eigen::ComputeThinU);
    Mat6 U = svd.matrixU();
    if (U.cols() < 6) {
      Mat6 U_full = Mat6::Identity();
      U_full.leftCols(U.cols()) = U;
      U = U_full;
    }
    if (init) {
      for (int i = 0; i < 6; ++i) {
        if (U.col(i).dot(U_prev.col(i)) < 0.0) U.col(i) *= -1.0;
      }
    }
    U_prev = U;
    Vec6 s_j = Vec6::Zero();
    const auto sv = svd.singularValues();
    for (int i = 0; i < sv.size() && i < 6; ++i) s_j[i] = sv[i];
    Vec6 s_raw = Vec6::Ones();
    for (int i = 0; i < 6; ++i) {
      if (sigma_ref > 1.0e-9 && s_j[i] < sigma_ref) {
        const double frac = s_j[i] / sigma_ref;
        s_raw[i] = std::max(frac * frac, min_frac);
      }
    }
    if (!init) {
      s_lpf = s_raw;
      init = true;
    } else {
      s_lpf = first_order_lpf_vec6(s_lpf, s_raw, dt, tau);
    }
    const Mat6 usu = U * s_lpf.asDiagonal() * U.transpose();
    return w_sqrt.asDiagonal() * usu * w_sqrt.asDiagonal();
  }
};

}  // namespace wbc_rt
