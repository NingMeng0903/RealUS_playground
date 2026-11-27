#pragma once

#include <optional>

#include <Eigen/Dense>

#include "wbc_rt/types.hpp"

namespace wbc_rt {
namespace srs {

constexpr double kDbs = 0.2405;
constexpr double kDse = 0.256;
constexpr double kDew = 0.210;
constexpr double kDwtFlange = 0.1612;
constexpr double kDwt = 0.3812;
constexpr double kRailOriginY = -0.4;
constexpr double kEpsSin = 1e-6;

inline const Eigen::Matrix<double, 7, 1>& q_lo() {
  static const Eigen::Matrix<double, 7, 1> v =
      (Eigen::Matrix<double, 7, 1>() << -3.106, -2.2689, -3.106, -2.356, -3.106, -2.234, -6.28)
          .finished();
  return v;
}
inline const Eigen::Matrix<double, 7, 1>& q_hi() {
  static const Eigen::Matrix<double, 7, 1> v =
      (Eigen::Matrix<double, 7, 1>() << 3.106, 2.2689, 3.106, 2.356, 3.106, 2.234, 6.28)
          .finished();
  return v;
}

inline double shoulder_y_from_q_rail(double q_rail) { return kRailOriginY + q_rail; }

Eigen::Matrix3d rot_z(double a);
Eigen::Matrix3d rot_y(double a);
Eigen::Matrix3d rot_x(double a);
Eigen::Matrix3d euler_xyz_to_R(double rx, double ry, double rz);
Vec6 rpy_xyz_from_R(const Eigen::Matrix3d& R);

int branch_from_q(const Vec8& q);
double psi_from_q(const Vec8& q);

// Closed-form Shimizu IK.  pose = [x,y,z,rx,ry,rz] (extrinsic xyz).
// y_rail is world shoulder Y (RAIL_ORIGIN_Y + q0).  Returns 7-vec or nullopt.
std::optional<Eigen::Matrix<double, 7, 1>> srs_ik(
    const Vec6& pose_tcp, double psi, int branch_id, double y_shoulder,
    const Eigen::Matrix3d* R_flange_tcp, const Eigen::Vector3d* t_flange_tcp);

}  // namespace srs
}  // namespace wbc_rt
