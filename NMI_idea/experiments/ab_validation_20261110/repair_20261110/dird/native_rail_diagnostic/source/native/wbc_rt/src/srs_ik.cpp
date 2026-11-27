#include "wbc_rt/srs_ik.hpp"

#include <cmath>

namespace wbc_rt {
namespace srs {
namespace {

Eigen::Vector3d v_ref() { return Eigen::Vector3d(0.0, 0.0, -1.0); }

double psi_from_sew(const Eigen::Vector3d& S, const Eigen::Vector3d& E,
                    const Eigen::Vector3d& W) {
  const Eigen::Vector3d sw = W - S;
  const double n_sw = sw.norm();
  if (n_sw < 1e-9) return 0.0;
  const Eigen::Vector3d w_hat = sw / n_sw;
  const Eigen::Vector3d se = E - S;
  const Eigen::Vector3d e_perp = se - w_hat * w_hat.dot(se);
  const Eigen::Vector3d r_perp = v_ref() - w_hat * w_hat.dot(v_ref());
  if (e_perp.norm() < 1e-9 || r_perp.norm() < 1e-9) return 0.0;
  const Eigen::Vector3d e_u = e_perp.normalized();
  const Eigen::Vector3d r_u = r_perp.normalized();
  return std::atan2(r_u.cross(e_u).dot(w_hat), r_u.dot(e_u));
}

std::optional<Eigen::Vector3d> E_from_psi(const Eigen::Vector3d& S, const Eigen::Vector3d& W,
                                          double psi) {
  const Eigen::Vector3d sw = W - S;
  const double dsw = sw.norm();
  if (dsw < std::abs(kDse - kDew) + 1e-9 || dsw > kDse + kDew - 1e-9) return std::nullopt;
  const Eigen::Vector3d w_hat = sw / dsw;
  const double cos_th = clip((kDse * kDse + dsw * dsw - kDew * kDew) / (2.0 * kDse * dsw), -1.0, 1.0);
  const double sin_th = std::sqrt(std::max(0.0, 1.0 - cos_th * cos_th));
  const Eigen::Vector3d E_c = S + (kDse * cos_th) * w_hat;
  const Eigen::Vector3d r_perp = v_ref() - w_hat * w_hat.dot(v_ref());
  if (r_perp.norm() < 1e-9) return std::nullopt;
  const Eigen::Vector3d r_u = r_perp.normalized();
  const Eigen::Vector3d r_bin = w_hat.cross(r_u);
  return E_c + (kDse * sin_th) * (std::cos(psi) * r_u + std::sin(psi) * r_bin);
}

void fk_sew_arm(const Eigen::Matrix<double, 7, 1>& q, Eigen::Vector3d* S, Eigen::Vector3d* E,
                Eigen::Vector3d* W) {
  const double q1 = q[0], q2 = q[1], q3 = q[2], q4 = q[3];
  *S = Eigen::Vector3d(0.0, 0.0, kDbs);
  const Eigen::Vector3d SE_dir(std::cos(q1) * std::sin(q2), std::sin(q1) * std::sin(q2),
                               std::cos(q2));
  *E = *S + kDse * SE_dir;
  const Eigen::Matrix3d R_sh = rot_z(q1) * rot_y(q2) * rot_z(q3);
  const Eigen::Vector3d EW_local(std::sin(q4), 0.0, std::cos(q4));
  *W = *E + kDew * (R_sh * EW_local);
}

}  // namespace

Eigen::Matrix3d rot_z(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}
Eigen::Matrix3d rot_y(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, 0, s, 0, 1, 0, -s, 0, c;
  return R;
}
Eigen::Matrix3d rot_x(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << 1, 0, 0, 0, c, -s, 0, s, c;
  return R;
}
Eigen::Matrix3d euler_xyz_to_R(double rx, double ry, double rz) {
  return rot_z(rz) * rot_y(ry) * rot_x(rx);
}

Vec6 rpy_xyz_from_R(const Eigen::Matrix3d& R) {
  Vec6 p = Vec6::Zero();
  const double ry = std::atan2(-R(2, 0), std::hypot(R(2, 1), R(2, 2)));
  const double rx = std::atan2(R(2, 1), R(2, 2));
  const double rz = std::atan2(R(1, 0), R(0, 0));
  p[3] = rx;
  p[4] = ry;
  p[5] = rz;
  return p;
}

int branch_from_q(const Vec8& q) {
  const int b_sh = q[2] >= 0.0 ? 0 : 1;
  const int b_el = q[4] >= 0.0 ? 0 : 1;
  const int b_wr = q[6] >= 0.0 ? 0 : 1;
  return (b_sh << 2) | (b_el << 1) | b_wr;
}

double psi_from_q(const Vec8& q) {
  Eigen::Matrix<double, 7, 1> arm = q.tail<7>();
  Eigen::Vector3d S, E, W;
  fk_sew_arm(arm, &S, &E, &W);
  return psi_from_sew(S, E, W);
}

std::optional<Eigen::Matrix<double, 7, 1>> srs_ik(
    const Vec6& pose_tcp, double psi, int branch_id, double y_shoulder,
    const Eigen::Matrix3d* R_flange_tcp, const Eigen::Vector3d* t_flange_tcp) {
  const Eigen::Vector3d p_tcp = pose_tcp.head<3>();
  const Eigen::Matrix3d R_tcp = euler_xyz_to_R(pose_tcp[3], pose_tcp[4], pose_tcp[5]);
  Eigen::Matrix3d R_flange = R_tcp;
  Eigen::Vector3d W;
  if (R_flange_tcp != nullptr && t_flange_tcp != nullptr) {
    R_flange = R_tcp * R_flange_tcp->transpose();
    const Eigen::Vector3d p_flange = p_tcp - R_flange * (*t_flange_tcp);
    W = p_flange - kDwtFlange * R_flange.col(2);
  } else {
    W = p_tcp - kDwt * R_tcp.col(2);
  }
  branch_id &= 0x7;
  const int b_sh = (branch_id >> 2) & 1;
  const int b_el = (branch_id >> 1) & 1;
  const int b_wr = branch_id & 1;
  const Eigen::Vector3d S(0.0, y_shoulder, kDbs);
  const double dsw = (W - S).norm();
  if (dsw <= std::abs(kDse - kDew) + 1e-6 || dsw >= kDse + kDew - 1e-6) return std::nullopt;
  const double cos_q4 =
      clip((dsw * dsw - kDse * kDse - kDew * kDew) / (2.0 * kDse * kDew), -1.0, 1.0);
  const double q4_mag = std::acos(cos_q4);
  const double q4 = (b_el == 0) ? q4_mag : -q4_mag;
  const auto E = E_from_psi(S, W, psi);
  if (!E) return std::nullopt;
  const Eigen::Vector3d SE_dir = (*E - S) / kDse;
  const double z = clip(SE_dir[2], -1.0, 1.0);
  const double q2_mag = std::acos(z);
  if (q2_mag < kEpsSin || q2_mag > M_PI - kEpsSin) return std::nullopt;
  double q1, q2;
  if (b_sh == 0) {
    q2 = q2_mag;
    q1 = std::atan2(SE_dir[1], SE_dir[0]);
  } else {
    q2 = -q2_mag;
    q1 = wrap_pi(std::atan2(SE_dir[1], SE_dir[0]) + M_PI);
  }
  const Eigen::Vector3d U = rot_y(-q2) * rot_z(-q1) * (W - S);
  if (U[0] * U[0] + U[1] * U[1] < 1e-16) return std::nullopt;
  const double sign_s4 = (std::sin(q4) >= 0.0) ? 1.0 : -1.0;
  const double q3 = std::atan2(sign_s4 * U[1], sign_s4 * U[0]);
  const Eigen::Matrix3d R_pre = rot_z(q1) * rot_y(q2) * rot_z(q3) * rot_y(q4);
  const Eigen::Matrix3d R_w = R_pre.transpose() * R_flange;
  const double cos_q6 = clip(R_w(2, 2), -1.0, 1.0);
  const double q6_mag = std::acos(cos_q6);
  if (q6_mag < kEpsSin || q6_mag > M_PI - kEpsSin) return std::nullopt;
  double q5, q6, q7;
  if (b_wr == 0) {
    q6 = q6_mag;
    q5 = std::atan2(R_w(1, 2), R_w(0, 2));
    q7 = std::atan2(R_w(2, 1), -R_w(2, 0));
  } else {
    q6 = -q6_mag;
    q5 = std::atan2(-R_w(1, 2), -R_w(0, 2));
    q7 = std::atan2(-R_w(2, 1), R_w(2, 0));
  }
  Eigen::Matrix<double, 7, 1> q_arm;
  q_arm << q1, q2, q3, q4, q5, q6, q7;
  for (int i = 0; i < 7; ++i) {
    if (q_arm[i] < q_lo()[i] - 1e-9 || q_arm[i] > q_hi()[i] + 1e-9) return std::nullopt;
  }
  return q_arm;
}

}  // namespace srs
}  // namespace wbc_rt
