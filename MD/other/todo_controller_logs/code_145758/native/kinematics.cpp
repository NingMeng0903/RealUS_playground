#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/srs_ik.hpp"

#include <stdexcept>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace wbc_rt {

Kinematics::Kinematics(const std::string& urdf_path)
    : data_(pinocchio::Model()), psi_data_(pinocchio::Model()) {
  pinocchio::urdf::buildModel(urdf_path, model_, false, false);
  if (model_.nv != kNv || model_.nq != kNv) {
    throw std::runtime_error("expected 8-DOF model");
  }
  data_ = pinocchio::Data(model_);
  psi_data_ = pinocchio::Data(model_);
  if (!model_.existFrame("tcp")) {
    throw std::runtime_error("URDF missing tcp frame");
  }
  tcp_id_ = model_.getFrameId("tcp");
  j_shoulder_ = model_.getJointId("joint_2");
  j_elbow_ = model_.getJointId("joint_4");
  j_wrist_ = model_.getJointId("joint_6");
  q_lo_ = model_.lowerPositionLimit;
  q_hi_ = model_.upperPositionLimit;
  v_max_ = model_.velocityLimit;
}

void Kinematics::update(const Vec8& q) {
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);
  pinocchio::computeJointJacobians(model_, data_, q);
  pinocchio::Data::Matrix6x J6(6, model_.nv);
  pinocchio::getFrameJacobian(model_, data_, tcp_id_, pinocchio::LOCAL_WORLD_ALIGNED, J6);
  J_ = J6;
  Eigen::JacobiSVD<Mat6x8> svd(J_, Eigen::ComputeThinU | Eigen::ComputeThinV);
  sigma_ = svd.singularValues();
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 7>> svd_a(
      J_.rightCols<7>(), Eigen::ComputeThinU | Eigen::ComputeThinV);
  sigma_arm_ = svd_a.singularValues().minCoeff();
  pinocchio::crba(model_, data_, q);
  data_.M.triangularView<Eigen::StrictlyLower>() =
      data_.M.transpose().triangularView<Eigen::StrictlyLower>();
  M_ = data_.M;
  tcp_xyz_ = data_.oMf[tcp_id_].translation();
  tcp_R_ = data_.oMf[tcp_id_].rotation();
}

Vec6 Kinematics::fk_pose_at(const Vec8& q) const {
  pinocchio::Data data(model_);
  pinocchio::forwardKinematics(model_, data, q);
  pinocchio::updateFramePlacement(model_, data, tcp_id_);
  Vec6 pose = srs::rpy_xyz_from_R(data.oMf[tcp_id_].rotation());
  pose.head<3>() = data.oMf[tcp_id_].translation();
  return pose;
}

Eigen::Vector3d Kinematics::joint_origin(const std::string& name) const {
  const auto jid = model_.getJointId(name);
  return data_.oMi[jid].translation();
}

double Kinematics::sew_psi(const Vec8& q) {
  pinocchio::forwardKinematics(model_, psi_data_, q);
  const Eigen::Vector3d S = psi_data_.oMi[j_shoulder_].translation();
  const Eigen::Vector3d E = psi_data_.oMi[j_elbow_].translation();
  const Eigen::Vector3d W = psi_data_.oMi[j_wrist_].translation();
  Eigen::Vector3d sw = W - S;
  const double n = sw.norm();
  if (n < 1e-9) return 0.0;
  const Eigen::Vector3d w_hat = sw / n;
  const Eigen::Vector3d se = E - S;
  const Eigen::Vector3d ne = se - w_hat * w_hat.dot(se);
  const Eigen::Vector3d vref(0.0, 0.0, -1.0);
  Eigen::Vector3d x = vref - w_hat * w_hat.dot(vref);
  if (x.norm() < 1e-9) return 0.0;
  x.normalize();
  const Eigen::Vector3d y = w_hat.cross(x);
  return std::atan2(y.dot(ne), x.dot(ne));
}

}  // namespace wbc_rt
