#pragma once

#include <string>
#include <vector>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "wbc_rt/types.hpp"

namespace wbc_rt {

class Kinematics {
 public:
  explicit Kinematics(const std::string& urdf_path);

  void update(const Vec8& q);
  Mat6x8 jacobian() const { return J_; }
  Vec6 singular_values() const { return sigma_; }
  double sigma_min() const { return sigma_.minCoeff(); }
  double sigma_arm() const { return sigma_arm_; }
  Mat8 mass() const { return M_; }
  Eigen::Vector3d tcp_xyz() const { return tcp_xyz_; }
  Eigen::Matrix3d tcp_R() const { return tcp_R_; }
  Vec6 fk_pose_at(const Vec8& q) const;
  Eigen::Vector3d joint_origin(const std::string& name) const;

  Vec8 q_lower() const { return q_lo_; }
  Vec8 q_upper() const { return q_hi_; }
  Vec8 v_max() const { return v_max_; }

  pinocchio::Model& model() { return model_; }
  pinocchio::Data& data() { return data_; }
  pinocchio::FrameIndex tcp_id() const { return tcp_id_; }

  double sew_psi(const Vec8& q);

 private:
  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::Data psi_data_;
  pinocchio::FrameIndex tcp_id_{};
  pinocchio::JointIndex j_shoulder_{};
  pinocchio::JointIndex j_elbow_{};
  pinocchio::JointIndex j_wrist_{};
  Mat6x8 J_ = Mat6x8::Zero();
  Vec6 sigma_ = Vec6::Zero();
  double sigma_arm_ = 0.0;
  Mat8 M_ = Mat8::Identity();
  Eigen::Vector3d tcp_xyz_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d tcp_R_ = Eigen::Matrix3d::Identity();
  Vec8 q_lo_ = Vec8::Zero();
  Vec8 q_hi_ = Vec8::Zero();
  Vec8 v_max_ = Vec8::Ones();
};

}  // namespace wbc_rt
