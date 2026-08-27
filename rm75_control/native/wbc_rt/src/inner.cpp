#include "wbc_rt/inner.hpp"
#include "wbc_rt/rail_command.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace wbc_rt {
namespace {

constexpr int kNVar = kNv + kNTaskSlack + kNPref;
constexpr int kNIn = kNv + kMaxCbf + kMaxPrefRows + kNPref;
constexpr double kRailDriveCap = 0.12;
constexpr double kRailPrefW = 64.0;
constexpr double kQuietLinEnter = 0.005;
constexpr double kQuietRotEnter = 0.05;
constexpr double kQuietLinExit = 0.008;
constexpr double kQuietRotExit = 0.08;
constexpr double kQuietTcp = 0.010;
constexpr double kQuietHold = 0.15;

uint32_t qp_status_code(proxsuite::proxqp::QPSolverOutput status) {
  using S = proxsuite::proxqp::QPSolverOutput;
  switch (status) {
    case S::PROXQP_SOLVED: return kQpSolved;
    case S::PROXQP_MAX_ITER_REACHED: return kQpMaxIter;
    case S::PROXQP_PRIMAL_INFEASIBLE: return kQpPrimalInfeasible;
    case S::PROXQP_DUAL_INFEASIBLE: return kQpDualInfeasible;
    case S::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: return kQpClosestPrimalFeasible;
    case S::PROXQP_NOT_RUN: return kQpNotRun;
  }
  return kQpException;
}

double inequality_violation(const MatX& C, const VecX& x, const VecX& lo, const VecX& hi) {
  if (C.cols() != x.size() || C.rows() != lo.size() || lo.size() != hi.size() ||
      !C.allFinite() || !x.allFinite() || !lo.allFinite() || !hi.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  const VecX value = C * x;
  if (!value.allFinite()) return std::numeric_limits<double>::infinity();
  double violation = 0.0;
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    violation = std::max(violation, lo[i] - value[i]);
    violation = std::max(violation, value[i] - hi[i]);
  }
  return std::max(violation, 0.0);
}

double equality_violation(const MatX& A, const VecX& x, const VecX& b) {
  if (A.cols() != x.size() || A.rows() != b.size() || !A.allFinite() || !x.allFinite() ||
      !b.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  const VecX error = A * x - b;
  return error.allFinite() ? error.lpNorm<Eigen::Infinity>()
                           : std::numeric_limits<double>::infinity();
}

double box_violation(const Vec8& qdot, const Vec8& lo, const Vec8& hi) {
  if (!qdot.allFinite() || !lo.allFinite() || !hi.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }
  double violation = 0.0;
  for (int i = 0; i < kNv; ++i) {
    violation = std::max(violation, lo[i] - qdot[i]);
    violation = std::max(violation, qdot[i] - hi[i]);
  }
  return std::max(violation, 0.0);
}

bool finite_tick_input(const TickIn& in, uint32_t flags) {
  if (!in.q_meas.allFinite() || !in.v_cmd.allFinite()) return false;
  if (!(std::isfinite(in.dt_nom) && std::isfinite(in.dt_wall) &&
        std::isfinite(in.t_mono))) return false;
  if ((flags & kInHasQdotFf) && !in.qdot_ff.allFinite()) return false;
  if ((flags & kInHasPoseD) && !in.pose_d.allFinite()) return false;
  if ((flags & kInHasVelFf) && !in.vel_ff.allFinite()) return false;
  if ((flags & kInHasPathTwist) && !in.path_twist.allFinite()) return false;
  if ((flags & kInHasFeedbackTwist) && !in.feedback_twist.allFinite()) return false;
  if ((flags & kInHasRailV) && !std::isfinite(in.rail_v)) return false;
  if ((flags & kInHasVForce) && !std::isfinite(in.v_force_z)) return false;
  return true;
}

void solve_dense_qp(proxsuite::proxqp::dense::QP<double>& qp,
                    bool* inited,
                    bool last_ok,
                    const MatX& H,
                    const VecX& g,
                    const MatX& A,
                    const VecX& b,
                    const MatX& C,
                    const VecX& lo,
                    const VecX& hi) {
  using IG = proxsuite::proxqp::InitialGuessStatus;
  if (!*inited) {
    qp.init(H, g, A, b, C, lo, hi, true);
    *inited = true;
  } else {
    qp.settings.initial_guess =
        last_ok ? IG::WARM_START_WITH_PREVIOUS_RESULT : IG::NO_INITIAL_GUESS;
    qp.update(H, g, A, b, C, lo, hi, true);
  }
  qp.solve();
}

Vec6 twist_to_base(const Vec6& twist, const Eigen::Matrix3d& R, bool tool) {
  if (!tool) return twist;
  Vec6 out;
  out.head<3>() = R * twist.head<3>();
  out.tail<3>() = R * twist.tail<3>();
  return out;
}

}  // namespace

Collision::Collision(pinocchio::Model& model, const Config& cfg)
    : model_(&model), geom_data_(pinocchio::GeometryModel()), cfg_(cfg) {
  if (!cfg.collision_enabled) return;
  if (cfg.collision_urdf.empty()) {
    throw std::runtime_error("collision enabled without collision URDF");
  }
  std::vector<std::string> dirs;
  const auto slash = cfg.collision_urdf.find_last_of('/');
  if (slash != std::string::npos) dirs.push_back(cfg.collision_urdf.substr(0, slash));
  pinocchio::urdf::buildGeom(model, cfg.collision_urdf, pinocchio::COLLISION, geom_model_,
                             dirs, ::coal::MeshLoaderPtr());
  geom_model_.addAllCollisionPairs();
  if (!cfg.pair_config.empty()) {
    std::ifstream in(cfg.pair_config);
    if (!in) throw std::runtime_error("cannot open collision pair config: " + cfg.pair_config);
    std::string line;
    while (std::getline(in, line)) {
      auto a0 = line.find('[');
      auto a1 = line.find(',');
      auto a2 = line.find(']');
      if (a0 == std::string::npos || a1 == std::string::npos || a2 == std::string::npos) continue;
      std::string na = line.substr(a0 + 1, a1 - a0 - 1);
      std::string nb = line.substr(a1 + 1, a2 - a1 - 1);
      auto trim = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '"')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '"')) s.pop_back();
        return s;
      };
      na = trim(na);
      nb = trim(nb);
      int ia = -1, ib = -1;
      for (std::size_t i = 0; i < geom_model_.geometryObjects.size(); ++i) {
        if (geom_model_.geometryObjects[i].name == na) ia = static_cast<int>(i);
        if (geom_model_.geometryObjects[i].name == nb) ib = static_cast<int>(i);
      }
      if (ia >= 0 && ib >= 0) {
        pinocchio::CollisionPair cp(static_cast<pinocchio::GeomIndex>(ia),
                                    static_cast<pinocchio::GeomIndex>(ib));
        if (geom_model_.existCollisionPair(cp)) geom_model_.removeCollisionPair(cp);
      }
    }
  }
  if (geom_model_.ngeoms == 0 || geom_model_.collisionPairs.empty()) {
    throw std::runtime_error("collision model has no usable geometry pairs");
  }
  geom_data_ = pinocchio::GeometryData(geom_model_);
  slots_.assign(static_cast<std::size_t>(cfg.max_pairs), -1);
}

void Collision::update(const Vec8& q, pinocchio::Data& data) {
  if (geom_model_.ngeoms == 0) return;
  pinocchio::updateGeometryPlacements(*model_, data, geom_model_, geom_data_, q);
  pinocchio::computeDistances(geom_model_, geom_data_);
}

int Collision::build_rows(pinocchio::Data& data, MatX* jac, VecX* lower, std::vector<int>* slots) {
  jac->resize(0, kNv);
  lower->resize(0);
  slots->clear();
  if (geom_model_.ngeoms == 0) return 0;
  auto skew = [](const Eigen::Vector3d& r) {
    Eigen::Matrix3d s;
    s << 0, -r.z(), r.y(), r.z(), 0, -r.x(), -r.y(), r.x(), 0;
    return s;
  };
  struct Hit {
    double d;
    Eigen::Vector3d n, pa, pb;
    int ga, gb;
  };
  std::vector<Hit> hits;
  for (std::size_t i = 0; i < geom_model_.collisionPairs.size(); ++i) {
    const auto& res = geom_data_.distanceResults[i];
    const double d = res.min_distance;
    if (d > cfg_.d_activate + 0.01) continue;
    Hit h;
    h.d = d;
    h.pa = Eigen::Vector3d(res.nearest_points[0][0], res.nearest_points[0][1],
                           res.nearest_points[0][2]);
    h.pb = Eigen::Vector3d(res.nearest_points[1][0], res.nearest_points[1][1],
                           res.nearest_points[1][2]);
    // Same witness normal as Python: (p_a - p_b).  coal's res.normal can flip.
    const Eigen::Vector3d n_ab = h.pa - h.pb;
    const double n_norm = n_ab.norm();
    h.n = (n_norm > 1e-9) ? (n_ab / n_norm) : Eigen::Vector3d(0.0, 0.0, 1.0);
    h.ga = static_cast<int>(geom_model_.collisionPairs[i].first);
    h.gb = static_cast<int>(geom_model_.collisionPairs[i].second);
    hits.push_back(h);
  }
  std::stable_sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
    if (a.d != b.d) return a.d < b.d;
    if (a.ga != b.ga) return a.ga < b.ga;
    return a.gb < b.gb;
  });
  const int n = std::min(static_cast<int>(hits.size()), cfg_.max_pairs);
  if (n == 0) return 0;
  jac->resize(n, kNv);
  lower->resize(n);
  jac->setZero();
  for (int i = 0; i < n; ++i) {
    const auto& h = hits[static_cast<std::size_t>(i)];
    const auto& goa = geom_model_.geometryObjects[static_cast<std::size_t>(h.ga)];
    const auto& gob = geom_model_.geometryObjects[static_cast<std::size_t>(h.gb)];
    pinocchio::Data::Matrix6x Ja(6, model_->nv), Jb(6, model_->nv);
    pinocchio::getFrameJacobian(*model_, data, goa.parentFrame, pinocchio::LOCAL_WORLD_ALIGNED, Ja);
    pinocchio::getFrameJacobian(*model_, data, gob.parentFrame, pinocchio::LOCAL_WORLD_ALIGNED, Jb);
    const Eigen::Vector3d oa = data.oMf[goa.parentFrame].translation();
    const Eigen::Vector3d ob = data.oMf[gob.parentFrame].translation();
    const Eigen::Matrix<double, 3, 8> Ja_lin =
        Ja.topRows<3>() - skew(h.pa - oa) * Ja.bottomRows<3>();
    const Eigen::Matrix<double, 3, 8> Jb_lin =
        Jb.topRows<3>() - skew(h.pb - ob) * Jb.bottomRows<3>();
    jac->row(i) = h.n.transpose() * (Ja_lin - Jb_lin);
    (*lower)[i] = -cfg_.cbf_gamma * (h.d - cfg_.d_safe);
    slots->push_back(i);
  }
  return n;
}

InnerLoop::InnerLoop(const Config& cfg)
    : cfg_(cfg), kin_(cfg.urdf, cfg.tcp_placement_R(), cfg.tcp_placement_t()),
      posture_(cfg) {
  q_lo_ = kin_.q_lower();
  q_hi_ = kin_.q_upper();
  v_max_ = kin_.v_max() * cfg.v_scale;
  v_max_[0] = std::min({v_max_[0], cfg.rail_v_max, kRailDriveCap});
  q_lo_[0] = std::max(q_lo_[0], cfg.hard_min);
  q_hi_[0] = std::min(q_hi_[0], cfg.hard_max);
  a_max_.setConstant(cfg.a_max_arm);
  a_max_[0] = cfg.a_max_rail;
  j_max_.setConstant(cfg.j_max_arm);
  j_max_[0] = cfg.j_max_rail;
  q_mid_ = 0.5 * (q_lo_ + q_hi_);
  half_ = (0.5 * (q_hi_ - q_lo_)).cwiseMax(1e-9);
  q_nominal_ = cfg.q_nominal;
  if (q_nominal_.norm() < 1e-12) q_nominal_ = q_mid_;
  q_star_ = q_nominal_;
  q_star_signs_ = q_nominal_;
  rail_mode_ = cfg.rail_mode;
  locked_style_ = cfg.locked_style;
  if (cfg.collision_enabled) {
    collision_ = std::make_unique<Collision>(kin_.model(), cfg);
  }
  qp1_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(kNVar, kNTaskSlack, kNIn);
  qp2_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(kNVar, kNTaskSlack, kNIn);
  qp1_->settings.eps_abs = cfg.eps_abs;
  qp2_->settings.eps_abs = cfg.eps_abs;
  qp1_->settings.max_iter = std::min(cfg.max_iter, cfg.max_iter_cap);
  qp2_->settings.max_iter = std::min(cfg.max_iter, cfg.max_iter_cap);
  qp1_->settings.verbose = false;
  qp2_->settings.verbose = false;
  qp1_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
}

void InnerLoop::enable() { enabled_ = true; }

void InnerLoop::clear_rail_box_tel() {
  rail_box_lo_ = 0.0;
  rail_box_hi_ = 0.0;
  rail_bind_lo_ = kRailBindNone;
  rail_bind_hi_ = kRailBindNone;
  rail_task_vel_used_ = 0.0;
  rail_h1_ = 0.0;
  rail_h2_ = 0.0;
  rail_qdot_prev_tel_ = 0.0;
  rail_qdot_prev2_tel_ = 0.0;
  qp1_status_ = kQpNotRun;
  qp2_status_ = kQpNotRun;
  fallback_level_ = kFallbackNone;
  failure_code_ = kFailureNone;
  qp1_hard_violation_ = 0.0;
  final_hard_violation_ = 0.0;
  task_lock_violation_ = 0.0;
  final_box_violation_ = 0.0;
  qp_overrun_ = false;
}

void InnerLoop::note_rail_bind(double old_lo, double old_hi, const Vec8& lo, const Vec8& hi,
                               uint32_t stage) {
  constexpr double kEps = 1.0e-12;
  if (lo[0] > old_lo + kEps) rail_bind_lo_ = stage;
  if (hi[0] < old_hi - kEps) rail_bind_hi_ = stage;
}

void InnerLoop::stop() {
  enabled_ = false;
  quiescent_ = true;
  quiet_s_ = kQuietHold;
  cmd_quiet_s_ = kQuietHold;
  v_r_ref_ = 0.0;
  v_r_a_ = 0.0;
  v_r_init_ = false;
  mid_integ_ = 0.0;
  slack_hold_latched_ = false;
  sec_qdot_.setZero();
  sec_acc_.setZero();
  sec_target_.setZero();
  sec_lpf_.setZero();
  gN_lpf_.setZero();
  gN_lpf_init_ = false;
  posture_gate_active_ = false;
  posture_gate_scale_ = 0.0;
  posture_gate_enter_s_ = posture_gate_exit_s_ = 0.0;
  clear_rail_box_tel();
}

void InnerLoop::reset(const Vec8& q0) {
  q_cmd_ = q0;
  qdot_prev_.setZero();
  qdot_seen_.setZero();
  qdot_prev2_.setZero();
  dq_prev_.setZero();
  have_dq_prev_ = false;
  v_r_ref_ = 0.0;
  v_r_a_ = 0.0;
  v_r_lpf_ = 0.0;
  v_r_init_ = false;
  wall_pi_frozen_ = false;
  u_alloc_ = u_mid_ = u_mid_committed_ = mid_integ_ = 0.0;
  u_task_raw_ = u_task_feasible_ = u_pi_raw_ = u_mid_cmd_ = 0.0;
  u_post_raw_ = u_post_feasible_ = u_mid_applied_ = d_star_dot_cmd_ = 0.0;
  u_escape_raw_ = u_escape_feasible_ = u_base_ = u_feasible_ = 0.0;
  e_d_ = V_d_proxy_ = j4_design_slack_ = sigma_slack_ = 0.0;
  d_star_ref_ = 0.0;
  d_star_ref_init_ = false;
  escape_dir_ = 0;
  have_valid_q_star_ = false;
  last_valid_q_star_ = q0;
  q_hat_ = q0[0];
  v_hat_ = 0.0;
  obs_init_ = true;
  last_sample_t_ = -1.0;
  last_slack_ = 0.0;
  slack_hold_latched_ = false;
  sat_scale_ = 1.0;
  quiet_s_ = 0.0;
  cmd_quiet_s_ = 0.0;
  quiescent_ = false;
  hold_d_prev_ = false;
  posture_gate_active_ = false;
  posture_gate_scale_ = 0.0;
  posture_gate_enter_s_ = 0.0;
  posture_gate_exit_s_ = 0.0;
  enabled_ = true;
  plan_drives_rail_ = false;
  direct_ptp_ = false;
  press_stall_s_ = 0.0;
  nudge_cool_s_ = 0.0;
  press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
  escape_active_ = false;
  escape_sign_ = 0.0;
  sec_qdot_.setZero();
  sec_acc_.setZero();
  sec_target_.setZero();
  sec_lpf_.setZero();
  gN_lpf_.setZero();
  gN_lpf_init_ = false;
  sec_age_ = 1e9;
  m_diag_init_ = false;
  box_t_init_ = false;
  box_h1_ = cfg_.dt;
  sigma_row_active_ = false;
  sigma_grad_.setZero();
  sigma_tick_ = 0;
  clear_rail_box_tel();
  task_weight_.reset();
  kin_.update(q0);
  posture_.reset(q0, kin_.fk_pose_at(q0));
  d_star_ = posture_.d_star();
  d_pref_ = d_star_;
  d0_ = d_star_;
  psi_cmd_ = posture_.psi_cmd();
  psi_star_ = posture_.psi_star();
  psi0_ = psi_cmd_;
  homotopy_s_ = 0.0;
  planned_ = false;
  q_star_ = q0;
  q_star_signs_ = q_nominal_;
}

void InnerLoop::begin_hybrid(const Vec8& q_meas, const Vec8& qdot_applied) {
  qdot_prev_ = qdot_applied;
  qdot_seen_ = qdot_applied;
  q_star_ = q_meas;
  q_star_signs_ = q_nominal_;
  kin_.update(q_meas);
  posture_.begin_unplanned(q_meas, kin_.fk_pose_at(q_meas));
  d_star_ = posture_.d_star();
  d_pref_ = d_star_;
  d0_ = d_star_;
  psi_cmd_ = posture_.psi_cmd();
  psi_star_ = posture_.psi_star();
  psi0_ = psi_cmd_;
  homotopy_s_ = 0.0;
  planned_ = false;
  hold_d_prev_ = false;
  slack_hold_latched_ = false;
  posture_gate_active_ = false;
  posture_gate_scale_ = 0.0;
  posture_gate_enter_s_ = 0.0;
  posture_gate_exit_s_ = 0.0;
}

void InnerLoop::set_rail_mode(uint32_t mode, uint32_t style, double q_ref, bool has_ref) {
  rail_mode_ = static_cast<int>(mode);
  locked_style_ = static_cast<int>(style);
  if (has_ref) {
    rail_q_ref_ = q_ref;
    has_rail_ref_ = true;
  } else if (mode == kRailLocked && style == kStyleHold) {
    rail_q_ref_ = q_cmd_[0];
    has_rail_ref_ = true;
  }
}

void InnerLoop::set_flags(uint32_t bits) {
  plan_drives_rail_ = bits & kFlagPlanDrivesRail;
  direct_ptp_ = bits & kFlagDirectPtp;
  arm_suppress_ = bits & kFlagArmSuppress;
  center_suppress_ = bits & kFlagCenterSuppress;
  manip_active_ = bits & kFlagManipActive;
  rail_ext_active_ = bits & kFlagRailExtActive;
}

void InnerLoop::set_stroke(double d_star, double psi_star) {
  d_star_ = d_star;
  d_pref_ = d_star;
  psi_star_ = psi_star;
  planned_ = true;
  posture_.set_planned_stroke(d_star, psi_star);
}

std::pair<double, double> InnerLoop::plan_stroke(const Vec8& q, double y_center, double amp) {
  kin_.update(q);
  const double y_lo = y_center - std::abs(amp);
  const double y_hi = y_center + std::abs(amp);
  const double rail_lo_s = cfg_.hard_min + cfg_.rail_margin;
  const double rail_hi_s = cfg_.hard_max - cfg_.rail_margin;
  const double d_min = y_hi - rail_hi_s;
  const double d_max = y_lo - rail_lo_s;
  if (d_min > d_max + 1e-9) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  const double d = 0.5 * (d_min + d_max);
  const double psi = fold_psi_to_positive(kin_.sew_psi(q));
  set_stroke(d, psi);
  return {d, psi};
}

void InnerLoop::set_rail_pose_target(double y, bool valid) {
  has_y_target_ = valid;
  y_rail_target_ = y;
}

void InnerLoop::capture_rail_ext_ref(const Vec8& q) {
  kin_.update(q);
  d_pref_ = kin_.tcp_xyz()[1] - q[0];
}

void InnerLoop::set_rail_ext_mode(int pose_attract) { rail_ext_mode_ = pose_attract; }

bool InnerLoop::apply_velocity_box(const Vec8& q_geom, const Vec8& q_cmd, const Vec8& q_meas,
                                   double dt, double h1, double h2, bool rail_locked,
                                   double rail_pin, bool has_pin, bool lead_exempt,
                                   Vec8* lo, Vec8* hi) {
  const double interval_tol = std::max(10.0 * cfg_.eps_abs, 1.0e-8);
  *lo = -v_max_;
  *hi = v_max_;
  rail_bind_lo_ = kRailBindVMaxDamper;
  rail_bind_hi_ = kRailBindVMaxDamper;
  Vec8 band = Vec8::Constant(cfg_.damper_band_rad);
  band[0] = cfg_.damper_band_rail;
  const Vec8 m = (Vec8() << cfg_.position_margin_rail_m,
                  cfg_.position_margin_rad, cfg_.position_margin_rad, cfg_.position_margin_rad,
                  cfg_.position_margin_rad, cfg_.position_margin_rad, cfg_.position_margin_rad,
                  cfg_.position_margin_rad)
                     .finished();
  double q_rail_hi = std::max(q_geom[0], q_cmd[0]);
  double q_rail_lo = std::min(q_geom[0], q_cmd[0]);
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    for (int i = 0; i < kNv; ++i) {
      const double b = std::max(band[i], 1e-9);
      double d_hi = clip(((q_hi_[i] - m[i]) - q_geom[i]) / b, 0.0, 1.0);
      double d_lo = clip((q_geom[i] - (q_lo_[i] + m[i])) / b, 0.0, 1.0);
      if (band[i] <= 1e-9) {
        d_hi = 1.0;
        d_lo = 1.0;
      }
      (*hi)[i] = std::min((*hi)[i], v_max_[i] * d_hi);
      (*lo)[i] = std::max((*lo)[i], -v_max_[i] * d_lo);
    }
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindVMaxDamper);
  }
  if (band[0] > 1e-9) {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    const double b0 = band[0];
    const double d_hi = clip((q_hi_[0] - m[0] - q_rail_hi) / b0, 0.0, 1.0);
    const double d_lo = clip((q_rail_lo - q_lo_[0] - m[0]) / b0, 0.0, 1.0);
    (*hi)[0] = std::min((*hi)[0], v_max_[0] * d_hi);
    (*lo)[0] = std::max((*lo)[0], -v_max_[0] * d_lo);
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindCmdDamper);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    double lo_cap, hi_cap;
    wall_cap(q_geom[0], q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_cap,
             &hi_cap);
    double lo_hi, hi_hi, lo_lo, hi_lo;
    wall_cap(q_rail_hi, q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_hi,
             &hi_hi);
    wall_cap(q_rail_lo, q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_lo,
             &hi_lo);
    (*hi)[0] = std::min({(*hi)[0], hi_cap, hi_hi, hi_lo});
    (*lo)[0] = std::max({(*lo)[0], lo_cap, lo_hi, lo_lo});
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindWallCap);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    for (int i = 0; i < kNv; ++i) {
      double p_lo = (q_lo_[i] + m[i] - q_geom[i]) / dt;
      double p_hi = (q_hi_[i] - m[i] - q_geom[i]) / dt;
      if (i == 0) {
        if (q_geom[0] < q_lo_[0] + m[0]) p_lo = std::min(p_lo, 0.0);
        if (q_geom[0] > q_hi_[0] - m[0]) p_hi = std::max(p_hi, 0.0);
      }
      (*lo)[i] = std::max((*lo)[i], p_lo);
      (*hi)[i] = std::min((*hi)[i], p_hi);
    }
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindPosBound);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    if (!reconcile_interval(lo, hi, interval_tol)) return false;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindCollapse);
  }
  const double a_dt = h1;
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    for (int i = 0; i < kNv; ++i) {
      (*lo)[i] = std::max((*lo)[i], qdot_prev_[i] - a_max_[i] * a_dt);
      (*hi)[i] = std::min((*hi)[i], qdot_prev_[i] + a_max_[i] * a_dt);
    }
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindAccel);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    if (!reconcile_interval(lo, hi, interval_tol)) return false;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindCollapse);
  }
  if (std::isfinite(h2) && h2 > 1e-9) {
    {
      const double olo = (*lo)[0];
      const double ohi = (*hi)[0];
      for (int i = 0; i < kNv; ++i) {
        const double centre = qdot_prev_[i] + (a_dt / h2) * (qdot_prev_[i] - qdot_prev2_[i]);
        const double span = j_max_[i] * a_dt * a_dt;
        (*lo)[i] = std::max((*lo)[i], centre - span);
        (*hi)[i] = std::min((*hi)[i], centre + span);
      }
      note_rail_bind(olo, ohi, *lo, *hi, kRailBindJerk);
    }
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    if (!reconcile_interval(lo, hi, interval_tol)) return false;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindCollapse);
  }
  Vec8 re = Vec8::Constant(cfg_.resync_err_rad);
  re[0] = cfg_.resync_err_rail_m;
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    for (int i = 0; i < kNv; ++i) {
      if (re[i] <= 0.0) continue;
      double lead = q_cmd[i] - q_meas[i];
      if (lead_exempt && i == 0) lead = 0.0;
      const double reaction = (i == 0) ? cfg_.rail_reaction_s : dt;
      const double toward_hi = stopping_velocity(re[i] - lead, a_max_[i], reaction);
      const double toward_lo = -stopping_velocity(re[i] + lead, a_max_[i], reaction);
      double chi = std::min((*hi)[i], toward_hi);
      double clo = std::max((*lo)[i], toward_lo);
      (*hi)[i] = chi;
      (*lo)[i] = clo;
    }
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindLead);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    if (!reconcile_interval(lo, hi, interval_tol)) return false;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindCollapse);
  }
  if (has_pin) {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    const double v = clip(rail_pin, (*lo)[0], (*hi)[0]);
    (*lo)[0] = v;
    (*hi)[0] = v;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindPin);
  } else if (rail_locked) {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    const double prev = qdot_prev_[0];
    double target = 0.0;
    if (std::abs(prev) <= cfg_.lock_vel_eps && (*lo)[0] <= 0.0 && 0.0 <= (*hi)[0]) {
      target = 0.0;
    } else {
      target = std::copysign(std::max(std::abs(prev) - a_max_[0] * dt, 0.0), prev);
      target = clip(target, (*lo)[0], (*hi)[0]);
    }
    (*lo)[0] = target;
    (*hi)[0] = target;
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindLocked);
  }
  return reconcile_interval(lo, hi, interval_tol);
}

void InnerLoop::tighten_branch(const Vec8& q, bool rail_open, Vec8* lo, Vec8* hi) {
  if (!cfg_.branch_enabled) return;
  double act = cfg_.branch_box_activate;
  if (act <= 1e-9) act = cfg_.branch_activate;
  const double eps = cfg_.branch_eps;
  const double band = std::max(act - eps, 1e-6);
  for (int i = 1; i < kNv; ++i) {
    const double qs = q_star_signs_[i];
    if (std::abs(qs) <= cfg_.branch_target_eps) continue;
    const double sign = qs >= 0.0 ? 1.0 : -1.0;
    const double margin = sign * q[i];
    const double d = clip((margin - eps) / band, 0.0, 1.0);
    if (sign > 0.0) (*lo)[i] = std::max((*lo)[i], -v_max_[i] * d);
    else (*hi)[i] = std::min((*hi)[i], v_max_[i] * d);
  }
  const double qs1 = q_star_signs_[1];
  if (std::abs(qs1) > cfg_.branch_target_eps) {
    const double sign1 = qs1 >= 0.0 ? 1.0 : -1.0;
    if (sign1 * q[1] > 0.0) {
      const double wall = std::max(cfg_.j1_overfold_abs, 1e-6);
      const double j1_eps = std::max(cfg_.j1_overfold_eps, 0.0);
      const double j1_act = std::max(cfg_.j1_overfold_activate, j1_eps + 1e-6);
      const double j1_band = std::max(j1_act - j1_eps, 1e-6);
      const double d_ov = clip((wall - std::abs(q[1]) - j1_eps) / j1_band, 0.0, 1.0);
      if (sign1 < 0.0) (*lo)[1] = std::max((*lo)[1], -v_max_[1] * d_ov);
      else (*hi)[1] = std::min((*hi)[1], v_max_[1] * d_ov);
    }
  }
  if (rail_open && q.size() > 4) {
    const double j4_eps = cfg_.j4_limit_eps;
    const double j4_act = std::max(cfg_.j4_limit_activate, j4_eps + 1e-6);
    const double j4_band = std::max(j4_act - j4_eps, 1e-6);
    const double d_hi = clip(((q_hi_[4] - q[4]) - j4_eps) / j4_band, 0.0, 1.0);
    const double d_lo = clip(((q[4] - q_lo_[4]) - j4_eps) / j4_band, 0.0, 1.0);
    (*hi)[4] = std::min((*hi)[4], v_max_[4] * d_hi);
    (*lo)[4] = std::max((*lo)[4], -v_max_[4] * d_lo);
  }
}

bool InnerLoop::solve_hqp(const Mat6x8& J, const Vec6& v_cmd, const Vec8& q_geom,
                          const Vec8& q_prev, const Vec8& qdot_nom, double rail_exec,
                          bool has_rail_exec, double rail_task_vel, double rail_w,
                          bool rail_locked, double dt, double h1, double h2, bool rail_open,
                          double rail_pin, bool has_pin, bool lead_exempt, double sigma_arm,
                          bool direct_pin, Vec8* qdot, Vec6* residual, double* slack) {
  qdot->setZero();
  *residual = v_cmd;
  *slack = residual->norm();
  const auto solve_start = std::chrono::steady_clock::now();
  const double certificate_tol = std::max(10.0 * cfg_.eps_abs, 1.0e-5);
  qp1_status_ = kQpNotRun;
  qp2_status_ = kQpNotRun;
  fallback_level_ = kFallbackNone;
  failure_code_ = kFailureNone;
  qp1_hard_violation_ = 0.0;
  final_hard_violation_ = 0.0;
  task_lock_violation_ = 0.0;
  final_box_violation_ = 0.0;
  qp_overrun_ = false;
  Mat6x8 J_task = J;
  Vec6 b_task = v_cmd;
  if (has_rail_exec) {
    const Vec6 rail_contrib = J.col(0) * rail_exec;
    J_task.col(0).setZero();
    b_task = v_cmd - rail_contrib;
  }
  rail_task_vel_used_ = rail_task_vel;
  rail_h1_ = h1;
  rail_h2_ = h2;
  rail_qdot_prev_tel_ = qdot_prev_[0];
  rail_qdot_prev2_tel_ = qdot_prev2_[0];
  Vec8 lo_box, hi_box;
  if (!apply_velocity_box(q_geom, q_prev, q_geom, dt, h1, h2, rail_locked, rail_pin,
                          has_pin, lead_exempt, &lo_box, &hi_box)) {
    rail_box_lo_ = lo_box[0];
    rail_box_hi_ = hi_box[0];
    failure_code_ = kFailureBoxInfeasible;
    fallback_level_ = kFallbackStop;
    return false;
  }
  {
    const double olo = lo_box[0];
    const double ohi = hi_box[0];
    tighten_branch(q_geom, rail_open, &lo_box, &hi_box);
    note_rail_bind(olo, ohi, lo_box, hi_box, kRailBindBranch);
  }
  {
    const double olo = lo_box[0];
    const double ohi = hi_box[0];
    if (!reconcile_interval(&lo_box, &hi_box, certificate_tol)) {
      rail_box_lo_ = lo_box[0];
      rail_box_hi_ = hi_box[0];
      failure_code_ = kFailureBoxInfeasible;
      fallback_level_ = kFallbackStop;
      return false;
    }
    note_rail_bind(olo, ohi, lo_box, hi_box, kRailBindCollapse);
  }
  rail_box_lo_ = lo_box[0];
  rail_box_hi_ = hi_box[0];

  // Direct joint PTP is still a QP path: make the requested feed-forward
  // velocity a hard per-joint target inside the same velocity/acceleration/
  // jerk box.  An out-of-box target is an explicit infeasibility, never a
  // post-solve rewrite.
  if (direct_pin) {
    for (int i = 0; i < kNv; ++i) {
      const double target = qdot_nom[i];
      if (!std::isfinite(target) || target < lo_box[i] - certificate_tol ||
          target > hi_box[i] + certificate_tol) {
        failure_code_ = kFailureBoxInfeasible;
        fallback_level_ = kFallbackStop;
        return false;
      }
      lo_box[i] = target;
      hi_box[i] = target;
    }
    rail_box_lo_ = lo_box[0];
    rail_box_hi_ = hi_box[0];
  }

  MatX C = MatX::Zero(kNIn, kNVar);
  VecX lo = VecX::Constant(kNIn, -1e20);
  VecX hi = VecX::Constant(kNIn, 1e20);
  C.block(0, 0, kNv, kNv) = Mat8::Identity();
  lo.head<kNv>() = lo_box;
  hi.head<kNv>() = hi_box;
  if (collision_) {
    collision_->update(q_geom, kin_.data());
    MatX cj;
    VecX cl;
    std::vector<int> slots;
    const int n = collision_->build_rows(kin_.data(), &cj, &cl, &slots);
    for (int i = 0; i < n; ++i) {
      C.block(kNv + i, 0, 1, kNv) = cj.row(i);
      lo[kNv + i] = cl[i];
      if (has_rail_exec) {
        lo[kNv + i] -= cj(i, 0) * rail_exec;
        C(kNv + i, 0) = 0.0;
      }
    }
  }
  for (int k = 0; k < kNPref; ++k) {
    C(kNv + kMaxCbf + kMaxPrefRows + k, kNv + kNTaskSlack + k) = 1.0;
    lo[kNv + kMaxCbf + kMaxPrefRows + k] = 0.0;
  }

  Eigen::Matrix<double, kNTaskSlack, kNTaskSlack> W = task_weight_.step(
      J_task, cfg_.task_weight, dt, cfg_.task_weight_lpf_tau_s, cfg_.sr_sigma_ref,
      cfg_.task_weight_min_frac, cfg_.aniso_task_damping);
  MatX H1 = MatX::Zero(kNVar, kNVar);
  H1.block(kNv, kNv, kNTaskSlack, kNTaskSlack) = W;
  VecX g1 = VecX::Zero(kNVar);
  MatX A1 = MatX::Zero(kNTaskSlack, kNVar);
  A1.leftCols<kNv>() = J_task;
  A1.block(0, kNv, kNTaskSlack, kNTaskSlack) = -Eigen::Matrix<double, 6, 6>::Identity();

  const double inset = std::max(2.0 * cfg_.eps_abs, 1e-8);
  VecX lo1 = lo, hi1 = hi;
  for (int i = 0; i < kNIn; ++i) {
    if (std::isfinite(lo1[i]) && std::isfinite(hi1[i]) && (hi1[i] - lo1[i]) > 2 * inset) {
      lo1[i] += inset;
      hi1[i] -= inset;
    }
  }

  try {
    solve_dense_qp(*qp1_, &qp1_inited_, qp1_last_ok_, H1, g1, A1, b_task, C, lo1, hi1);
    qp1_status_ = qp_status_code(qp1_->results.info.status);
  } catch (...) {
    qp1_status_ = kQpException;
  }
  const bool qp1_solved = qp1_status_ == kQpSolved;
  qp1_last_ok_ = qp1_solved;
  if (!qp1_solved) {
    failure_code_ = kFailureQp1Status;
    fallback_level_ = kFallbackStop;
    *qdot = Vec8::Zero();
    *residual = b_task;
    *slack = residual->norm();
    return false;
  }
  VecX x1 = qp1_->results.x;
  if (x1.size() != kNVar || !x1.allFinite()) {
    qp1_status_ = kQpNonfinite;
    failure_code_ = kFailureQp1Certificate;
    fallback_level_ = kFallbackStop;
    return false;
  }
  Vec8 qdot1 = x1.head<kNv>();
  if (has_rail_exec && !direct_pin) {
    double seed = rail_exec;
    if (std::isfinite(rail_task_vel) && !rail_locked) seed = rail_task_vel;
    qdot1[0] = clip(seed, lo_box[0], hi_box[0]);
    x1[0] = qdot1[0];
  }
  const double qp1_ineq = inequality_violation(C, x1, lo, hi);
  const double qp1_eq = equality_violation(A1, x1, b_task);
  qp1_hard_violation_ = std::max(qp1_ineq, qp1_eq);
  if (!std::isfinite(qp1_hard_violation_) || qp1_hard_violation_ > certificate_tol) {
    qp1_status_ = kQpCertificateFailed;
    failure_code_ = kFailureQp1Certificate;
    fallback_level_ = kFallbackStop;
    return false;
  }
  const double qp1_elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start)
          .count();
  if (qp1_elapsed_ms > cfg_.max_solve_ms) {
    qp_overrun_ = true;
    failure_code_ = kFailureSolveOverrun;
    fallback_level_ = kFallbackStop;
    return false;
  }
  const Vec6 t1 = J_task * qdot1;
  last_lock_J_ = has_rail_exec ? J_task : J;
  last_lock_v_ = t1;
  const MatX C_hard = C;
  const VecX lo_hard = lo;
  const VecX hi_hard = hi;

  Vec8 w_reg = cfg_.reg;
  if (rail_locked) w_reg[0] *= cfg_.lock_reg_scale;
  Vec8 h_reg = w_reg;
  if (cfg_.use_mass_weighted_reg) {
    Vec8 mdiag = kin_.mass().diagonal().cwiseMax(cfg_.mass_reg_floor);
    if (cfg_.mass_weight_exempt_rail) mdiag[0] = 1.0;
    if (cfg_.mass_reg_lpf_tau_s > 1e-9) {
      if (!m_diag_init_) {
        m_diag_lpf_ = mdiag;
        m_diag_init_ = true;
      } else {
        m_diag_lpf_ = first_order_lpf_vec(m_diag_lpf_, mdiag, dt, cfg_.mass_reg_lpf_tau_s);
      }
      mdiag = m_diag_lpf_;
    }
    h_reg = w_reg.cwiseProduct(mdiag);
  }
  VecX slack_w = VecX::Zero(kNPref);
  slack_w[0] = cfg_.sigma_slack_w;
  slack_w[1] = cfg_.branch_slack_w * dwell_scale_;
  for (int k = 2; k < kNPref; ++k) slack_w[k] = cfg_.comfort_slack_w;
  if (cfg_.j4_design_enabled) slack_w[2] = cfg_.j4_design_slack_w;
  else slack_w[2] = 0.0;

  MatX H2 = MatX::Zero(kNVar, kNVar);
  H2.topLeftCorner<kNv, kNv>() = h_reg.asDiagonal();
  H2.block(kNv, kNv, kNTaskSlack, kNTaskSlack) =
      1e-10 * Eigen::Matrix<double, 6, 6>::Identity();
  for (int k = 0; k < kNPref; ++k) H2(kNv + kNTaskSlack + k, kNv + kNTaskSlack + k) = slack_w[k];
  VecX g2 = VecX::Zero(kNVar);
  g2.head<kNv>() = -h_reg.cwiseProduct(qdot_nom);
  if (rail_w > 0.0 && !rail_locked && !has_pin) {
    H2(0, 0) += rail_w;
    g2[0] -= rail_w * rail_task_vel;
  }
  if (cfg_.smoothness.maxCoeff() > 0.0) {
    H2.topLeftCorner<kNv, kNv>() += cfg_.smoothness.cwiseMax(0.0).asDiagonal();
    g2.head<kNv>() -= cfg_.smoothness.cwiseMax(0.0).cwiseProduct(qdot_prev_);
  }

  // Comfort pref (J4)
  int pref_n = 0;
  MatX pref_J = MatX::Zero(kMaxPrefRows, kNv);
  VecX pref_lo = VecX::Zero(kMaxPrefRows);
  Eigen::VectorXi pref_s = Eigen::VectorXi::Zero(kMaxPrefRows);
  if (cfg_.comfort_enabled) {
    const double d_hi = q_hi_[4] - q_geom[4];
    const double d_lo = q_geom[4] - q_lo_[4];
    const double margin = std::min(d_hi, d_lo);
    const double band = std::max(cfg_.comfort_activate - cfg_.comfort_m, 1e-6);
    const double w = smoothstep01((cfg_.comfort_activate - margin) / band);
    if (w > 1e-6) {
      pref_J(pref_n, 4) = (d_hi <= d_lo) ? -w : w;
      pref_lo[pref_n] = -cfg_.comfort_gamma * (margin - cfg_.comfort_m) * w;
      pref_s[pref_n] = 2 + 3;  // J4 physical slack
      ++pref_n;
    }
  }
  if (cfg_.j4_design_enabled) {
    const int j4 = j4_index(kNv);
    pref_J(pref_n, j4) = 1.0;
    pref_lo[pref_n] = -cfg_.j4_design_gamma * (q_geom[j4] - cfg_.j4_design_lo);
    pref_s[pref_n] = 2;
    ++pref_n;
    pref_J(pref_n, j4) = -1.0;
    pref_lo[pref_n] = -cfg_.j4_design_gamma * (cfg_.j4_design_hi - q_geom[j4]);
    pref_s[pref_n] = 2;
    ++pref_n;
  }
  if (cfg_.sigma_enabled) {
    if (sigma_arm < cfg_.sigma_activate) sigma_row_active_ = true;
    if (sigma_arm >= cfg_.sigma_exit) sigma_row_active_ = false;
    if (sigma_row_active_ && sigma_grad_.norm() > 1e-12) {
      pref_J.row(pref_n) = sigma_grad_.transpose();
      pref_lo[pref_n] = -cfg_.sigma_gamma * (sigma_arm - cfg_.sigma_safe);
      pref_s[pref_n] = 0;
      ++pref_n;
    }
  }
  const int pref_base = kNv + kMaxCbf;
  for (int k = 0; k < pref_n; ++k) {
    C.block(pref_base + k, 0, 1, kNv) = pref_J.row(k);
    C(pref_base + k, kNv + kNTaskSlack + pref_s[k]) = 1.0;
    lo[pref_base + k] = pref_lo[k];
  }

  MatX A2 = MatX::Zero(kNTaskSlack, kNVar);
  A2.leftCols<kNv>() = last_lock_J_;
  VecX x2_seed = VecX::Zero(kNVar);
  x2_seed.head<kNv>() = qdot1;
  for (int k = 0; k < kNPref; ++k) {
    const int col = kNv + kNTaskSlack + k;
    double needed = 0.0;
    for (int row = 0; row < kNIn; ++row) {
      if (C(row, col) <= 0.5) continue;
      const double base = C.row(row).head(kNv).dot(qdot1);
      needed = std::max(needed, lo[row] - base);
    }
    x2_seed[col] = std::max(needed, 0.0) + inset;
  }
  try {
    if (!qp2_inited_) {
      qp2_->init(H2, g2, A2, last_lock_v_, C, lo, hi, true);
      qp2_inited_ = true;
    } else {
      qp2_->update(H2, g2, A2, last_lock_v_, C, lo, hi, true);
    }
    qp2_->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START;
    qp2_->solve(x2_seed, proxsuite::nullopt, proxsuite::nullopt);
    qp2_status_ = qp_status_code(qp2_->results.info.status);
  } catch (...) {
    qp2_status_ = kQpException;
  }
  bool qp2_ok = qp2_status_ == kQpSolved && qp2_->results.x.size() == kNVar &&
                qp2_->results.x.allFinite();
  double qp2_hard = std::numeric_limits<double>::infinity();
  double qp2_lock = std::numeric_limits<double>::infinity();
  if (qp2_ok) {
    qp2_hard = inequality_violation(C, qp2_->results.x, lo, hi);
    qp2_lock = equality_violation(A2, qp2_->results.x, last_lock_v_);
    qp2_ok = std::isfinite(qp2_hard) && std::isfinite(qp2_lock) &&
             qp2_hard <= certificate_tol && qp2_lock <= certificate_tol;
    if (!qp2_ok) qp2_status_ = kQpCertificateFailed;
  } else if (qp2_status_ == kQpSolved) {
    qp2_status_ = kQpNonfinite;
  }
  qp2_last_ok_ = qp2_ok;
  Vec8 qdot_out = qdot1;
  VecX x_out = x1;
  MatX C_out = C_hard;
  VecX lo_out = lo_hard;
  VecX hi_out = hi_hard;
  if (qp2_ok) {
    qdot_out = qp2_->results.x.head<kNv>();
    x_out = qp2_->results.x;
    C_out = C;
    lo_out = lo;
    hi_out = hi;
    final_hard_violation_ = qp2_hard;
    task_lock_violation_ = qp2_lock;
  } else {
    fallback_level_ = kFallbackQp1;
    failure_code_ = (qp2_status_ == kQpCertificateFailed || qp2_status_ == kQpNonfinite)
                        ? kFailureQp2Certificate
                        : kFailureQp2Status;
    final_hard_violation_ = qp1_ineq;
    task_lock_violation_ = (last_lock_J_ * qdot1 - last_lock_v_).lpNorm<Eigen::Infinity>();
  }
  j4_design_slack_ = 0.0;
  sigma_slack_ = 0.0;
  if (x_out.size() > kNv + kNTaskSlack) {
    sigma_slack_ = std::max(0.0, x_out[kNv + kNTaskSlack + 0]);
  }
  if (x_out.size() > kNv + kNTaskSlack + 2) {
    j4_design_slack_ = std::max(0.0, x_out[kNv + kNTaskSlack + 2]);
  }
  final_box_violation_ = box_violation(qdot_out, lo_box, hi_box);
  const double final_eq = (last_lock_J_ * qdot_out - last_lock_v_).lpNorm<Eigen::Infinity>();
  const double final_hard = inequality_violation(C_out, x_out, lo_out, hi_out);
  final_hard_violation_ = std::max(final_hard_violation_, final_hard);
  task_lock_violation_ = std::max(task_lock_violation_, final_eq);
  if (!std::isfinite(final_hard_violation_) || !std::isfinite(task_lock_violation_) ||
      !std::isfinite(final_box_violation_) || final_hard_violation_ > certificate_tol ||
      task_lock_violation_ > certificate_tol || final_box_violation_ > certificate_tol) {
    failure_code_ = kFailureFinalCertificate;
    fallback_level_ = kFallbackStop;
    return false;
  }
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start)
          .count();
  if (elapsed_ms > cfg_.max_solve_ms) {
    qp_overrun_ = true;
    failure_code_ = kFailureSolveOverrun;
    fallback_level_ = kFallbackStop;
    return false;
  }
  *qdot = qdot_out;
  *residual = b_task - J_task * qdot_out;
  *slack = residual->norm();
  last_C_ = C_out;
  last_lo_ = lo_out;
  last_hi_ = hi_out;
  last_x_ = x_out;
  return true;
}

TickOut InnerLoop::step(const TickIn& in) {
  const auto t0 = std::chrono::steady_clock::now();
  TickOut out;
  uint32_t input_flags = in.flags;
  const bool command_stale = !enabled_ || (input_flags & kInStale);
  if (command_stale) {
    input_flags &= ~(kInHasQdotFf | kInHasPoseD | kInHasVelFf | kInHasPathTwist |
                     kInHasFeedbackTwist | kInHasPosture | kInHasQStar | kInHasVForce);
    out.flags |= kOutStale;
  }
  if (!finite_tick_input(in, input_flags)) {
    qp1_status_ = kQpNotRun;
    qp2_status_ = kQpNotRun;
    fallback_level_ = kFallbackStop;
    failure_code_ = kFailureInputNonfinite;
    qp_overrun_ = false;
    out.q_cmd = q_cmd_;
    out.qdot.setZero();
    out.status = kStatusFail;
    out.flags |= kOutFailed;
    out.failure_code = failure_code_;
    out.fallback_level = fallback_level_;
    return out;
  }
  Vec6 twist = in.v_cmd;
  if (command_stale) {
    twist.setZero();
  }
  const double dt_nom = (std::isfinite(in.dt_nom) && in.dt_nom > 0.0) ? in.dt_nom : cfg_.dt;
  const double dt = integration_period(dt_nom, in.dt_wall);
  if (input_flags & kInSeedQcmd) {
    q_cmd_ = in.q_meas;
    q_hat_ = in.q_meas[0];
    if (input_flags & kInHasQdotFf) {
      qdot_prev_ = in.qdot_ff;
      qdot_seen_ = in.qdot_ff;
      qdot_prev2_ = in.qdot_ff;
      dq_prev_ = in.qdot_ff * dt;
      have_dq_prev_ = true;
    }
  }
  Vec8 q_prev = q_cmd_;
  if (rail_mode_ == kRailCoupled && obs_init_ && last_sample_t_ >= 0.0) {
    q_prev[0] = q_hat_;
  }
  const Vec8 q_state = in.q_meas;
  kin_.update(q_state);
  const Mat6x8 J = kin_.jacobian();
  last_sigma_ = kin_.sigma_min();
  const double sigma_arm = kin_.sigma_arm();
  const Eigen::Matrix3d R = kin_.tcp_R();
  const Vec6 twist_base = twist_to_base(twist, R, cfg_.control_frame == "tool");
  out.v_recv = twist_base;

  // Stale/disabled input is a control ownership boundary.  Do not run the
  // allocator with an old rail observer velocity or feed-forward state; hold
  // q_cmd and publish an explicit zero-velocity stop record.
  if (command_stale) {
    out.q_cmd = q_cmd_;
    out.qdot.setZero();
    out.v_feas.setZero();
    out.v_tcp.setZero();
    out.residual = twist_base;
    out.slack = twist_base.norm();
    out.e_qp = out.slack;
    out.flags |= kOutStale | kOutFailed;
    out.status = kStatusFail;
    out.qp1_status = kQpNotRun;
    out.qp2_status = kQpNotRun;
    out.fallback_level = kFallbackStop;
    out.failure_code = kFailureInputStale;
    out.posture_gate = posture_gate_scale_;
    return out;
  }

  const bool locked_hold = rail_mode_ == kRailLocked && locked_style_ == kStyleHold;
  const bool rail_only = rail_mode_ == kRailLocked && locked_style_ == kStyleRailOnly;
  const bool tcp_fixed = rail_mode_ == kRailLocked && locked_style_ == kStyleTcpFixed;

  double rail_exec = qdot_prev_[0];
  bool has_rail_exec = false;
  if (input_flags & kInHasRailV) {
    rail_exec = in.rail_v;
    has_rail_exec = true;
  } else if (obs_init_ && last_sample_t_ >= 0.0) {
    rail_exec = v_hat_;
    has_rail_exec = true;
  }

  const double now = in.t_mono > 0.0 ? in.t_mono : 0.0;
  if (!obs_init_) {
    q_hat_ = q_state[0];
    v_hat_ = 0.0;
    obs_init_ = true;
  } else {
    const double v_pred = (input_flags & kInHasRailV) ? in.rail_v : v_hat_;
    q_hat_ += v_pred * dt;
    v_hat_ = first_order_lpf(v_hat_, v_pred, dt, lpf_tau_from_fc(cfg_.observer_vel_lpf_hz));
    if (now > last_sample_t_ + 1e-9) {
      const double age = std::max(0.0, now - (last_sample_t_ < 0 ? now : last_sample_t_));
      const double innov = q_state[0] - (q_hat_ - v_pred * age);
      q_hat_ += cfg_.observer_pos_gain * innov;
      v_hat_ += cfg_.observer_vel_gain * innov;
      last_sample_t_ = now;
    }
    v_hat_ = clip(v_hat_, -v_max_[0], v_max_[0]);
  }

  double h1 = dt;
  double h2 = std::numeric_limits<double>::quiet_NaN();
  if (!box_t_init_) {
    box_t_init_ = true;
    box_h1_ = dt;
  } else {
    h2 = box_h1_;
    box_h1_ = dt;
  }

  const double z_now = kin_.tcp_xyz()[2];
  const double y_tcp = kin_.tcp_xyz()[1];
  double y_tcp_d = y_tcp;
  bool has_pose_d = input_flags & kInHasPoseD;
  if (has_pose_d && std::isfinite(in.pose_d[1])) y_tcp_d = in.pose_d[1];
  const double tool_y_err = y_tcp_d - y_tcp;

  const bool contact = input_flags & kInContact;
  const bool has_vf = input_flags & kInHasVForce;
  const bool demanding = contact && has_vf && std::abs(in.v_force_z) >= cfg_.press_v_force_min;
  if (demanding) {
    if (!std::isfinite(press_z_mark_)) press_z_mark_ = z_now;
    if (std::abs(z_now - press_z_mark_) > cfg_.press_dz_max) {
      press_z_mark_ = z_now;
      press_stall_s_ = 0.0;
    } else {
      press_stall_s_ += dt;
    }
  } else {
    press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
    press_stall_s_ = 0.0;
  }
  const bool press_stalled = press_stall_s_ + 1e-12 >= cfg_.press_stall_s;
  double soft_lo = 0.0, soft_hi = 0.0;
  soft_rail_travel(q_lo_[0], q_hi_[0], cfg_.soft_min, cfg_.soft_max, &soft_lo, &soft_hi);
  const auto soft = std::pair<double, double>{soft_lo, soft_hi};
  const double open_travel = std::max(q_state[0] - soft.first, soft.second - q_state[0]);
  const bool has_travel = open_travel > cfg_.open_travel_min;
  const bool j4_blocked =
      (q_hi_[4] - q_prev[4]) <= cfg_.comfort_m || (q_prev[4] - q_lo_[4]) <= cfg_.comfort_m;
  const bool arm_starved = std::abs(tool_y_err) >= cfg_.press_y_err;
  const double pol_sign = policy_escape_sign(
      cfg_.escape_sign_policy, q_state[0], soft.first, soft.second,
      escape_active_ ? escape_sign_ : 0.0);
  const bool policy_leave = in_leave_band(q_state[0], soft.first, soft.second,
                                          leave_margin_m(cfg_.escape_leave, cfg_.pin_margin),
                                          pol_sign);
  const bool allow_press = press_escape_allowed_from_flags(
      demanding, has_travel, press_stalled, j4_blocked, arm_starved, policy_leave);

  const double lin = twist_base.head<3>().norm();
  const double rot = twist_base.tail<3>().norm();
  const double tcp_lin = last_tcp_est_.head<3>().norm();
  const bool cmd_quiet_enter = lin < kQuietLinEnter && rot < kQuietRotEnter;
  const bool cmd_active_exit = lin > kQuietLinExit || rot > kQuietRotExit;
  const bool tcp_quiet = tcp_lin < kQuietTcp;
  if (cmd_quiet_enter && tcp_quiet) quiet_s_ += dt;
  else quiet_s_ = 0.0;
  if (allow_press) {
    quiescent_ = false;
    cmd_quiet_s_ = 0.0;
  } else if (quiescent_) {
    if (cmd_active_exit) {
      quiescent_ = false;
      cmd_quiet_s_ = 0.0;
    }
  } else {
    if (cmd_quiet_enter) cmd_quiet_s_ += dt;
    else cmd_quiet_s_ = 0.0;
    if (cmd_quiet_s_ + 1e-12 >= kQuietHold) quiescent_ = true;
  }
  if (!slack_hold_latched_ && last_slack_ >= cfg_.slack_enter) slack_hold_latched_ = true;
  else if (slack_hold_latched_ && last_slack_ <= cfg_.slack_exit) slack_hold_latched_ = false;
  const bool slack_high = slack_hold_latched_;
  const bool hold_d = quiescent_ || slack_high;
  hold_d_prev_ = hold_d;

  // Gate posture/d-star motion on task relevance of the physical rail
  // direction.  Allocation, hard boxes and explicit escape remain untouched.
  if (cfg_.posture_gate_enabled && rail_mode_ == kRailCoupled && !quiescent_ &&
      !command_stale) {
    Eigen::Vector3d gate_ref = twist_base.head<3>();
    if ((input_flags & kInHasPathTwist) && in.path_twist.head<3>().norm() > 1e-9) {
      gate_ref = twist_to_base(in.path_twist, R, cfg_.control_frame == "tool").head<3>();
    } else if ((input_flags & kInHasVelFf) && in.vel_ff.head<3>().norm() > 1e-9) {
      gate_ref = twist_to_base(in.vel_ff, R, cfg_.control_frame == "tool").head<3>();
    }
    const double ref_norm = gate_ref.norm();
    const Eigen::Vector3d jrail = J.topLeftCorner<3, 1>();
    const double jnorm = jrail.norm();
    const double projected = (ref_norm > 1e-9 && jnorm > 1e-9)
                                 ? std::abs(jrail.dot(gate_ref)) / jnorm
                                 : 0.0;
    const double ratio = ref_norm > 1e-9 ? projected / ref_norm : 0.0;
    const bool enter_candidate = ratio >= cfg_.posture_gate_enter_ratio &&
                                 projected >= cfg_.posture_gate_enter_speed;
    const bool exit_candidate = ratio <= cfg_.posture_gate_exit_ratio ||
                                projected <= cfg_.posture_gate_exit_speed;
    if (!posture_gate_active_) {
      posture_gate_exit_s_ = 0.0;
      posture_gate_enter_s_ = enter_candidate ? posture_gate_enter_s_ + dt : 0.0;
      if (posture_gate_enter_s_ + 1e-12 >= cfg_.posture_gate_enter_dwell) {
        posture_gate_active_ = true;
        posture_gate_enter_s_ = 0.0;
      }
    } else {
      posture_gate_enter_s_ = 0.0;
      posture_gate_exit_s_ = exit_candidate ? posture_gate_exit_s_ + dt : 0.0;
      if (posture_gate_exit_s_ + 1e-12 >= cfg_.posture_gate_exit_dwell) {
        posture_gate_active_ = false;
        posture_gate_exit_s_ = 0.0;
      }
    }
  } else if (command_stale) {
    posture_gate_active_ = false;
    posture_gate_enter_s_ = posture_gate_exit_s_ = 0.0;
  } else if (!cfg_.posture_gate_enabled) {
    posture_gate_active_ = true;
    posture_gate_enter_s_ = posture_gate_exit_s_ = 0.0;
  }
  {
    const double target = posture_gate_active_ ? 1.0 : 0.0;
    const double tau = posture_gate_active_ ? cfg_.posture_gate_open_tau
                                            : cfg_.posture_gate_close_tau;
    if (tau <= 1e-9 || dt <= 0.0) {
      posture_gate_scale_ = target;
    } else {
      posture_gate_scale_ += clip(dt / tau, 0.0, 1.0) * (target - posture_gate_scale_);
    }
  }

  if (cfg_.psi_enabled && rail_mode_ == kRailCoupled) {
    const Vec6 pose = kin_.fk_pose_at(q_prev);
    if ((!slack_high || quiescent_) && posture_gate_active_) {
      posture_.step(q_prev, pose, dt, q_lo_[0], q_hi_[0], quiescent_);
    }
    d_star_ = posture_.d_star();
    psi_cmd_ = posture_.psi_cmd();
    psi_star_ = posture_.psi_star();
    homotopy_s_ = posture_.homotopy_s();
    planned_ = posture_.planned();
    d_pref_ = d_star_;
    const Vec8 cand = posture_.q_star();
    if (q_finite_in_limits(cand, q_lo_, q_hi_)) {
      q_star_ = cand;
      last_valid_q_star_ = cand;
      have_valid_q_star_ = true;
    } else if (have_valid_q_star_) {
      q_star_ = last_valid_q_star_;
    } else {
      q_star_ = q_nominal_;
    }
  }
  // Host-supplied posture/q* references are optional per-tick ownership.  The
  // protocol carries them explicitly so native shadow execution follows the
  // same target as Python; absent flags leave the internally retargeted value
  // untouched.
  if (input_flags & kInHasPosture) {
    if (std::isfinite(in.posture_d)) {
      d_star_ = in.posture_d;
      d_pref_ = in.posture_d;
    }
    if (std::isfinite(in.posture_psi)) {
      psi_cmd_ = in.posture_psi;
      psi_star_ = in.posture_psi;
    }
  }
  if (input_flags & kInHasQStar) {
    if (in.posture_q.allFinite() && q_finite_in_limits(in.posture_q, q_lo_, q_hi_)) {
      q_star_ = in.posture_q;
      last_valid_q_star_ = in.posture_q;
      have_valid_q_star_ = true;
    }
  }
  if (press_stalled && allow_press && contact && has_vf && nudge_cool_s_ <= 0.0) {
    const double away = (q_prev[0] > 0.5 * (soft.first + soft.second)) ? 1.0 : -1.0;
    const double y_des = has_pose_d ? y_tcp_d : y_tcp;
    const double d_n = posture_.nudge_d_star(-away * cfg_.d_star_nudge, y_des, soft.first,
                                            soft.second, dt);
    if (std::isfinite(d_n)) {
      d_star_ = d_n;
      d_pref_ = d_n;
    }
    nudge_cool_s_ = cfg_.press_stall_s;
  } else {
    nudge_cool_s_ = std::max(0.0, nudge_cool_s_ - dt);
  }

  double rail_task_vel = 0.0;
  double rail_task_w = 0.0;
  double rail_ext_vel = 0.0;
  bool pose_attract_mode = false;
  bool have_rail_vel = false;
  last_v_escape_ = 0.0;
  last_e_mid_ = 0.0;
  if (cfg_.rail_ext_enabled && rail_ext_active_ && rail_mode_ == kRailCoupled) {
    const double y = q_state[0];
    // e_mid = (y_tcp − d*) − y_rail. SERVO_TWIST latches pose_d at set_origin;
    // using that Y pulls the rail back to the start instead of tracking d*.
    const double y_des = y_tcp;
    const bool pose_attract = rail_ext_mode_ != 0 && has_y_target_;
    pose_attract_mode = pose_attract;
    const double rail_ff = pose_attract ? y_rail_target_ : (y_des - d_pref_);
    const double err_raw = rail_ff - y;
    double band = (planned_ || pose_attract) ? 0.0 : cfg_.d_band;
    const double err = err_raw - clip(err_raw, -band, band);
    last_e_mid_ = err;
    if (pose_attract) {
      const double e0 = std::max(cfg_.pose_e0, 0.0);
      const double e1 = std::max(cfg_.pose_e1, e0 + 1e-6);
      rail_ext_vel = std::abs(err) <= e0 ? 0.0 : clip(cfg_.k_pose * err, -cfg_.v_max_ext,
                                                       cfg_.v_max_ext);
      rail_task_w = cfg_.pose_w_max * smoothstep01((std::abs(err) - e0) / (e1 - e0));
    }
    double v_ff = 0.0;
    if (input_flags & kInHasVelFf) {
      const Eigen::Vector3d j_rail = J.topLeftCorner<3, 1>();
      const double den = j_rail.squaredNorm();
      if (den > 1e-12) v_ff = cfg_.k_ff * j_rail.dot(in.vel_ff.head<3>()) / den;
    }
    last_v_ff_ = v_ff;
    if (allow_press) {
      if (!escape_active_ || std::abs(escape_sign_) < 1.0e-12) {
        escape_sign_ = policy_escape_sign(cfg_.escape_sign_policy, y, soft.first, soft.second, 0.0);
      }
      last_v_escape_ = clip(0.25 * cfg_.k_esc * escape_sign_, -cfg_.v_max_ext, cfg_.v_max_ext);
      escape_active_ = std::abs(last_v_escape_) > 1e-12;
    } else {
      escape_active_ = false;
      last_v_escape_ = 0.0;
      escape_sign_ = 0.0;
    }
    last_ext_w_ = cfg_.w_max_ext;
    rail_task_w = last_ext_w_;
  } else {
    escape_active_ = false;
    last_v_escape_ = 0.0;
    escape_sign_ = 0.0;
  }

  if (rail_mode_ == kRailCoupled && !locked_hold) {
    const double d_live = y_tcp - q_state[0];
    const bool hold_ref = (hold_d && !escape_active_) || !posture_gate_active_ ||
                          !std::isfinite(d_star_);
    if (!d_star_ref_init_ || !std::isfinite(d_star_ref_)) {
      d_star_ref_ = d_live;
      d_star_ref_init_ = true;
      d_star_dot_cmd_ = 0.0;
    } else if (hold_ref || dt <= 1e-12) {
      d_star_dot_cmd_ = 0.0;
    } else {
      const double lim = std::abs(cfg_.d_center_rate) * dt;
      const double delta = clip(d_star_ - d_star_ref_, -lim, lim);
      d_star_ref_ += delta;
      d_star_dot_cmd_ = delta / dt;
    }
    d_pref_ = d_star_ref_;
    e_d_ = d_live - d_star_ref_;
    last_e_mid_ = e_d_;
    V_d_proxy_ = 0.5 * cfg_.kp_mid * e_d_ * e_d_;

    const double lam = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                         cfg_.sr_sigma_floor);
    const Vec8 mw = margin_weight_from_activation(q_prev, q_mid_, half_, cfg_.k_margin,
                                                  cfg_.ns_activation);
    auto [u_a, qall] = allocate_rail(J, twist_base, v_max_, mw, lam, cfg_.v0, cfg_.w0,
                                     last_e_mid_, cfg_.k_err_rail, cfg_.e_ref);
    (void)qall;
    u_alloc_ = u_a;
    u_task_raw_ = u_a;
    u_escape_raw_ = last_v_escape_;
    const double leave = wall_leave_only_sign(q_state[0], q_lo_[0], q_hi_[0], cfg_.damper_band_rail);
    wall_pi_frozen_ = (leave != 0.0);
    if (wall_pi_frozen_) {
      v_r_lpf_ = project_lpf_into_wall(v_r_lpf_, leave);
      v_r_ref_ = project_lpf_into_wall(v_r_ref_, leave);
    }
    escape_dir_ = update_escape_dir(escape_active_, u_escape_raw_, escape_dir_);
    const int guard_dir = escape_active_ ? escape_dir_ : 0;
    double u_lo = 0.0, u_hi = 0.0;
    wall_velocity_bounds(v_max_[0], leave, &u_lo, &u_hi);

    const bool task_hold = (quiescent_ || command_stale) && !escape_active_;
    const bool posture_hold = slack_high || task_hold || !posture_gate_active_ || command_stale;
    if (posture_hold) {
      mid_integ_ = -cfg_.kp_mid * e_d_;
      u_pi_raw_ = 0.0;
      u_mid_cmd_ = 0.0;
      u_post_raw_ = 0.0;
      d_star_dot_cmd_ = 0.0;
      u_mid_ = 0.0;
    } else {
      u_pi_raw_ = cfg_.kp_mid * e_d_ + mid_integ_;
      u_mid_cmd_ = clip(u_pi_raw_, -cfg_.u_mid_max, cfg_.u_mid_max);
      u_post_raw_ = (u_mid_cmd_ - d_star_dot_cmd_) * posture_gate_scale_;
      u_mid_ = u_mid_cmd_;
    }
    const RailShares shares = allocate_rail_shares(
        task_hold ? 0.0 : u_task_raw_,
        posture_hold ? 0.0 : u_post_raw_,
        task_hold ? 0.0 : u_escape_raw_,
        task_hold ? 0 : guard_dir,
        u_lo, u_hi);
    u_task_feasible_ = shares.u_task_feasible;
    u_escape_feasible_ = shares.u_escape_feasible;
    u_base_ = shares.u_base;
    u_post_feasible_ = shares.u_post_feasible;
    u_feasible_ = shares.u_feasible;
    u_mid_applied_ = u_post_feasible_ + d_star_dot_cmd_;
    if (!posture_hold && !wall_pi_frozen_ && dt > 0.0) {
      mid_integ_ += (cfg_.ki_mid * e_d_ + cfg_.kaw_mid * (u_mid_applied_ - u_pi_raw_)) * dt;
    }

    auto [a_mir, j_mir] = arm_mirror_rail_limits(J, a_max_, j_max_, cfg_.rho_a, cfg_.rho_j);
    const double tau = lpf_tau_from_fc(cfg_.f_c_hz);
    double v_f = u_feasible_;
    if (v_r_init_ && tau > 1e-9) v_f = first_order_lpf(v_r_lpf_, u_feasible_, dt, tau);
    v_r_init_ = true;
    v_r_lpf_ = v_f;
    const double v_prev = v_r_ref_;
    double a_raw = (v_f - v_prev) / dt;
    const double a_lim = std::min(cfg_.a_max_rail, a_mir);
    const double j_lim = std::min(cfg_.j_max_rail, j_mir);
    double a = clip(a_raw, v_r_a_ - j_lim * dt, v_r_a_ + j_lim * dt);
    a = clip(a, -a_lim, a_lim);
    double v = clip(v_prev + a * dt, -v_max_[0], v_max_[0]);
    double lo_c, hi_c;
    wall_cap(q_state[0], cfg_.hard_min, cfg_.hard_max, a_lim, cfg_.rail_reaction_s, &lo_c, &hi_c);
    v = clip(v, lo_c, hi_c);
    if (std::abs(v) < 5e-4 && std::abs(u_feasible_) < 5e-4) v = 0.0;
    v_r_ref_ = v;
    v_r_a_ = (v - v_prev) / dt;
    rail_task_vel = pose_attract_mode ? rail_ext_vel : v;
    have_rail_vel = true;
    rail_task_w = std::max(rail_task_w, kRailPrefW);
  } else {
    u_alloc_ = 0.0;
    u_mid_ = 0.0;
    u_feasible_ = 0.0;
    u_mid_applied_ = 0.0;
  }

  Vec8 qdot_center = Vec8::Zero();
  if (!center_suppress_) {
    const Vec8 u_t = (q_prev - q_star_).cwiseQuotient(half_);
    qdot_center = -cfg_.k_center * u_t;
    if (cfg_.k_limit > 0.0) {
      const Vec8 u_l = (q_prev - q_mid_).cwiseQuotient(half_);
      const double span = std::max(1.0 - cfg_.ns_activation, 1e-6);
      for (int i = 0; i < kNv; ++i) {
        const double over = clip((std::abs(u_l[i]) - cfg_.ns_activation) / span, 0.0, 1.0);
        qdot_center[i] -= cfg_.k_limit * ((u_l[i] >= 0) ? 1.0 : -1.0) * over * over;
      }
    }
  }
  Vec8 qdot_damp = Vec8::Zero();
  if (cfg_.d_null > 0.0) qdot_damp = cfg_.d_null * qdot_prev_;
  Vec8 sec = qdot_center - qdot_damp;
  sec[0] = 0.0;
  Vec8 qdot_mu = Vec8::Zero();
  if (manip_active_ && cfg_.k_mu > 0.0 && sigma_grad_.norm() > 1e-12) {
    const double fade = clip(last_sigma_ / std::max(cfg_.sigma_fade_ref, 1e-9), 0.0, 1.0);
    qdot_mu = cfg_.k_mu * fade * sigma_grad_;
    sec += qdot_mu;
  }
  Vec8 qdot_arm = Vec8::Zero();
  if (cfg_.arm_enabled && !arm_suppress_) {
    const double psi = kin_.sew_psi(q_prev);
    const double err = wrap_pi(psi_cmd_ - fold_psi_to_positive(psi));
    Vec8 g = Vec8::Zero();
    const double eps = std::max(cfg_.fd_eps, 1e-5);
    for (int i = 1; i < kNv; ++i) {
      Vec8 qp = q_prev, qm = q_prev;
      qp[i] += eps;
      qm[i] -= eps;
      g[i] = (kin_.sew_psi(qp) - kin_.sew_psi(qm)) / (2 * eps);
    }
    kin_.update(q_state);
    const double lam = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                         cfg_.sr_sigma_floor);
    const Vec8 gN = project_nullspace(J, g, lam);
    if (!gN_lpf_init_) {
      gN_lpf_ = gN;
      gN_lpf_init_ = true;
    } else {
      gN_lpf_ = first_order_lpf_vec(gN_lpf_, gN, dt, lpf_tau_from_fc(cfg_.ns_grad_lpf_hz));
    }
    const double den = std::max(g.dot(gN_lpf_), cfg_.safe_denom_eps);
    qdot_arm = cfg_.k_psi * err * gN_lpf_ / den;
    sec += qdot_arm;
  }
  {
    const double enter = cfg_.slack_enter;
    const double exit_ = cfg_.slack_exit;
    const double lo = clip(cfg_.secondary_scale, 0.0, 1.0);
    const double span = std::max(enter - exit_, 1e-9);
    const double s = clip((last_slack_ - exit_) / span, 0.0, 1.0);
    const double target = 1.0 + smoothstep01(s) * (lo - 1.0);
    const double tau = cfg_.secondary_scale_tau_s;
    if (tau <= 1e-9 || dt <= 0.0) {
      sat_scale_ = target;
    } else {
      const double alpha = std::min(1.0, dt / tau);
      sat_scale_ += alpha * (target - sat_scale_);
    }
    sec.tail<7>() *= sat_scale_;
  }
  sec[0] = 0.0;
  out.ns_centering = qdot_center.norm();
  out.ns_manip = qdot_mu.norm();
  out.ns_arm_angle = qdot_arm.norm();
  out.ns_damping = qdot_damp.norm();
  out.ns_rail_lock = 0.0;
  if (quiescent_) sec.setZero();
  sec_lpf_ = first_order_lpf_vec(sec_lpf_, sec, dt, lpf_tau_from_fc(cfg_.sec_input_lpf_hz));
  const double period = 1.0 / std::max(cfg_.sec_target_hz, 1.0e-6);
  sec_age_ += dt;
  if (quiescent_ || sec_age_ >= period) {
    sec_target_ = sec_lpf_;
    sec_age_ = 0.0;
  }
  if (dt > 1e-9) {
    const double wn = 2.0 * M_PI * 8.0;
    const double zeta = std::max(cfg_.sec_filter_zeta, 0.0);
    Vec8 j = wn * wn * (sec_target_ - sec_qdot_) - 2.0 * zeta * wn * sec_acc_;
    for (int i = 0; i < kNv; ++i) j[i] = clip(j[i], -j_max_[i], j_max_[i]);
    sec_acc_ += j * dt;
    sec_qdot_ += sec_acc_ * dt;
  }
  Vec8 sec_filt = sec_qdot_;
  if (cfg_.max_qdot_frac > 0.0) {
    Vec8 cap = kin_.v_max() * cfg_.max_qdot_frac;
    cap[0] = v_max_[0];
    for (int i = 0; i < kNv; ++i) sec_filt[i] = clip(sec_filt[i], -cap[i], cap[i]);
    sec_qdot_ = sec_filt;
  }

  const int sigma_period = std::max(cfg_.sigma_grad_period, 1);
  const bool sigma_activated_edge =
      !sigma_row_active_ && (sigma_arm < cfg_.sigma_activate);
  const bool refresh_sigma_grad =
      sigma_tick_ == 0 || sigma_activated_edge || (sigma_tick_ % sigma_period == 0);
  if (cfg_.sigma_enabled && refresh_sigma_grad) {
    const double eps = cfg_.sigma_grad_eps;
    sigma_grad_.setZero();
    for (int i = 1; i < kNv; ++i) {
      Vec8 qp = q_state;
      qp[i] += eps;
      kin_.update(qp);
      const double sp = kin_.sigma_arm();
      sigma_grad_[i] = (sp - sigma_arm) / eps;
    }
    kin_.update(q_state);
  }
  ++sigma_tick_;

  const bool has_qdot_ff = input_flags & kInHasQdotFf;
  const bool direct_active = direct_ptp_ && has_qdot_ff;
  const bool plan_rail = rail_only || tcp_fixed || plan_drives_rail_ || direct_active;
  double rail_pin = 0.0;
  bool has_pin = false;
  if (has_qdot_ff && plan_rail) {
    rail_pin = in.qdot_ff[0];
    has_pin = true;
  }
  Vec8 qdot_nom = sec_filt;
  Mat6x8 J_qp = J;
  Vec6 twist_qp = twist_base;
  if (has_qdot_ff) {
    if (direct_active) {
      qdot_nom = in.qdot_ff;
      if (rail_only) qdot_nom.tail<7>().setZero();
      J_qp.setZero();
      twist_qp.setZero();
    } else {
      Vec8 qdot_ff_sec = in.qdot_ff;
      if (plan_rail) qdot_ff_sec[0] = 0.0;
      qdot_nom += qdot_ff_sec;
    }
  }
  const bool lead_exempt = std::abs(q_prev[0] - q_state[0]) > cfg_.resync_err_rail_m;
  Vec8 qdot;
  Vec6 residual;
  double slack = 0.0;
  bool ok = solve_hqp(J_qp, twist_qp, q_state, q_prev, qdot_nom, rail_exec, has_rail_exec,
                      have_rail_vel ? rail_task_vel : 0.0, rail_task_w, locked_hold, dt, h1, h2,
                      rail_mode_ == kRailCoupled && has_travel && !locked_hold, rail_pin, has_pin,
                      lead_exempt, sigma_arm, direct_active, &qdot, &residual, &slack);
  Vec8 q_candidate = q_prev;
  if (ok) {
    q_candidate = q_prev + qdot * dt;
    if (!q_finite_in_limits(q_candidate, q_lo_, q_hi_)) {
      ok = false;
      failure_code_ = kFailureFinalCertificate;
      fallback_level_ = kFallbackStop;
    }
  }
  if (ok) {
    q_cmd_ = q_candidate;
    dq_prev_ = q_cmd_ - q_prev;
    have_dq_prev_ = true;
    qdot_prev2_ = qdot_prev_;
    qdot_seen_ = qdot;
    qdot_prev_ = qdot;
    last_slack_ = slack;
    last_tcp_est_ = J * qdot;
    u_mid_committed_ = u_mid_applied_;
  } else {
    qdot.setZero();
    residual = twist_qp;
    slack = residual.norm();
    last_tcp_est_.setZero();
  }

  const auto t1 = std::chrono::steady_clock::now();
  out.q_cmd = q_cmd_;
  out.qdot = qdot;
  out.v_feas = ok ? last_lock_v_ : Vec6::Zero();
  out.v_tcp = last_tcp_est_;
  out.residual = residual;
  out.slack = slack;
  out.e_qp = residual.norm();
  out.u_alloc = u_alloc_;
  out.u_mid = u_mid_;
  out.v_r_ref = v_r_ref_;
  out.psi = psi_cmd_;
  out.d_star = d_star_;
  out.d_pref = d_pref_;
  out.sigma_min = last_sigma_;
  out.sigma_arm = sigma_arm;
  out.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (int i = 1; i < kNv; ++i) {
    if (qdot[i] >= v_max_[i] - 1e-6 || qdot[i] <= -v_max_[i] + 1e-6) out.joint_limited = 1;
  }
  out.rail_limited = (std::abs(qdot[0]) >= v_max_[0] - 1e-6) ? 1 : 0;
  out.wall_active =
      (q_state[0] <= q_lo_[0] + cfg_.damper_band_rail || q_state[0] >= q_hi_[0] - cfg_.damper_band_rail)
          ? 1
          : 0;
  out.secondary_suppressed = (quiescent_ || slack_high) ? 1 : 0;
  out.ns_norm = sec_filt.norm();
  out.sat_scale = sat_scale_;
  out.sec_target_norm = sec_target_.norm();
  out.homotopy_s = homotopy_s_;
  out.psi_star = psi_star_;
  out.u_task_raw = u_task_raw_;
  out.u_task_feasible = u_task_feasible_;
  out.u_pi_raw = u_pi_raw_;
  out.u_mid_cmd = u_mid_cmd_;
  out.u_post_raw = u_post_raw_;
  out.u_post_feasible = u_post_feasible_;
  out.u_mid_applied = u_mid_applied_;
  out.d_star_dot_cmd = d_star_dot_cmd_;
  out.u_escape_raw = u_escape_raw_;
  out.u_escape_feasible = u_escape_feasible_;
  out.escape_active = escape_active_ ? 1.0 : 0.0;
  out.escape_dir = static_cast<double>(escape_dir_);
  out.u_base = u_base_;
  out.u_feasible = u_feasible_;
  out.v_r_lpf = v_r_lpf_;
  out.e_d = e_d_;
  out.V_d_proxy = V_d_proxy_;
  out.j4_design_slack = j4_design_slack_;
  out.sigma_slack = sigma_slack_;
  out.rail_box_lo = rail_box_lo_;
  out.rail_box_hi = rail_box_hi_;
  out.rail_bind_lo = rail_bind_lo_;
  out.rail_bind_hi = rail_bind_hi_;
  out.rail_task_vel_used = rail_task_vel_used_;
  out.rail_h1 = rail_h1_;
  out.rail_h2 = rail_h2_;
  out.rail_qdot_prev = rail_qdot_prev_tel_;
  out.rail_qdot_prev2 = rail_qdot_prev2_tel_;
  out.qp1_status = qp1_status_;
  out.qp2_status = qp2_status_;
  out.fallback_level = fallback_level_;
  out.failure_code = failure_code_;
  out.qp1_hard_violation = qp1_hard_violation_;
  out.final_hard_violation = final_hard_violation_;
  out.task_lock_violation = task_lock_violation_;
  out.final_box_violation = final_box_violation_;
  out.qp_overrun = qp_overrun_ ? 1u : 0u;
  out.posture_gate = posture_gate_scale_;
  {
    const Vec6 twist_rail = J.col(0) * rail_exec;
    const Vec6 twist_arm = J.rightCols<7>() * qdot.tail<7>();
    Eigen::Vector3d motion = twist_base.head<3>();
    if ((input_flags & kInHasVelFf) && in.vel_ff.head<3>().norm() > 1e-6) {
      motion = in.vel_ff.head<3>();
    }
    double n_dir = motion.norm();
    if (n_dir <= 1e-9) {
      motion = J.topLeftCorner<3, 1>();
      n_dir = motion.norm();
    }
    out.rail_motion_share = std::numeric_limits<double>::quiet_NaN();
    if (n_dir > 1e-9) {
      const Eigen::Vector3d u = motion / n_dir;
      const double rc = twist_rail.head<3>().dot(u);
      const double ac = twist_arm.head<3>().dot(u);
      const double den = std::abs(rc) + std::abs(ac);
      if (den > 1e-9) out.rail_motion_share = std::abs(rc) / den;
    }
  }
  if (out.joint_limited) out.flags |= kOutJointLimited;
  if (out.rail_limited) out.flags |= kOutRailLimited;
  if (out.wall_active) out.flags |= kOutWallActive;
  if (out.secondary_suppressed) out.flags |= kOutSecSuppressed;
  if (!ok) {
    out.flags |= kOutFailed;
    out.status = kStatusFail;
  } else {
    out.status = kStatusOk;
  }
  return out;
}

}  // namespace wbc_rt
