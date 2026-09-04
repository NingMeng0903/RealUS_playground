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
constexpr double kRailDriveCap = 0.40;
constexpr double kRailPrefW = 64.0;
constexpr double kQuietLinEnter = 0.005;
constexpr double kQuietRotEnter = 0.05;
constexpr double kQuietLinExit = 0.008;
constexpr double kQuietRotExit = 0.08;
constexpr double kQuietTcp = 0.010;
constexpr double kQuietHold = 0.15;

uint32_t qp_status_code(proxsuite::proxqp::QPSolverOutput s) {
  using S = proxsuite::proxqp::QPSolverOutput;
  if (s == S::PROXQP_SOLVED) return kQpSolved;
  if (s == S::PROXQP_MAX_ITER_REACHED) return kQpMaxIter;
  return kQpFailed;
}

bool qp_is_candidate(uint32_t code) {
  return code == kQpSolved || code == kQpMaxIter;
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
                    const VecX& hi,
                    const VecX* seed = nullptr) {
  using IG = proxsuite::proxqp::InitialGuessStatus;
  if (!*inited) {
    qp.init(H, g, A, b, C, lo, hi, true);
    *inited = true;
  } else {
    qp.settings.initial_guess =
        last_ok ? IG::WARM_START_WITH_PREVIOUS_RESULT : IG::NO_INITIAL_GUESS;
    qp.update(H, g, A, b, C, lo, hi, true);
  }
  if (seed != nullptr && seed->size() == H.rows()) {
    qp.settings.initial_guess = IG::WARM_START;
    qp.results.x = *seed;
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
  if (!cfg.collision_enabled || cfg.collision_urdf.empty()) return;
  std::vector<std::string> dirs;
  const auto slash = cfg.collision_urdf.find_last_of('/');
  if (slash != std::string::npos) dirs.push_back(cfg.collision_urdf.substr(0, slash));
  pinocchio::urdf::buildGeom(model, cfg.collision_urdf, pinocchio::COLLISION, geom_model_,
                             dirs, ::coal::MeshLoaderPtr());
  geom_model_.addAllCollisionPairs();
  if (!cfg.pair_config.empty()) {
    std::ifstream in(cfg.pair_config);
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
  std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.d < b.d; });
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
  v_max_[0] = std::min(cfg.rail_v_max, kRailDriveCap);
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
    try {
      collision_ = std::make_unique<Collision>(kin_.model(), cfg);
    } catch (...) {
      collision_.reset();
    }
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
  qdot_prev_tel_.setZero();
  qdot_prev2_tel_.setZero();
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
  leave_sign_ = 0.0;
  mid_integ_ = 0.0;
  slack_hold_latched_ = false;
  sec_qdot_.setZero();
  sec_acc_.setZero();
  sec_target_.setZero();
  sec_lpf_.setZero();
  gN_lpf_.setZero();
  gN_lpf_init_ = false;
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
  leave_sign_ = 0.0;
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
  ns_enter_t_ = 1e9;
  ns_homotopy_open_ = false;
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
  slack_hold_latched_ = false;
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
  const bool next_center = bits & kFlagCenterSuppress;
  if (center_suppress_ && !next_center) {
    ns_enter_t_ = 0.0;
    ns_homotopy_open_ = true;
  }
  center_suppress_ = next_center;
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
  const double d_live = kin_.tcp_xyz()[1] - q[0];
  d_pref_ = d_live;
  d_star_ref_ = d_live;
  d_star_ref_init_ = true;
  d_star_dot_cmd_ = 0.0;
  e_d_ = 0.0;
  last_e_mid_ = 0.0;
}

void InnerLoop::set_rail_ext_mode(int pose_attract) { rail_ext_mode_ = pose_attract; }

void InnerLoop::track_rail_authority(double d_live, double d_star_target, double v_applied,
                                     double dt) {
  const double target = std::isfinite(d_star_target) ? d_star_target : d_live;
  d_star_ref_ = target;
  d_star_ref_init_ = true;
  d_star_dot_cmd_ = 0.0;
  u_base_ = v_applied;
  u_feasible_ = v_applied;
  (void)d_live;
  if (v_r_init_ && dt > 1e-12) {
    double a = (v_applied - v_r_ref_) / dt;
    const double da = kRailRefJerk * dt;
    a = clip(a, v_r_a_ - da, v_r_a_ + da);
    a = clip(a, -cfg_.a_max_rail, cfg_.a_max_rail);
    v_r_a_ = a;
  } else {
    v_r_a_ = 0.0;
  }
  v_r_ref_ = v_applied;
  v_r_lpf_ = v_applied;
  v_r_init_ = true;
}

void InnerLoop::fill_mixer_out(TickOut* out) const {
  out->u_alloc = u_alloc_;
  out->u_mid = u_mid_;
  out->v_r_ref = v_r_ref_;
  out->d_star = d_star_;
  out->d_pref = d_pref_;
  out->u_task_raw = u_task_raw_;
  out->u_task_feasible = u_task_feasible_;
  out->u_pi_raw = u_pi_raw_;
  out->u_mid_cmd = u_mid_cmd_;
  out->u_post_raw = u_post_raw_;
  out->u_post_feasible = u_post_feasible_;
  out->u_mid_applied = u_mid_applied_;
  out->d_star_dot_cmd = d_star_dot_cmd_;
  out->u_escape_raw = u_escape_raw_;
  out->u_escape_feasible = u_escape_feasible_;
  out->escape_active = escape_active_ ? 1.0 : 0.0;
  out->escape_dir = static_cast<double>(escape_dir_);
  out->u_base = u_base_;
  out->u_feasible = u_feasible_;
  out->v_r_lpf = v_r_lpf_;
  out->e_d = e_d_;
  out->V_d_proxy = V_d_proxy_;
}

void InnerLoop::apply_velocity_box(const Vec8& q_geom, const Vec8& q_cmd, const Vec8& q_meas,
                                   double dt, double h1, double h2, bool rail_locked,
                                   double rail_pin, bool has_pin, bool lead_exempt,
                                   Vec8* lo, Vec8* hi) {
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
  if (cfg_.j4_design_enabled && cfg_.j4_design_hi > cfg_.j4_design_lo &&
      cfg_.j4_design_gamma > 0.0) {
    const int j4 = j4_index(kNv);
    (*lo)[j4] = std::max((*lo)[j4], -cfg_.j4_design_gamma * (q_geom[j4] - cfg_.j4_design_lo));
    (*hi)[j4] = std::min((*hi)[j4], cfg_.j4_design_gamma * (cfg_.j4_design_hi - q_geom[j4]));
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
    collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
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
    collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
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
    collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
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
      if (clo > chi) {
        if (lead >= 0.0) chi = clo;
        else clo = chi;
      }
      (*hi)[i] = chi;
      (*lo)[i] = clo;
    }
    note_rail_bind(olo, ohi, *lo, *hi, kRailBindLead);
  }
  {
    const double olo = (*lo)[0];
    const double ohi = (*hi)[0];
    collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
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
                          Vec8* qdot, Vec6* residual, double* slack) {
  const auto t_asm0 = std::chrono::steady_clock::now();
  qp1_status_ = kQpNotRun;
  qp2_status_ = kQpNotRun;
  qp1_iter_ = qp2_iter_ = 0;
  qp1_ms_ = qp2_ms_ = assembly_ms_ = fallback_ms_ = 0.0;
  Mat6x8 J_task = J;
  Vec6 b_task = v_cmd;
  if (has_rail_exec) {
    const Vec6 rail_contrib = J.col(0) * rail_exec;
    J_task.col(0).setZero();
    b_task = v_cmd - rail_contrib;
  }
  rail_h1_ = h1;
  rail_h2_ = h2;
  rail_qdot_prev_tel_ = qdot_prev_[0];
  rail_qdot_prev2_tel_ = qdot_prev2_[0];
  qdot_prev_tel_ = qdot_prev_;
  qdot_prev2_tel_ = qdot_prev2_;
  Vec8 lo_box, hi_box;
  apply_velocity_box(q_geom, q_prev, q_geom, dt, h1, h2, rail_locked, rail_pin, has_pin,
                     lead_exempt, &lo_box, &hi_box);
  {
    const double olo = lo_box[0];
    const double ohi = hi_box[0];
    tighten_branch(q_geom, rail_open, &lo_box, &hi_box);
    note_rail_bind(olo, ohi, lo_box, hi_box, kRailBindBranch);
  }
  {
    const double olo = lo_box[0];
    const double ohi = hi_box[0];
    collapse_interval(&lo_box, &hi_box, &qdot_prev_, &a_max_, dt);
    note_rail_bind(olo, ohi, lo_box, hi_box, kRailBindCollapse);
  }
  if (std::isfinite(rail_task_vel)) {
    rail_task_vel = clip(rail_task_vel, lo_box[0], hi_box[0]);
  }
  rail_task_vel_used_ = rail_task_vel;
  rail_box_lo_ = lo_box[0];
  rail_box_hi_ = hi_box[0];

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
    const bool flo = std::isfinite(lo1[i]);
    const bool fhi = std::isfinite(hi1[i]);
    if (flo && fhi && (hi1[i] - lo1[i]) > 2 * inset) {
      lo1[i] += inset;
      hi1[i] -= inset;
    } else if (flo && !fhi) {
      lo1[i] += inset;
    } else if (!flo && fhi) {
      hi1[i] -= inset;
    }
  }

  last_lo_box_ = lo_box;
  last_hi_box_ = hi_box;
  n_cbf_active_ = 0;
  if (collision_) {
    // Count CBF rows that have a finite lower bound (active collision).
    for (int i = 0; i < kMaxCbf; ++i) {
      if (std::isfinite(lo[kNv + i]) && lo[kNv + i] > -1e19) ++n_cbf_active_;
    }
  }

  const auto t_qp1_0 = std::chrono::steady_clock::now();
  assembly_ms_ = std::chrono::duration<double, std::milli>(t_qp1_0 - t_asm0).count();
  solve_dense_qp(*qp1_, &qp1_inited_, qp1_last_ok_, H1, g1, A1, b_task, C, lo1, hi1);
  const auto t_qp1_1 = std::chrono::steady_clock::now();
  qp1_ms_ = std::chrono::duration<double, std::milli>(t_qp1_1 - t_qp1_0).count();
  qp1_status_ = qp_status_code(qp1_->results.info.status);
  qp1_iter_ = static_cast<uint32_t>(qp1_->results.info.iter);
  const bool qp1_ok = qp_is_candidate(qp1_status_);
  qp1_last_ok_ = qp1_ok;
  if (!qp1_ok) {
    qp2_status_ = kQpNotRun;
    qp2_iter_ = 0;
    qp2_ms_ = 0.0;
    *qdot = Vec8::Zero();
    *residual = b_task;
    *slack = residual->norm();
    return false;
  }
  VecX x1 = qp1_->results.x;
  Vec8 qdot1 = x1.head<kNv>();
  if (has_rail_exec) {
    double seed = rail_exec;
    if (std::isfinite(rail_task_vel) && !rail_locked) seed = rail_task_vel;
    qdot1[0] = clip(seed, lo_box[0], hi_box[0]);
    x1[0] = qdot1[0];
  }
  {
    double excess = 0.0;
    uint32_t deg = 0, inf = 0;
    bool subst = false;
    measure_qdot_box(qdot1, lo_box, hi_box, &excess, &deg, &inf, &subst);
    if (subst) {
      qp2_status_ = kQpNotRun;
      qp2_iter_ = 0;
      qp2_ms_ = 0.0;
      *qdot = qdot1;
      *residual = b_task - J_task * qdot1;
      *slack = residual->norm();
      return false;
    }
  }
  const Vec6 t1 = J_task * qdot1;
  last_lock_J_ = has_rail_exec ? J_task : J;
  last_lock_v_ = t1;

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
  if (x1.size() >= kNv + kNTaskSlack) {
    x2_seed.segment<kNTaskSlack>(kNv) = x1.segment<kNTaskSlack>(kNv);
  }
  for (int k = 0; k < pref_n; ++k) {
    const int col = kNv + kNTaskSlack + pref_s[k];
    const double base = pref_J.row(k).dot(qdot1);
    const double need = std::max(pref_lo[k] - base, 0.0);
    x2_seed[col] = std::max(x2_seed[col], need + inset);
  }
  const auto t_qp2_0 = std::chrono::steady_clock::now();
  solve_dense_qp(*qp2_, &qp2_inited_, qp2_last_ok_, H2, g2, A2, last_lock_v_, C, lo, hi,
                 &x2_seed);
  const auto t_qp2_1 = std::chrono::steady_clock::now();
  qp2_ms_ = std::chrono::duration<double, std::milli>(t_qp2_1 - t_qp2_0).count();
  qp2_status_ = qp_status_code(qp2_->results.info.status);
  qp2_iter_ = static_cast<uint32_t>(qp2_->results.info.iter);
  const bool qp2_ok = qp_is_candidate(qp2_status_);
  qp2_last_ok_ = qp2_ok;
  Vec8 qdot_out = qdot1;
  VecX x_pub = x1;
  if (qp2_ok) {
    Vec8 qdot2 = qp2_->results.x.head<kNv>();
    double excess = 0.0;
    uint32_t deg = 0, inf = 0;
    bool subst = false;
    measure_qdot_box(qdot2, lo_box, hi_box, &excess, &deg, &inf, &subst);
    if (!subst) {
      qdot_out = qdot2;
      x_pub = qp2_->results.x;
    } else {
      // Rail excess > 1e-6: discard QP2 and publish QP1's x* (qdot1).
      // qdot1[0] is already clip(v_r_ref, box).  Do not splice axes.
      qp2_status_ = kQpFailed;
    }
  }
  {
    double excess = 0.0;
    uint32_t deg = 0, inf = 0;
    bool subst = false;
    measure_qdot_box(qdot_out, lo_box, hi_box, &excess, &deg, &inf, &subst);
    if (subst) {
      *qdot = qdot1;
      *residual = b_task - J_task * qdot1;
      *slack = residual->norm();
      last_C_ = C;
      last_lo_ = lo;
      last_hi_ = hi;
      last_lock_v_ = last_lock_J_ * qdot1;
      return false;
    }
  }
  j4_design_slack_ = 0.0;
  sigma_slack_ = 0.0;
  if (x_pub.size() > kNv + kNTaskSlack) {
    sigma_slack_ = std::max(0.0, x_pub[kNv + kNTaskSlack + 0]);
  }
  if (x_pub.size() > kNv + kNTaskSlack + 2) {
    j4_design_slack_ = std::max(0.0, x_pub[kNv + kNTaskSlack + 2]);
  }
  *qdot = qdot_out;
  *residual = b_task - J_task * qdot_out;
  *slack = residual->norm();
  last_C_ = C;
  last_lo_ = lo;
  last_hi_ = hi;
  last_lock_v_ = last_lock_J_ * qdot_out;
  last_qdot_qp_ = qdot_out;
  return true;
}

TickOut InnerLoop::step(const TickIn& in) {
  const auto t0 = std::chrono::steady_clock::now();
  TickOut out;
  Vec6 twist = in.v_cmd;
  if (!enabled_ || (in.flags & kInStale)) {
    twist.setZero();
    out.flags |= kOutStale;
  }
  const double dt_nom = (std::isfinite(in.dt_nom) && in.dt_nom > 0.0) ? in.dt_nom : cfg_.dt;
  const double dt = dt_nom;
  if (in.flags & kInSeedQcmd) {
    q_cmd_ = in.q_meas;
    q_hat_ = in.q_meas[0];
    if (in.flags & kInHasQdotFf) {
      qdot_prev_ = in.qdot_ff;
      qdot_seen_ = in.qdot_ff;
      dq_prev_ = in.qdot_ff * dt;
      have_dq_prev_ = true;
    }
  }
  Vec8 q_prev = q_cmd_;
  const bool skip_rail_rebase = direct_ptp_ && plan_drives_rail_;
  if (rail_mode_ == kRailCoupled && obs_init_ && last_sample_t_ >= 0.0 &&
      !skip_rail_rebase) {
    q_prev[0] = q_hat_;
    q_cmd_[0] = q_hat_;
  }
  const Vec8 q_state = in.q_meas;
  kin_.update(q_state);
  const Mat6x8 J = kin_.jacobian();
  last_sigma_ = kin_.sigma_min();
  const double sigma_arm = kin_.sigma_arm();
  const Eigen::Matrix3d R = kin_.tcp_R();
  const Vec6 twist_base = twist_to_base(twist, R, cfg_.control_frame == "tool");
  out.v_recv = twist_base;

  const bool locked_hold = rail_mode_ == kRailLocked && locked_style_ == kStyleHold;
  const bool rail_only = rail_mode_ == kRailLocked && locked_style_ == kStyleRailOnly;
  const bool tcp_fixed = rail_mode_ == kRailLocked && locked_style_ == kStyleTcpFixed;

  qdot_prev2_ = qdot_seen_;
  qdot_seen_ = qdot_prev_;

  const double now = in.t_mono > 0.0 ? in.t_mono : 0.0;
  if (!obs_init_) {
    q_hat_ = q_state[0];
    v_hat_ = 0.0;
    obs_init_ = true;
  } else {
    const double v_pred = (in.flags & kInHasRailV) ? in.rail_v : v_hat_;
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

  if (direct_ptp_ && (in.flags & kInHasQdotFf)) {
    Vec8 qdot = in.qdot_ff;
    for (int i = 0; i < kNv; ++i) qdot[i] = clip(qdot[i], -v_max_[i], v_max_[i]);
    if (rail_only) qdot.tail<7>().setZero();
    if (locked_hold) qdot[0] = 0.0;
    q_cmd_ = q_prev + qdot * dt;
    if (locked_hold && cfg_.lock_hard_pin && has_rail_ref_) {
      q_cmd_[0] = rail_q_ref_;
      qdot[0] = 0.0;
    }
    qdot_prev_ = qdot;
    if (cfg_.psi_enabled && rail_mode_ == kRailCoupled) {
      const Vec6 pose = kin_.fk_pose_at(q_cmd_);
      posture_.follow_live(q_cmd_, pose, dt);
      d_star_ = posture_.d_star();
      psi_cmd_ = posture_.psi_cmd();
      psi_star_ = posture_.psi_star();
      homotopy_s_ = posture_.homotopy_s();
      d_pref_ = d_star_;
    }
    {
      const Vec6 pose_cmd = kin_.fk_pose_at(q_cmd_);
      const double d_live = pose_cmd[1] - q_cmd_[0];
      const double d_target = std::isfinite(d_star_) ? d_star_ : d_live;
      track_rail_authority(d_live, d_target, qdot[0], dt);
    }
    out.q_cmd = q_cmd_;
    out.qdot = qdot;
    out.sigma_min = last_sigma_;
    out.sigma_arm = sigma_arm;
    out.homotopy_s = homotopy_s_;
    out.psi = psi_cmd_;
    fill_mixer_out(&out);
    return out;
  }

  double rail_exec = qdot_prev_[0];
  bool has_rail_exec = false;
  if (in.flags & kInHasRailV) {
    rail_exec = in.rail_v;
    has_rail_exec = true;
  } else if (obs_init_ && last_sample_t_ >= 0.0) {
    rail_exec = v_hat_;
    has_rail_exec = true;
  }

  double h1 = dt_nom;
  double h2 = std::numeric_limits<double>::quiet_NaN();
  if (!box_t_init_) {
    box_t_init_ = true;
    box_last_t_ = now;
    box_h1_ = dt_nom;
  } else {
    h2 = dt_nom;
    box_last_t_ = now;
    box_h1_ = dt_nom;
    h1 = dt_nom;
  }

  const double z_now = kin_.tcp_xyz()[2];
  const double y_tcp = kin_.tcp_xyz()[1];
  double y_tcp_d = y_tcp;
  bool has_pose_d = in.flags & kInHasPoseD;
  if (has_pose_d && std::isfinite(in.pose_d[1])) y_tcp_d = in.pose_d[1];
  const double tool_y_err = y_tcp_d - y_tcp;

  const bool contact = in.flags & kInContact;
  const bool has_vf = in.flags & kInHasVForce;
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
  secondary_alpha_ = raised_cosine_alpha(last_slack_, cfg_.slack_exit, cfg_.slack_enter,
                                         last_sigma_, cfg_.sigma_fade_ref);
  {
    const double enter = std::max(0.0, cfg_.ns_enter_fade_s);
    if (ns_enter_t_ < enter) ns_enter_t_ = std::min(ns_enter_t_ + dt, enter);
  }
  hold_d_prev_ = quiescent_;

  if (cfg_.psi_enabled && rail_mode_ == kRailCoupled) {
    const Vec6 pose = kin_.fk_pose_at(q_prev);
    posture_.step(q_prev, pose, dt, q_lo_[0], q_hi_[0], false, 1.0);
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
  bool have_rail_vel = false;
  last_v_escape_ = 0.0;
  last_e_mid_ = 0.0;
  if (cfg_.rail_ext_enabled && rail_ext_active_ && rail_mode_ == kRailCoupled) {
    const double y = q_state[0];
    // e_mid = (y_tcp − d*) − y_rail. SERVO_TWIST latches pose_d at set_origin;
    // using that Y pulls the rail back to the start instead of tracking d*.
    const double y_des = y_tcp;
    const double rail_ff = y_des - d_pref_;
    const double err_raw = rail_ff - y;
    double band = planned_ ? 0.0 : cfg_.d_band;
    const double err = err_raw - clip(err_raw, -band, band);
    last_e_mid_ = err;
    double v_ff = 0.0;
    if (in.flags & kInHasVelFf) {
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
    const bool hold_ref = !std::isfinite(d_star_);
    if (!d_star_ref_init_ || !std::isfinite(d_star_ref_)) {
      d_star_ref_ = d_live;
      d_star_ref_init_ = true;
      d_star_dot_cmd_ = 0.0;
    } else if (hold_ref || dt <= 1e-12) {
      d_star_dot_cmd_ = 0.0;
    } else {
      const double lim = std::abs(cfg_.d_center_rate) * dt;
      const double err = d_star_ - d_star_ref_;
      const double delta = (lim > 1e-15) ? lim * std::tanh(err / lim) : 0.0;
      d_star_ref_ += delta;
      d_star_dot_cmd_ = delta / dt;
    }
    d_pref_ = d_star_ref_;
    e_d_ = d_live - d_star_ref_;
    last_e_mid_ = e_d_;
    V_d_proxy_ = 0.5 * cfg_.kp_mid * e_d_ * e_d_;

    const double lam = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                         cfg_.sr_sigma_floor);
    Vec8 mw = margin_weight_from_activation(q_prev, q_mid_, half_, cfg_.k_margin,
                                           cfg_.ns_activation);
    auto [u_a, qall] = allocate_rail(J, twist_base, v_max_, mw, lam, cfg_.v0, cfg_.w0,
                                     last_e_mid_, cfg_.k_err_rail, cfg_.e_ref);
    if (cfg_.j4_design_enabled && cfg_.j4_design_hi > cfg_.j4_design_lo) {
      const int j4 = j4_index(kNv);
      const double box_mw = margin_weight_toward_box(
          q_prev[j4], cfg_.j4_design_lo, cfg_.j4_design_hi, qall[j4], cfg_.k_margin);
      if (box_mw > mw[j4]) {
        mw[j4] = box_mw;
        auto again = allocate_rail(J, twist_base, v_max_, mw, lam, cfg_.v0, cfg_.w0,
                                   last_e_mid_, cfg_.k_err_rail, cfg_.e_ref);
        u_a = again.first;
        qall = again.second;
      }
    }
    (void)qall;
    u_alloc_ = u_a;
    u_task_raw_ = u_a;
    u_escape_raw_ = last_v_escape_;
    double leave_raw =
        wall_leave_only_sign(q_state[0], q_lo_[0], q_hi_[0], cfg_.damper_band_rail);
    if (planned_ && q_state[0] >= cfg_.soft_max - cfg_.escape_leave) {
      leave_raw = std::max(leave_raw, 1.0);
    }
    const double leave = update_leave_sign(leave_raw, q_state[0], q_lo_[0], q_hi_[0],
                                           cfg_.damper_band_rail, cfg_.leave_exit_eps,
                                           leave_sign_);
    leave_sign_ = leave;
    wall_pi_frozen_ = (leave != 0.0);
    escape_dir_ = update_escape_dir(escape_active_, u_escape_raw_, escape_dir_);
    const int guard_dir = escape_active_ ? escape_dir_ : 0;
    double u_lo = 0.0, u_hi = 0.0;
    if (leave * u_task_raw_ > 1.0e-4) {
      u_lo = 0.0;
      u_hi = 0.0;
    } else {
      wall_velocity_bounds(v_max_[0], leave, &u_lo, &u_hi);
    }

    const double alpha = secondary_alpha_;
    u_pi_raw_ = cfg_.kp_mid * e_d_ + mid_integ_;
    u_mid_cmd_ = soft_saturate(u_pi_raw_, cfg_.u_mid_max);
    u_post_raw_ = alpha * (u_mid_cmd_ - d_star_dot_cmd_);
    u_mid_ = u_mid_cmd_;
    const RailShares shares = allocate_rail_shares(
        u_task_raw_,
        u_post_raw_,
        u_escape_raw_,
        guard_dir,
        u_lo, u_hi);
    u_task_feasible_ = shares.u_task_feasible;
    u_escape_feasible_ = shares.u_escape_feasible;
    u_base_ = shares.u_base;
    u_post_feasible_ = shares.u_post_feasible;
    u_feasible_ = shares.u_feasible;
    u_mid_applied_ = u_post_feasible_ + d_star_dot_cmd_;
    if (!wall_pi_frozen_ && dt > 0.0) {
      if (alpha < 1.0e-6) {
        mid_integ_ = -cfg_.kp_mid * e_d_;
      } else {
        mid_integ_ += (cfg_.ki_mid * e_d_ + cfg_.kaw_mid * (u_mid_applied_ - u_pi_raw_)) * dt;
        mid_integ_ = (1.0 - alpha) * (-cfg_.kp_mid * e_d_) + alpha * mid_integ_;
      }
    }

    auto [a_mir, j_mir] = arm_mirror_rail_limits(J, a_max_, j_max_, cfg_.rho_a, cfg_.rho_j);
    const double tau = lpf_tau_from_fc(cfg_.f_c_hz);
    double v_f = u_feasible_;
    if (v_r_init_ && tau > 1e-9) v_f = first_order_lpf(v_r_ref_, u_feasible_, dt, tau);
    v_f = project_lpf_into_wall(v_f, leave);
    v_r_ref_ = project_lpf_into_wall(v_r_ref_, leave);
    v_r_init_ = true;
    v_r_lpf_ = v_f;
    double a_raw = (v_f - v_r_ref_) / dt;
    const double a_lim = std::min(cfg_.a_max_rail, a_mir);
    const double j_lim = std::min(kRailRefJerk, j_mir);
    double a = clip(a_raw, v_r_a_ - j_lim * dt, v_r_a_ + j_lim * dt);
    a = clip(a, -a_lim, a_lim);
    double v = clip(v_r_ref_ + a * dt, -v_max_[0], v_max_[0]);
    double lo_c, hi_c;
    wall_cap(q_state[0], cfg_.hard_min, cfg_.hard_max, a_lim, cfg_.rail_reaction_s, &lo_c, &hi_c);
    v = clip(v, lo_c, hi_c);
    v_r_ref_ = v;
    v_r_a_ = (v - (v_r_ref_ - a * dt)) / dt;
    rail_task_vel = v;
    have_rail_vel = true;
    rail_task_w = std::max(rail_task_w, kRailPrefW);
  } else {
    u_alloc_ = 0.0;
    u_mid_ = 0.0;
    u_feasible_ = 0.0;
    u_mid_applied_ = 0.0;
    u_post_raw_ = 0.0;
    u_post_feasible_ = 0.0;
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
  {
    const double period = cfg_.ns_enter_fade_s;
    if (period > 1e-9 && ns_enter_t_ < period) {
      const double u = clip(ns_enter_t_ / period, 0.0, 1.0);
      sec.tail<7>() *= smoothstep01(u);
    }
  }
  sec[0] = 0.0;
  out.ns_centering = qdot_center.norm();
  out.ns_manip = qdot_mu.norm();
  out.ns_arm_angle = qdot_arm.norm();
  out.ns_damping = qdot_damp.norm();
  out.ns_rail_lock = locked_hold ? std::abs(qdot_center[0] - qdot_damp[0]) : 0.0;
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

  {
    const double lam_ns = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                            cfg_.sr_sigma_floor);
    sec_filt = project_nullspace(J, sec_filt, lam_ns);
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

  const bool plan_rail = rail_only || tcp_fixed || plan_drives_rail_;
  double rail_pin = 0.0;
  bool has_pin = false;
  if ((in.flags & kInHasQdotFf) && plan_rail) {
    rail_pin = in.qdot_ff[0];
    has_pin = true;
  }
  const bool lead_exempt = std::abs(q_prev[0] - q_state[0]) > cfg_.resync_err_rail_m;
  Vec8 qdot;
  Vec6 residual;
  double slack = 0.0;
  const auto t_fb0 = std::chrono::steady_clock::now();
  const bool ok = solve_hqp(J, twist_base, q_state, q_prev, sec_filt, rail_exec, has_rail_exec,
                            have_rail_vel ? rail_task_vel : 0.0, rail_task_w, locked_hold, dt, h1,
                            h2, rail_mode_ == kRailCoupled && has_travel && !locked_hold, rail_pin,
                            has_pin, lead_exempt, sigma_arm, &qdot, &residual, &slack);
  const Vec8 qdot_qp = qdot;
  bool published_ok = ok;
  if (!ok) {
    qdot = inbox_brake(qdot_prev_, last_lo_box_, last_hi_box_, a_max_, h1);
    fallback_ms_ = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_fb0)
                       .count();
    published_ok = false;
  } else {
    fallback_ms_ = 0.0;
  }
  last_qdot_qp_ = qdot_qp;
  q_cmd_ = q_prev + qdot * dt;
  q_cmd_[0] = clip(q_cmd_[0], q_lo_[0], q_hi_[0]);
  if (q_cmd_[0] <= q_lo_[0] + 1e-4 && qdot[0] < 0.0) {
    q_cmd_[0] = q_lo_[0];
    qdot[0] = 0.0;
  } else if (q_cmd_[0] >= q_hi_[0] - 1e-4 && qdot[0] > 0.0) {
    q_cmd_[0] = q_hi_[0];
    qdot[0] = 0.0;
  }
  if (locked_hold && cfg_.lock_hard_pin && has_rail_ref_) {
    q_cmd_[0] = rail_q_ref_;
    qdot[0] = 0.0;
  }
  if (plan_rail && (in.flags & kInHasQdotFf)) {
    q_cmd_[0] = clip(q_prev[0] + in.qdot_ff[0] * dt, q_lo_[0], q_hi_[0]);
    qdot[0] = (q_cmd_[0] - q_prev[0]) / dt;
    if (rail_only) {
      q_cmd_.tail<7>() = q_prev.tail<7>();
      qdot.tail<7>().setZero();
    }
  }
  Vec8 q_shadow, dq_s;
  bool would = false;
  clamp_command_step(q_prev, q_cmd_, have_dq_prev_ ? &dq_prev_ : nullptr, a_max_, dt, &q_shadow,
                     &dq_s, &would);
  if (would) {
    const Vec8 qdot_s = (q_shadow - q_prev) / dt;
    const Vec6 lock_err = last_lock_J_ * qdot_s - last_lock_v_;
    if (lock_err.norm() <= std::max(10.0 * cfg_.eps_abs, 1e-5)) {
      q_cmd_ = q_shadow;
      qdot = qdot_s;
    }
  }
  dq_prev_ = q_cmd_ - q_prev;
  have_dq_prev_ = true;
  qdot_prev_ = qdot;
  last_slack_ = slack;
  last_tcp_est_ = J * qdot;
  u_mid_committed_ = u_mid_applied_;
  const bool mixer_owned =
      published_ok && rail_mode_ == kRailCoupled && !locked_hold && !plan_rail;
  if (!mixer_owned) {
    const Vec6 pose_cmd = kin_.fk_pose_at(q_cmd_);
    const double d_live = pose_cmd[1] - q_cmd_[0];
    const double d_target = std::isfinite(d_star_) ? d_star_ : d_live;
    track_rail_authority(d_live, d_target, qdot[0], dt);
  }

  const auto t1 = std::chrono::steady_clock::now();
  out.q_cmd = q_cmd_;
  out.qdot = qdot;
  out.v_feas = last_lock_v_;
  out.v_tcp = last_tcp_est_;
  out.residual = residual;
  out.slack = slack;
  out.e_qp = residual.norm();
  out.psi = psi_cmd_;
  out.sigma_min = last_sigma_;
  out.sigma_arm = sigma_arm;
  out.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  fill_mixer_out(&out);
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
  out.qdot_prev_used = qdot_prev_tel_;
  out.qdot_prev2_used = qdot_prev2_tel_;
  out.qp1_status = qp1_status_;
  out.qp2_status = qp2_status_;
  out.qp1_iter = qp1_iter_;
  out.qp2_iter = qp2_iter_;
  out.n_cbf_active = n_cbf_active_;
  out.qp1_solve_ms = qp1_ms_;
  out.qp2_solve_ms = qp2_ms_;
  out.assembly_ms = assembly_ms_;
  out.fallback_ms = fallback_ms_;
  out.rail_exec = rail_exec;
  out.follow_err_rad = (q_cmd_.tail<7>() - q_state.tail<7>()).norm();
  out.qdot_qp_vs_sent_max = (last_qdot_qp_ - qdot).cwiseAbs().maxCoeff();
  out.dual_cancel = dual_cancel_frac(u_task_feasible_, u_post_feasible_);
  out.secondary_alpha = secondary_alpha_;
  out.manip_active = manip_active_ ? 1u : 0u;
  {
    double excess = 0.0;
    uint32_t deg = 0, inf = 0;
    bool subst = false;
    measure_qdot_box(qdot, last_lo_box_, last_hi_box_, &excess, &deg, &inf, &subst);
    out.box_excess_max = excess;
    out.box_degenerate = deg;
    out.box_infeasible = inf;
    out.box_lo = last_lo_box_;
    out.box_hi = last_hi_box_;
    (void)subst;
  }
  out.hard_residual_max = out.box_excess_max;
  out.equality_residual_max = residual.cwiseAbs().maxCoeff();
  {
    const Vec6 twist_rail = J.col(0) * rail_exec;
    const Vec6 twist_arm = J.rightCols<7>() * qdot.tail<7>();
    Eigen::Vector3d motion = twist_base.head<3>();
    if ((in.flags & kInHasVelFf) && in.vel_ff.head<3>().norm() > 1e-6) {
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
  out.status = published_ok ? kStatusOk : kStatusFail;
  return out;
}

}  // namespace wbc_rt
