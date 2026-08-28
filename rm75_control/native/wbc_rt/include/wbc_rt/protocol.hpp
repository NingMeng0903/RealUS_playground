#pragma once

#include <cstdint>
#include <cstring>

namespace wbc_rt {

static constexpr uint32_t kMagic = 0x57424331u;  // 'WBC1'
static constexpr uint32_t kVersion = 6;

enum Cmd : uint32_t {
  kCmdNone = 0,
  kCmdStep = 1,
  kCmdEnable = 2,
  kCmdStop = 3,
  kCmdReset = 4,
  kCmdBeginHybrid = 5,
  kCmdSetRailMode = 6,
  kCmdSetFlags = 7,
  kCmdPlanStroke = 8,
  kCmdSetStroke = 9,
  kCmdSetRailPoseTarget = 10,
  kCmdCaptureRailExtRef = 11,
  kCmdSetRailExtMode = 12,
  kCmdShutdown = 13,
};

enum InFlag : uint32_t {
  kInContact = 1u << 0,
  kInStale = 1u << 1,
  kInHasQdotFf = 1u << 2,
  kInHasPoseD = 1u << 3,
  kInHasVelFf = 1u << 4,
  kInHasRailV = 1u << 5,
  kInHasVForce = 1u << 6,
  kInHasPathTwist = 1u << 7,
  kInHasFeedbackTwist = 1u << 8,
  kInSeedQcmd = 1u << 9,
  kInHasPosture = 1u << 10,
  kInHasQStar = 1u << 11,
};

enum CtrlFlag : uint32_t {
  kFlagPlanDrivesRail = 1u << 0,
  kFlagDirectPtp = 1u << 1,
  kFlagArmSuppress = 1u << 2,
  kFlagCenterSuppress = 1u << 3,
  kFlagManipActive = 1u << 4,
  kFlagRailExtActive = 1u << 5,
};

enum OutFlag : uint32_t {
  kOutJointLimited = 1u << 0,
  kOutRailLimited = 1u << 1,
  kOutWallActive = 1u << 2,
  kOutSecSuppressed = 1u << 3,
  kOutStale = 1u << 4,
  kOutReady = 1u << 5,
  kOutFailed = 1u << 6,
  kOutSecondarySuppressed = 1u << 3,
};

enum Status : uint32_t {
  kStatusBoot = 0,
  kStatusReady = 1,
  kStatusOk = 2,
  kStatusFail = 3,
  kStatusShutdown = 4,
};

enum QpStatusU : uint32_t {
  kQpNotRun = 0,
  kQpSolved = 1,
  kQpMaxIter = 2,
  kQpFailed = 3,
};

enum RailModeU : uint32_t {
  kRailCoupled = 0,
  kRailLocked = 1,
};

enum LockedStyleU : uint32_t {
  kStyleHold = 0,
  kStyleRailOnly = 1,
  kStyleTcpFixed = 2,
};

#pragma pack(push, 1)
struct WbcIn {
  uint32_t magic;
  uint32_t version;
  uint64_t seq;
  uint64_t cmd_seq;
  uint32_t cmd;
  uint32_t flags;
  double t_mono;
  double dt_wall;
  double dt_nom;
  double v_cmd[6];
  double q_meas[8];
  double rail_q;
  double rail_v;
  double v_force_z;
  double pose_d[6];
  double vel_ff[6];
  double qdot_ff[8];
  double path_twist[6];
  double feedback_twist[6];
  double cmd_f[16];
  uint32_t cmd_u[8];
};

struct WbcOut {
  uint32_t magic;
  uint32_t version;
  uint64_t seq;
  uint64_t cmd_ack;
  uint32_t status;
  uint32_t flags;
  double q_cmd[8];
  double qdot[8];
  double v_cmd_received[6];
  double v_cmd_feasible[6];
  double v_tcp_estimated[6];
  double task_residual[6];
  double slack;
  double e_qp;
  double u_alloc;
  double u_mid;
  double v_r_ref;
  double psi;
  double d_star;
  double d_pref;
  double solve_ms;
  double sigma_min;
  double sigma_arm;
  double cmd_f[8];
  uint32_t joint_limited;
  uint32_t rail_limited;
  uint32_t wall_active;
  uint32_t secondary_suppressed;
  double ns_norm;
  double ns_centering;
  double ns_manip;
  double ns_arm_angle;
  double ns_damping;
  double ns_rail_lock;
  double sat_scale;
  double sec_target_norm;
  double homotopy_s;
  double psi_star;
  double rail_motion_share;
  double u_task_raw;
  double u_task_feasible;
  double u_pi_raw;
  double u_mid_cmd;
  double u_post_raw;
  double u_post_feasible;
  double u_mid_applied;
  double d_star_dot_cmd;
  double u_escape_raw;
  double u_escape_feasible;
  double escape_active;
  double escape_dir;
  double u_base;
  double u_feasible;
  double v_r_lpf;
  double e_d;
  double V_d_proxy;
  double j4_design_slack;
  double sigma_slack;
  double rail_box_lo;
  double rail_box_hi;
  uint32_t rail_bind_lo;
  uint32_t rail_bind_hi;
  double rail_task_vel_used;
  double rail_h1;
  double rail_h2;
  double rail_qdot_prev;
  double rail_qdot_prev2;
  uint32_t qp1_status;
  uint32_t qp2_status;
  uint32_t qp1_iter;
  uint32_t qp2_iter;
  uint32_t n_cbf_active;
  uint32_t box_degenerate;
  uint32_t box_infeasible;
  uint32_t manip_active;
  double qp1_solve_ms;
  double qp2_solve_ms;
  double assembly_ms;
  double fallback_ms;
  double hard_residual_max;
  double equality_residual_max;
  double rail_exec;
  double box_excess_max;
  double follow_err_rad;
  double qdot_qp_vs_sent_max;
  double dual_cancel;
  double secondary_alpha;
  double box_lo[8];
  double box_hi[8];
  double qdot_prev[8];
  double qdot_prev2[8];
};
#pragma pack(pop)

static_assert(sizeof(WbcIn) == 608, "WbcIn layout drift");
static_assert(sizeof(WbcOut) == 1208, "WbcOut layout drift");

inline void clear_in(WbcIn* s) {
  std::memset(s, 0, sizeof(WbcIn));
  s->magic = kMagic;
  s->version = kVersion;
}

inline void clear_out(WbcOut* s) {
  std::memset(s, 0, sizeof(WbcOut));
  s->magic = kMagic;
  s->version = kVersion;
}

}  // namespace wbc_rt
