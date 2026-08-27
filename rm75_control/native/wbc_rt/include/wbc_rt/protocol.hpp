#pragma once

#include <cstdint>
#include <cstring>
#include <cstddef>
#include <atomic>

namespace wbc_rt {

static constexpr uint32_t kMagic = 0x57424331u;  // 'WBC1'
static constexpr uint32_t kVersion = 5;

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

enum QpStatus : uint32_t {
  kQpNotRun = 0,
  kQpSolved = 1,
  kQpMaxIter = 2,
  kQpPrimalInfeasible = 3,
  kQpDualInfeasible = 4,
  kQpClosestPrimalFeasible = 5,
  kQpNonfinite = 6,
  kQpCertificateFailed = 7,
  kQpOverrun = 8,
  kQpException = 9,
};

enum FallbackLevel : uint32_t {
  kFallbackNone = 0,
  kFallbackQp1 = 1,
  kFallbackStop = 2,
};

enum FailureCode : uint32_t {
  kFailureNone = 0,
  kFailureInputNonfinite = 1,
  kFailureBoxInfeasible = 2,
  kFailureQp1Status = 3,
  kFailureQp1Certificate = 4,
  kFailureQp2Status = 5,
  kFailureQp2Certificate = 6,
  kFailureSolveOverrun = 7,
  kFailureFinalCertificate = 8,
  kFailureInputStale = 9,
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
  uint64_t generation;
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
  uint64_t generation;
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
  uint32_t fallback_level;
  uint32_t failure_code;
  double qp1_hard_violation;
  double final_hard_violation;
  double task_lock_violation;
  double final_box_violation;
  uint32_t qp_overrun;
  uint32_t reserved_status;
  double posture_gate;
};
#pragma pack(pop)

static_assert(sizeof(WbcIn) == 616, "WbcIn layout drift");
static_assert(sizeof(WbcOut) == 896, "WbcOut layout drift");

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

// Shared-memory records use a generation seqlock.  Writers publish an odd
// generation while mutating a record and finish with the next even value.
// Readers copy only a stable even generation, preventing torn Eigen inputs or
// telemetry from crossing the process boundary.
template <typename T>
inline bool read_snapshot(const T* shared, T* local) {
  const auto* gen = &shared->generation;
  const uint64_t g0 = __atomic_load_n(gen, __ATOMIC_ACQUIRE);
  if (g0 & 1u) return false;
  std::memcpy(local, shared, sizeof(T));
  std::atomic_thread_fence(std::memory_order_acquire);
  const uint64_t g1 = __atomic_load_n(gen, __ATOMIC_ACQUIRE);
  return g0 == g1 && !(g1 & 1u) && local->generation == g1;
}

template <typename T>
inline void publish_snapshot(T* shared, const T& value) {
  const auto* gen_const = &shared->generation;
  uint64_t current = __atomic_load_n(gen_const, __ATOMIC_RELAXED);
  if (current & 1u) ++current;
  const uint64_t odd = current + 1u;
  const uint64_t even = odd + 1u;
  __atomic_store_n(&shared->generation, odd, __ATOMIC_RELEASE);
  // Keep generation itself atomic and copy the two surrounding byte ranges.
  constexpr std::size_t off = offsetof(T, generation);
  std::memcpy(reinterpret_cast<unsigned char*>(shared),
              reinterpret_cast<const unsigned char*>(&value), off);
  std::memcpy(reinterpret_cast<unsigned char*>(shared) + off + sizeof(uint64_t),
              reinterpret_cast<const unsigned char*>(&value) + off + sizeof(uint64_t),
              sizeof(T) - off - sizeof(uint64_t));
  std::atomic_thread_fence(std::memory_order_release);
  __atomic_store_n(&shared->generation, even, __ATOMIC_RELEASE);
}

}  // namespace wbc_rt
