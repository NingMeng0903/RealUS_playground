#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "wbc_rt/config.hpp"
#include "wbc_rt/inner.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/posture.hpp"
#include "wbc_rt/protocol.hpp"
#include "wbc_rt/shm.hpp"
#include "wbc_rt/srs_ik.hpp"

namespace {
std::atomic<bool> g_stop{false};
void on_sig(int) { g_stop = true; }

std::vector<double> take_nums(char** argv, int argc, int* i, int n) {
  std::vector<double> out;
  while (*i + 1 < argc && static_cast<int>(out.size()) < n) {
    char* end = nullptr;
    const double x = std::strtod(argv[*i + 1], &end);
    if (end == argv[*i + 1] || *end != '\0') break;
    ++*i;
    out.push_back(x);
  }
  return out;
}

int run_srs_ik(int argc, char** argv) {
  wbc_rt::Vec6 pose = wbc_rt::Vec6::Zero();
  double psi = 0.0;
  double y_s = 0.0;
  int branch = 0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  bool have_R = false;
  bool have_t = false;
  bool have_pose = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--srs-ik") continue;
    if (a == "--pose") {
      const auto v = take_nums(argv, argc, &i, 6);
      if (v.size() != 6) {
        std::cerr << "wbc_rt --srs-ik: --pose needs 6 numbers\n";
        return 2;
      }
      for (int k = 0; k < 6; ++k) pose[k] = v[static_cast<size_t>(k)];
      have_pose = true;
    } else if (a == "--psi") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) {
        std::cerr << "wbc_rt --srs-ik: --psi needs 1 number\n";
        return 2;
      }
      psi = v[0];
    } else if (a == "--branch") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) {
        std::cerr << "wbc_rt --srs-ik: --branch needs 1 number\n";
        return 2;
      }
      branch = static_cast<int>(v[0] + (v[0] >= 0.0 ? 0.5 : -0.5));
    } else if (a == "--y-shoulder") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) {
        std::cerr << "wbc_rt --srs-ik: --y-shoulder needs 1 number\n";
        return 2;
      }
      y_s = v[0];
    } else if (a == "--R") {
      const auto v = take_nums(argv, argc, &i, 9);
      if (v.size() != 9) {
        std::cerr << "wbc_rt --srs-ik: --R needs 9 numbers\n";
        return 2;
      }
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R(r, c) = v[static_cast<size_t>(r * 3 + c)];
      have_R = true;
    } else if (a == "--t") {
      const auto v = take_nums(argv, argc, &i, 3);
      if (v.size() != 3) {
        std::cerr << "wbc_rt --srs-ik: --t needs 3 numbers\n";
        return 2;
      }
      t << v[0], v[1], v[2];
      have_t = true;
    }
  }
  if (!have_pose) {
    std::cerr << "wbc_rt --srs-ik: --pose required\n";
    return 2;
  }
  if (have_R != have_t) {
    std::cerr << "wbc_rt --srs-ik: --R and --t must be given together\n";
    return 2;
  }
  const Eigen::Matrix3d* Rp = have_R ? &R : nullptr;
  const Eigen::Vector3d* tp = have_t ? &t : nullptr;
  const auto q = wbc_rt::srs::srs_ik(pose, psi, branch, y_s, Rp, tp);
  std::cout << std::setprecision(17);
  if (!q) {
    std::cout << "none\n";
    return 0;
  }
  for (int i = 0; i < 7; ++i) {
    if (i) std::cout << " ";
    std::cout << (*q)[i];
  }
  std::cout << "\n";
  return 0;
}

int run_psi_from_q(int argc, char** argv) {
  wbc_rt::Vec8 q = wbc_rt::Vec8::Zero();
  bool have_q = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--psi-from-q") continue;
    if (a == "--q") {
      const auto v = take_nums(argv, argc, &i, 8);
      if (v.size() != 8) {
        std::cerr << "wbc_rt --psi-from-q: --q needs 8 numbers\n";
        return 2;
      }
      for (int k = 0; k < 8; ++k) q[k] = v[static_cast<size_t>(k)];
      have_q = true;
    }
  }
  if (!have_q) {
    std::cerr << "wbc_rt --psi-from-q: --q required\n";
    return 2;
  }
  std::cout << std::setprecision(17) << wbc_rt::srs::psi_from_q(q) << " "
            << wbc_rt::srs::branch_from_q(q) << "\n";
  return 0;
}

int run_fk_pose(int argc, char** argv) {
  std::string config_path;
  wbc_rt::Vec8 q = wbc_rt::Vec8::Zero();
  bool have_q = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--fk-pose") continue;
    if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (a == "--q") {
      const auto v = take_nums(argv, argc, &i, 8);
      if (v.size() != 8) {
        std::cerr << "wbc_rt --fk-pose: --q needs 8 numbers\n";
        return 2;
      }
      for (int k = 0; k < 8; ++k) q[k] = v[static_cast<size_t>(k)];
      have_q = true;
    }
  }
  if (config_path.empty() || !have_q) {
    std::cerr << "wbc_rt --fk-pose: --config and --q required\n";
    return 2;
  }
  const auto cfg = wbc_rt::Config::load(config_path);
  wbc_rt::Kinematics kin(cfg.urdf);
  const wbc_rt::Vec6 pose = kin.fk_pose_at(q);
  std::cout << std::setprecision(17);
  for (int i = 0; i < 6; ++i) {
    if (i) std::cout << " ";
    std::cout << pose[i];
  }
  std::cout << "\n";
  return 0;
}

int run_posture_tick(int argc, char** argv) {
  std::string config_path;
  wbc_rt::Vec8 q = wbc_rt::Vec8::Zero();
  bool have_q = false;
  double dt = 0.005;
  double rail_lo = 0.005;
  double rail_hi = 0.78;
  int hold = 0;
  int ticks = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--posture-tick") continue;
    if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (a == "--q") {
      const auto v = take_nums(argv, argc, &i, 8);
      if (v.size() != 8) {
        std::cerr << "wbc_rt --posture-tick: --q needs 8 numbers\n";
        return 2;
      }
      for (int k = 0; k < 8; ++k) q[k] = v[static_cast<size_t>(k)];
      have_q = true;
    } else if (a == "--dt") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      dt = v[0];
    } else if (a == "--rail-lo") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      rail_lo = v[0];
    } else if (a == "--rail-hi") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      rail_hi = v[0];
    } else if (a == "--hold") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      hold = static_cast<int>(v[0] + (v[0] >= 0.0 ? 0.5 : -0.5));
    } else if (a == "--ticks") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      ticks = static_cast<int>(v[0] + (v[0] >= 0.0 ? 0.5 : -0.5));
    }
  }
  if (config_path.empty() || !have_q) {
    std::cerr << "wbc_rt --posture-tick: --config and --q required\n";
    return 2;
  }
  const auto cfg = wbc_rt::Config::load(config_path);
  wbc_rt::Kinematics kin(cfg.urdf);
  wbc_rt::PostureRetarget posture(cfg);
  const wbc_rt::Vec6 pose0 = kin.fk_pose_at(q);
  posture.reset(q, pose0);
  auto dump = [&]() {
    std::cout << std::setprecision(17) << posture.homotopy_s() << " " << posture.d_star() << " "
              << posture.psi_cmd() << " " << posture.psi_star();
    const auto& qs = posture.q_star();
    for (int i = 0; i < wbc_rt::kNv; ++i) std::cout << " " << qs[i];
    std::cout << "\n";
  };
  dump();
  for (int k = 0; k < ticks; ++k) {
    const wbc_rt::Vec6 pose = kin.fk_pose_at(q);
    posture.step(q, pose, dt, rail_lo, rail_hi, hold != 0);
    dump();
  }
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  std::string in_name = "rm75_wbc_in";
  std::string out_name = "rm75_wbc_out";
  bool want_srs = false;
  bool want_psi = false;
  bool want_fk = false;
  bool want_posture = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (a == "--in" && i + 1 < argc) in_name = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_name = argv[++i];
    else if (a == "--srs-ik") want_srs = true;
    else if (a == "--psi-from-q") want_psi = true;
    else if (a == "--fk-pose") want_fk = true;
    else if (a == "--posture-tick") want_posture = true;
    else if (a == "--help") {
      std::cout << "wbc_rt --config FILE --in NAME --out NAME\n"
                << "wbc_rt --srs-ik --pose 6 --psi P --branch B --y-shoulder Y [--R 9 --t 3]\n"
                << "wbc_rt --psi-from-q --q 8\n"
                << "wbc_rt --fk-pose --config FILE --q 8\n"
                << "wbc_rt --posture-tick --config FILE --q 8 [--dt --rail-lo --rail-hi --hold --ticks]\n";
      return 0;
    } else if (a == "--sizes") {
      std::cout << sizeof(wbc_rt::WbcIn) << " " << sizeof(wbc_rt::WbcOut) << "\n";
      return 0;
    }
  }
  if (want_srs) return run_srs_ik(argc, argv);
  if (want_psi) return run_psi_from_q(argc, argv);
  if (want_fk) return run_fk_pose(argc, argv);
  if (want_posture) return run_posture_tick(argc, argv);
  if (config_path.empty()) {
    std::cerr << "wbc_rt: --config required\n";
    return 2;
  }
  std::signal(SIGINT, on_sig);
  std::signal(SIGTERM, on_sig);

  try {
    const auto cfg = wbc_rt::Config::load(config_path);
    wbc_rt::InnerLoop loop(cfg);
    wbc_rt::ShmMap in_map;
    wbc_rt::ShmMap out_map;
    in_map.open(in_name, sizeof(wbc_rt::WbcIn));
    out_map.open(out_name, sizeof(wbc_rt::WbcOut));
    auto* in = reinterpret_cast<wbc_rt::WbcIn*>(in_map.ptr);
    auto* out = reinterpret_cast<wbc_rt::WbcOut*>(out_map.ptr);
    wbc_rt::clear_out(out);
    out->status = wbc_rt::kStatusReady;
    out->flags = wbc_rt::kOutReady;
    std::uint64_t last_seq = 0;
    while (!g_stop) {
      const std::uint64_t seq = in->seq;
      if (seq == last_seq) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      last_seq = seq;
      const std::uint32_t cmd = in->cmd;
      if (cmd == wbc_rt::kCmdShutdown) {
        out->status = wbc_rt::kStatusShutdown;
        out->seq = seq;
        out->cmd_ack = in->cmd_seq;
        break;
      }
      auto publish_q = [&]() {
        const auto& q = loop.q_cmd();
        for (int i = 0; i < wbc_rt::kNv; ++i) out->q_cmd[i] = q[i];
      };
      if (cmd == wbc_rt::kCmdEnable) {
        loop.enable();
      } else if (cmd == wbc_rt::kCmdStop) {
        loop.stop();
      } else if (cmd == wbc_rt::kCmdReset) {
        wbc_rt::Vec8 q;
        for (int i = 0; i < wbc_rt::kNv; ++i) q[i] = in->q_meas[i];
        loop.reset(q);
      } else if (cmd == wbc_rt::kCmdBeginHybrid) {
        wbc_rt::Vec8 q, qd = wbc_rt::Vec8::Zero();
        for (int i = 0; i < wbc_rt::kNv; ++i) {
          q[i] = in->q_meas[i];
          qd[i] = in->cmd_f[i];
        }
        loop.begin_hybrid(q, qd);
      } else if (cmd == wbc_rt::kCmdSetRailMode) {
        loop.set_rail_mode(in->cmd_u[0], in->cmd_u[1], in->cmd_f[0], in->cmd_f[1] > 0.5);
      } else if (cmd == wbc_rt::kCmdSetFlags) {
        loop.set_flags(in->cmd_u[0]);
      } else if (cmd == wbc_rt::kCmdSetStroke) {
        loop.set_stroke(in->cmd_f[0], in->cmd_f[1]);
      } else if (cmd == wbc_rt::kCmdPlanStroke) {
        wbc_rt::Vec8 q;
        for (int i = 0; i < wbc_rt::kNv; ++i) q[i] = in->q_meas[i];
        auto [d, psi] = loop.plan_stroke(q, in->cmd_f[0], in->cmd_f[1]);
        out->cmd_f[0] = d;
        out->cmd_f[1] = psi;
      } else if (cmd == wbc_rt::kCmdSetRailPoseTarget) {
        loop.set_rail_pose_target(in->cmd_f[0], in->cmd_f[1] > 0.5);
      } else if (cmd == wbc_rt::kCmdCaptureRailExtRef) {
        wbc_rt::Vec8 q;
        for (int i = 0; i < wbc_rt::kNv; ++i) q[i] = in->q_meas[i];
        loop.capture_rail_ext_ref(q);
      } else if (cmd == wbc_rt::kCmdSetRailExtMode) {
        loop.set_rail_ext_mode(in->cmd_f[0] > 0.5 ? 1 : 0);
      } else if (cmd == wbc_rt::kCmdStep) {
        wbc_rt::TickIn tin;
        for (int i = 0; i < 6; ++i) {
          tin.v_cmd[i] = in->v_cmd[i];
          tin.pose_d[i] = in->pose_d[i];
          tin.vel_ff[i] = in->vel_ff[i];
          tin.path_twist[i] = in->path_twist[i];
          tin.feedback_twist[i] = in->feedback_twist[i];
        }
        for (int i = 0; i < 8; ++i) {
          tin.q_meas[i] = in->q_meas[i];
          tin.qdot_ff[i] = in->qdot_ff[i];
        }
        tin.dt_nom = in->dt_nom;
        tin.dt_wall = in->dt_wall;
        tin.t_mono = in->t_mono;
        tin.rail_v = in->rail_v;
        tin.v_force_z = in->v_force_z;
        tin.flags = in->flags;
        tin.posture_d = in->cmd_f[0];
        tin.posture_psi = in->cmd_f[1];
        if (in->flags & wbc_rt::kInHasQStar) {
          for (int i = 0; i < wbc_rt::kNv; ++i) tin.posture_q[i] = in->cmd_f[3 + i];
        }
        const auto tout = loop.step(tin);
        for (int i = 0; i < 8; ++i) {
          out->q_cmd[i] = tout.q_cmd[i];
          out->qdot[i] = tout.qdot[i];
        }
        for (int i = 0; i < 6; ++i) {
          out->v_cmd_received[i] = tout.v_recv[i];
          out->v_cmd_feasible[i] = tout.v_feas[i];
          out->v_tcp_estimated[i] = tout.v_tcp[i];
          out->task_residual[i] = tout.residual[i];
        }
        out->slack = tout.slack;
        out->e_qp = tout.e_qp;
        out->u_alloc = tout.u_alloc;
        out->u_mid = tout.u_mid;
        out->v_r_ref = tout.v_r_ref;
        out->psi = tout.psi;
        out->d_star = tout.d_star;
        out->d_pref = tout.d_pref;
        out->solve_ms = tout.solve_ms;
        out->sigma_min = tout.sigma_min;
        out->sigma_arm = tout.sigma_arm;
        out->joint_limited = tout.joint_limited;
        out->rail_limited = tout.rail_limited;
        out->wall_active = tout.wall_active;
        out->secondary_suppressed = tout.secondary_suppressed;
        out->ns_norm = tout.ns_norm;
        out->ns_centering = tout.ns_centering;
        out->ns_manip = tout.ns_manip;
        out->ns_arm_angle = tout.ns_arm_angle;
        out->ns_damping = tout.ns_damping;
        out->ns_rail_lock = tout.ns_rail_lock;
        out->sat_scale = tout.sat_scale;
        out->sec_target_norm = tout.sec_target_norm;
        out->flags = tout.flags;
        out->status = tout.status;
      }
      if (cmd != wbc_rt::kCmdStep) {
        publish_q();
      }
      out->magic = wbc_rt::kMagic;
      out->version = wbc_rt::kVersion;
      out->cmd_ack = in->cmd_seq;
      out->seq = seq;
    }
  } catch (const std::exception& e) {
    std::cerr << "wbc_rt: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
