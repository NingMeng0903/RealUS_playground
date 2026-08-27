#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "wbc_rt/config.hpp"
#include "wbc_rt/inner.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/posture.hpp"
#include "wbc_rt/protocol.hpp"
#include "wbc_rt/rail_command.hpp"
#include "wbc_rt/shm.hpp"
#include "wbc_rt/srs_ik.hpp"
#include "wbc_rt/task_weight.hpp"

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
  wbc_rt::Kinematics kin(cfg.urdf, cfg.tcp_placement_R(), cfg.tcp_placement_t());
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
  wbc_rt::Kinematics kin(cfg.urdf, cfg.tcp_placement_R(), cfg.tcp_placement_t());
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

bool parse_flag01(const std::string& s, bool* ok) {
  if (s == "1" || s == "true" || s == "True" || s == "yes") {
    *ok = true;
    return true;
  }
  if (s == "0" || s == "false" || s == "False" || s == "no") {
    *ok = true;
    return false;
  }
  *ok = false;
  return false;
}

int run_press_escape(int argc, char** argv) {
  bool demanding = false, has_travel = false, press_stalled = false;
  bool j4_blocked = false, arm_starved = false, policy_leave = false;
  int got = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--press-escape") continue;
    if (a == "--flags") {
      const auto v = take_nums(argv, argc, &i, 6);
      if (v.size() != 6) {
        std::cerr << "wbc_rt --press-escape: --flags needs 6 numbers\n";
        return 2;
      }
      demanding = v[0] > 0.5;
      has_travel = v[1] > 0.5;
      press_stalled = v[2] > 0.5;
      j4_blocked = v[3] > 0.5;
      arm_starved = v[4] > 0.5;
      policy_leave = v[5] > 0.5;
      got = 6;
    } else if (a == "--demanding" && i + 1 < argc) {
      bool ok = false;
      demanding = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    } else if (a == "--has-travel" && i + 1 < argc) {
      bool ok = false;
      has_travel = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    } else if (a == "--press-stalled" && i + 1 < argc) {
      bool ok = false;
      press_stalled = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    } else if (a == "--j4-blocked" && i + 1 < argc) {
      bool ok = false;
      j4_blocked = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    } else if (a == "--arm-starved" && i + 1 < argc) {
      bool ok = false;
      arm_starved = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    } else if (a == "--policy-leave-flag" && i + 1 < argc) {
      bool ok = false;
      policy_leave = parse_flag01(argv[++i], &ok);
      if (!ok) return 2;
      ++got;
    }
  }
  if (got < 6) {
    std::cerr << "wbc_rt --press-escape: --flags 6 required\n";
    return 2;
  }
  const bool allowed = wbc_rt::press_escape_allowed_from_flags(
      demanding, has_travel, press_stalled, j4_blocked, arm_starved, policy_leave);
  std::cout << (allowed ? "1\n" : "0\n");
  return 0;
}

int run_policy_leave(int argc, char** argv) {
  double y = 0.0, urdf_lo = 0.0, urdf_hi = 0.785, soft_min = 0.005, soft_max = 0.78;
  double escape_leave = 0.04, pin_margin = 0.008, latched_sign = 0.0;
  std::string policy = "auto";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--policy-leave") continue;
    if (a == "--y") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      y = v[0];
    } else if (a == "--urdf-lo") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      urdf_lo = v[0];
    } else if (a == "--urdf-hi") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      urdf_hi = v[0];
    } else if (a == "--soft-min") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      soft_min = v[0];
    } else if (a == "--soft-max") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      soft_max = v[0];
    } else if (a == "--escape-leave") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      escape_leave = v[0];
    } else if (a == "--pin-margin") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      pin_margin = v[0];
    } else if (a == "--latched-sign") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      latched_sign = v[0];
    } else if (a == "--policy" && i + 1 < argc) {
      policy = argv[++i];
    }
  }
  double lo = 0.0, hi = 0.0;
  wbc_rt::soft_rail_travel(urdf_lo, urdf_hi, soft_min, soft_max, &lo, &hi);
  const double sign = wbc_rt::policy_escape_sign(policy, y, lo, hi, latched_sign);
  const double leave = wbc_rt::leave_margin_m(escape_leave, pin_margin);
  const bool in_leave = wbc_rt::in_leave_band(y, lo, hi, leave, sign);
  std::cout << std::setprecision(17) << sign << " " << (in_leave ? 1 : 0) << "\n";
  return 0;
}

int run_task_weight(int argc, char** argv) {
  std::vector<wbc_rt::Mat6x8> Js;
  wbc_rt::Vec6 w = wbc_rt::Vec6::Ones();
  double dt = 0.005;
  double tau = 0.25;
  double sigma_ref = 0.08;
  double min_frac = 0.05;
  bool aniso = true;
  bool zero_rail = false;
  int ticks = 1;
  int reset_before_tick = -1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--task-weight") continue;
    if (a == "--J") {
      const auto v = take_nums(argv, argc, &i, 48);
      if (v.size() != 48) {
        std::cerr << "wbc_rt --task-weight: --J needs 48 numbers (row-major 6x8)\n";
        return 2;
      }
      wbc_rt::Mat6x8 J = wbc_rt::Mat6x8::Zero();
      for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 8; ++c) J(r, c) = v[static_cast<size_t>(r * 8 + c)];
      Js.push_back(J);
    } else if (a == "--w") {
      const auto v = take_nums(argv, argc, &i, 6);
      if (v.size() != 6) {
        std::cerr << "wbc_rt --task-weight: --w needs 6 numbers\n";
        return 2;
      }
      for (int k = 0; k < 6; ++k) w[k] = v[static_cast<size_t>(k)];
    } else if (a == "--dt") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      dt = v[0];
    } else if (a == "--tau") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      tau = v[0];
    } else if (a == "--sigma-ref") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      sigma_ref = v[0];
    } else if (a == "--min-frac") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      min_frac = v[0];
    } else if (a == "--aniso") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      aniso = v[0] > 0.5;
    } else if (a == "--ticks") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      ticks = static_cast<int>(v[0] + (v[0] >= 0.0 ? 0.5 : -0.5));
    } else if (a == "--reset-before-tick") {
      const auto v = take_nums(argv, argc, &i, 1);
      if (v.size() != 1) return 2;
      reset_before_tick = static_cast<int>(v[0] + (v[0] >= 0.0 ? 0.5 : -0.5));
    } else if (a == "--zero-rail") {
      zero_rail = true;
    }
  }
  if (Js.empty()) {
    std::cerr << "wbc_rt --task-weight: --J required\n";
    return 2;
  }
  if (ticks < 1) ticks = 1;
  wbc_rt::TaskWeightState st;
  std::cout << std::setprecision(17);
  for (int t = 1; t <= ticks; ++t) {
    if (reset_before_tick > 0 && t == reset_before_tick) st.reset();
    wbc_rt::Mat6x8 Jt = Js[static_cast<size_t>(std::min(t, static_cast<int>(Js.size())) - 1)];
    if (zero_rail) Jt.col(0).setZero();
    const wbc_rt::Mat6 W = st.step(Jt, w, dt, tau, sigma_ref, min_frac, aniso);
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        if (r || c) std::cout << " ";
        std::cout << W(r, c);
      }
    }
    std::cout << "\n";
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
  bool want_press = false;
  bool want_leave = false;
  bool want_tw = false;
  bool want_protocol_info = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    else if (a == "--in" && i + 1 < argc) in_name = argv[++i];
    else if (a == "--out" && i + 1 < argc) out_name = argv[++i];
    else if (a == "--srs-ik") want_srs = true;
    else if (a == "--psi-from-q") want_psi = true;
    else if (a == "--fk-pose") want_fk = true;
    else if (a == "--posture-tick") want_posture = true;
    else if (a == "--press-escape") want_press = true;
    else if (a == "--policy-leave") want_leave = true;
    else if (a == "--task-weight") want_tw = true;
    else if (a == "--protocol-info") want_protocol_info = true;
    else if (a == "--help") {
      std::cout << "wbc_rt --config FILE --in NAME --out NAME\n"
                << "wbc_rt --srs-ik --pose 6 --psi P --branch B --y-shoulder Y [--R 9 --t 3]\n"
                << "wbc_rt --psi-from-q --q 8\n"
                << "wbc_rt --fk-pose --config FILE --q 8\n"
                << "wbc_rt --posture-tick --config FILE --q 8 [--dt --rail-lo --rail-hi --hold --ticks]\n"
                << "wbc_rt --press-escape --flags 6\n"
                << "wbc_rt --policy-leave --y Y --policy P [--soft-min --soft-max --latched-sign ...]\n"
                << "wbc_rt --task-weight --J 48 [--w 6 --dt --tau --aniso --ticks --zero-rail]\n"
                << "wbc_rt --protocol-info  # print version and packed SHM sizes\n";
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
  if (want_press) return run_press_escape(argc, argv);
  if (want_leave) return run_policy_leave(argc, argv);
  if (want_tw) return run_task_weight(argc, argv);
  if (want_protocol_info) {
    std::cout << wbc_rt::kVersion << " " << sizeof(wbc_rt::WbcIn) << " "
              << sizeof(wbc_rt::WbcOut) << "\n";
    return 0;
  }
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
    auto* in_shared = reinterpret_cast<wbc_rt::WbcIn*>(in_map.ptr);
    auto* out_shared = reinterpret_cast<wbc_rt::WbcOut*>(out_map.ptr);
    wbc_rt::WbcOut out_initial;
    wbc_rt::clear_out(&out_initial);
    out_initial.status = wbc_rt::kStatusReady;
    out_initial.flags = wbc_rt::kOutReady;
    wbc_rt::publish_snapshot(out_shared, out_initial);
    wbc_rt::WbcIn in_snapshot;
    wbc_rt::WbcOut out_snapshot;
    std::uint64_t last_seq = 0;
    while (!g_stop) {
      if (!wbc_rt::read_snapshot(in_shared, &in_snapshot)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      const auto* in = &in_snapshot;
      if (!wbc_rt::read_snapshot(out_shared, &out_snapshot)) {
        wbc_rt::clear_out(&out_snapshot);
        out_snapshot.status = wbc_rt::kStatusReady;
        out_snapshot.flags = wbc_rt::kOutReady;
      }
      auto* out = &out_snapshot;
      const std::uint64_t seq = in->seq;
      if (seq == last_seq) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      last_seq = seq;
      if (in->magic != wbc_rt::kMagic || in->version != wbc_rt::kVersion) {
        out->status = wbc_rt::kStatusFail;
        out->flags = wbc_rt::kOutFailed;
        out->cmd_ack = in->cmd_seq;
        out->seq = seq;
        out->magic = wbc_rt::kMagic;
        out->version = wbc_rt::kVersion;
        wbc_rt::publish_snapshot(out_shared, *out);
        continue;
      }
      const std::uint32_t cmd = in->cmd;
      if (cmd == wbc_rt::kCmdShutdown) {
        out->status = wbc_rt::kStatusShutdown;
        out->seq = seq;
        out->cmd_ack = in->cmd_seq;
        out->magic = wbc_rt::kMagic;
        out->version = wbc_rt::kVersion;
        wbc_rt::publish_snapshot(out_shared, *out);
        break;
      }
      auto publish_q = [&]() {
        const auto& q = loop.q_cmd();
        for (int i = 0; i < wbc_rt::kNv; ++i) out->q_cmd[i] = q[i];
      };
      if (cmd == wbc_rt::kCmdEnable) {
        loop.enable();
        out->flags = wbc_rt::kOutReady;
        out->status = wbc_rt::kStatusReady;
      } else if (cmd == wbc_rt::kCmdStop) {
        loop.stop();
        std::fill(std::begin(out->qdot), std::end(out->qdot), 0.0);
        std::fill(std::begin(out->v_cmd_feasible), std::end(out->v_cmd_feasible), 0.0);
        std::fill(std::begin(out->v_tcp_estimated), std::end(out->v_tcp_estimated), 0.0);
        out->flags = wbc_rt::kOutStale | wbc_rt::kOutFailed;
        out->status = wbc_rt::kStatusFail;
        out->fallback_level = wbc_rt::kFallbackStop;
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
        out->homotopy_s = tout.homotopy_s;
        out->psi_star = tout.psi_star;
        out->rail_motion_share = tout.rail_motion_share;
        out->u_task_raw = tout.u_task_raw;
        out->u_task_feasible = tout.u_task_feasible;
        out->u_pi_raw = tout.u_pi_raw;
        out->u_mid_cmd = tout.u_mid_cmd;
        out->u_post_raw = tout.u_post_raw;
        out->u_post_feasible = tout.u_post_feasible;
        out->u_mid_applied = tout.u_mid_applied;
        out->d_star_dot_cmd = tout.d_star_dot_cmd;
        out->u_escape_raw = tout.u_escape_raw;
        out->u_escape_feasible = tout.u_escape_feasible;
        out->escape_active = tout.escape_active;
        out->escape_dir = tout.escape_dir;
        out->u_base = tout.u_base;
        out->u_feasible = tout.u_feasible;
        out->v_r_lpf = tout.v_r_lpf;
        out->e_d = tout.e_d;
        out->V_d_proxy = tout.V_d_proxy;
        out->j4_design_slack = tout.j4_design_slack;
        out->sigma_slack = tout.sigma_slack;
        out->rail_box_lo = tout.rail_box_lo;
        out->rail_box_hi = tout.rail_box_hi;
        out->rail_bind_lo = tout.rail_bind_lo;
        out->rail_bind_hi = tout.rail_bind_hi;
        out->rail_task_vel_used = tout.rail_task_vel_used;
        out->rail_h1 = tout.rail_h1;
        out->rail_h2 = tout.rail_h2;
        out->rail_qdot_prev = tout.rail_qdot_prev;
        out->rail_qdot_prev2 = tout.rail_qdot_prev2;
        out->qp1_status = tout.qp1_status;
        out->qp2_status = tout.qp2_status;
        out->fallback_level = tout.fallback_level;
        out->failure_code = tout.failure_code;
        out->qp1_hard_violation = tout.qp1_hard_violation;
        out->final_hard_violation = tout.final_hard_violation;
        out->task_lock_violation = tout.task_lock_violation;
        out->final_box_violation = tout.final_box_violation;
        out->qp_overrun = tout.qp_overrun;
        out->posture_gate = tout.posture_gate;
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
      wbc_rt::publish_snapshot(out_shared, *out);
    }
  } catch (const std::exception& e) {
    std::cerr << "wbc_rt: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
