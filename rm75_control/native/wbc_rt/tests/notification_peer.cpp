// Transport-only test peer: no QP, kinematics, or hardware dependencies.
#include <chrono>
#include <cstdlib>
#include <sys/prctl.h>
#include "wbc_rt/notification.hpp"
#include "wbc_rt/protocol.hpp"
#include "wbc_rt/shm.hpp"

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  // Reproduce inherited desktop timer slack. FD readiness must still wake
  // immediately; a 50-us sleep poll can overshoot the 20-ms client deadline.
  if (::prctl(PR_SET_TIMERSLACK, 50000000UL, 0, 0, 0) != 0) return 3;
  wbc_rt::NotificationChannel channel(std::atoi(argv[1]));
  wbc_rt::ShmMap input, output;
  input.open(argv[2], sizeof(wbc_rt::WbcIn));
  output.open(argv[3], sizeof(wbc_rt::WbcOut));
  const auto* in = static_cast<wbc_rt::WbcIn*>(input.ptr);
  auto* out = static_cast<wbc_rt::WbcOut*>(output.ptr);
  wbc_rt::clear_out(out);
  out->status = wbc_rt::kStatusReady;
  if (!channel.notify()) return 0;
  const std::atomic<bool> stop{false};
  while (channel.wait(stop)) {
    const wbc_rt::WbcIn request = *in;
    if (request.cmd == wbc_rt::kCmdShutdown) return 0;
    // Deliberately write partial, unacknowledged telemetry while solving.
    out->solve_ms = 777.0;
    for (int k = 0; k < 8; ++k) out->q_cmd[k] = 999.0;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    while (std::chrono::steady_clock::now() < until) {}
    for (int k = 0; k < 8; ++k) out->q_cmd[k] = request.q_meas[k];
    out->solve_ms = 1.0;
    out->status = wbc_rt::kStatusOk;
    out->cmd_ack = request.cmd_seq;
    __atomic_store_n(reinterpret_cast<std::uint64_t*>(
        static_cast<char*>(output.ptr) + offsetof(wbc_rt::WbcOut, seq)),
        request.seq, __ATOMIC_RELEASE);
    if (!channel.notify()) break;
  }
  return 0;
}
