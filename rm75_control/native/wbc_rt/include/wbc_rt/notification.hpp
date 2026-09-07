#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <stdexcept>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace wbc_rt {

// A socket inherited from the Python owner carries wakeups; payloads remain
// in SHM. Peer closure also stops the worker if its owner exits unexpectedly.
class NotificationChannel {
 public:
  explicit NotificationChannel(int fd) : fd_(fd) {}
  ~NotificationChannel() { if (fd_ >= 0) ::close(fd_); }
  bool enabled() const { return fd_ >= 0; }

  bool wait(const std::atomic<bool>& stop) const {
    while (!stop) {
      pollfd pfd{fd_, POLLIN, 0};
      const int ready = ::poll(&pfd, 1, -1);
      if (ready < 0 && errno == EINTR) continue;
      if (ready < 0) throw std::runtime_error("wbc_rt notification poll failed");
      if (pfd.revents & POLLNVAL) return false;
      std::uint8_t byte;
      const auto got = ::recv(fd_, &byte, sizeof(byte), 0);
      if (got < 0 && errno == EINTR) continue;
      return got == sizeof(byte);
    }
    return false;
  }

  bool notify() const {
    if (!enabled()) return true;
    const std::uint8_t byte = 1;
    while (::send(fd_, &byte, sizeof(byte), MSG_NOSIGNAL) < 0) {
      if (errno != EINTR) return false;
    }
    return true;
  }

 private:
  int fd_;
};

}  // namespace wbc_rt
