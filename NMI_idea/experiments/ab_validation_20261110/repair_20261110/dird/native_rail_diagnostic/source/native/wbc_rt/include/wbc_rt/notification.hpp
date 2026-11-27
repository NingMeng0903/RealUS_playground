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

  // timeout_ms < 0 waits forever. A timeout returns true so the caller can
  // re-read SHM; the byte is only an optimization, not the request itself.
  bool wait(const std::atomic<bool>& stop, int timeout_ms = -1) const {
    while (!stop) {
      pollfd pfd{fd_, POLLIN, 0};
      const int ready = ::poll(&pfd, 1, timeout_ms);
      if (ready < 0 && errno == EINTR) continue;
      if (ready < 0) throw std::runtime_error("wbc_rt notification poll failed");
      if (pfd.revents & POLLNVAL) return false;
      if (ready == 0) return true;
      std::uint8_t buf[16];
      const auto got = ::recv(fd_, buf, sizeof(buf), 0);
      if (got < 0 && errno == EINTR) continue;
      if (got < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
      return got > 0;
    }
    return false;
  }

  bool notify() const {
    if (!enabled()) return true;
    const std::uint8_t byte = 1;
    while (::send(fd_, &byte, sizeof(byte), MSG_NOSIGNAL | MSG_DONTWAIT) < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return true;
      if (errno != EINTR) return false;
    }
    return true;
  }

 private:
  int fd_;
};

}  // namespace wbc_rt
