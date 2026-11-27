#pragma once

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace wbc_rt {

struct ShmMap {
  void* ptr = nullptr;
  int fd = -1;
  std::size_t size = 0;
  std::string name;

  ShmMap() = default;
  ShmMap(const ShmMap&) = delete;
  ShmMap& operator=(const ShmMap&) = delete;
  ~ShmMap() { close(); }

  void open(const std::string& raw, std::size_t bytes) {
    name = raw;
    if (name.empty() || name[0] != '/') name = "/" + raw;
    fd = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd < 0) {
      throw std::runtime_error(std::string("shm_open ") + name + ": " + std::strerror(errno));
    }
    ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
      ::close(fd);
      fd = -1;
      ptr = nullptr;
      throw std::runtime_error(std::string("mmap ") + name + ": " + std::strerror(errno));
    }
    size = bytes;
  }

  void close() {
    if (ptr && ptr != MAP_FAILED) munmap(ptr, size);
    if (fd >= 0) ::close(fd);
    ptr = nullptr;
    fd = -1;
    size = 0;
  }
};

}  // namespace wbc_rt
