// Hardware counters for this process and the threads it starts later.
#pragma once

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>

// Create it before the first OpenMP parallel region. Threads started later
// inherit the counters. A counter that cannot be opened reads as -1.
//
// The cache events are raw AMD Zen 2/3/4 codes. They count cache lines.
//   l2_miss     demand requests from the core that missed L2 (PMCx064)
//   l2pf_l3     L2 prefetches served by L3                   (PMCx071)
//   l2pf_dram   L2 prefetches that missed L3                 (PMCx072)
//   dram_fills  L1D fills sourced from DRAM                  (PMCx044)
// Lines that cross into L2 = l2_miss + l2pf_l3 + l2pf_dram.
// Lines read from DRAM ~= l2pf_dram + min(l2_miss, dram_fills).
// On other CPUs these columns have no meaning.
class PerfCounters {
 public:
  static constexpr int kCount = 6;
  static constexpr std::array<const char*, kCount> kNames = {
      "cycles", "instructions", "l2_miss", "l2pf_l3", "l2pf_dram", "dram_fills"};
  using Sample = std::array<std::array<std::uint64_t, 3>, kCount>;
  using Counts = std::array<std::int64_t, kCount>;

  PerfCounters() {
    fds_[0] = open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    fds_[1] = open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    fds_[2] = open(PERF_TYPE_RAW, 0x0964);
    fds_[3] = open(PERF_TYPE_RAW, 0xff71);
    fds_[4] = open(PERF_TYPE_RAW, 0xff72);
    fds_[5] = open(PERF_TYPE_RAW, 0x0844);
  }
  ~PerfCounters() {
    for (int fd : fds_) {
      if (fd >= 0) close(fd);
    }
  }
  PerfCounters(const PerfCounters&) = delete;
  PerfCounters& operator=(const PerfCounters&) = delete;

  // Each entry is {value, time enabled, time running}.
  Sample read() const {
    Sample s{};
    for (int i = 0; i < kCount; ++i) {
      if (fds_[i] < 0 || ::read(fds_[i], s[i].data(), sizeof s[i]) != sizeof s[i]) {
        s[i] = {0, 0, 0};
      }
    }
    return s;
  }

  // When there are more events than hardware counters the kernel runs each
  // event only part of the time. The count is then scaled up to the full time.
  static Counts delta(const Sample& before, const Sample& after) {
    Counts d;
    for (int i = 0; i < kCount; ++i) {
      const double value = static_cast<double>(after[i][0] - before[i][0]);
      const double enabled = static_cast<double>(after[i][1] - before[i][1]);
      const double running = static_cast<double>(after[i][2] - before[i][2]);
      d[i] = running > 0 ? static_cast<std::int64_t>(value * enabled / running) : -1;
    }
    return d;
  }

 private:
  static int open(std::uint32_t type, std::uint64_t config) {
    perf_event_attr attr;
    std::memset(&attr, 0, sizeof attr);
    attr.size = sizeof attr;
    attr.type = type;
    attr.config = config;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.inherit = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0));
  }

  std::array<int, kCount> fds_;
};
