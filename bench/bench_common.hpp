// Shared by the benchmark programs. Times repetitions and writes CSV rows.
#pragma once

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

// Each row is synced to disk. An interrupted run then keeps everything
// measured so far.
class Recorder {
 public:
  Recorder(const std::string& out, const char* columns) {
    if (!out.empty()) {
      file_ = std::fopen(out.c_str(), "a");
      if (!file_) {
        std::perror(out.c_str());
        std::exit(1);
      }
    }
    if (file_ == stdout || std::ftell(file_) == 0) {
      std::fprintf(file_, "%s,rep,seconds,gflops", columns);
      std::fprintf(file_, ",rel_err\n");
    }
  }
  ~Recorder() {
    if (file_ != stdout) std::fclose(file_);
  }
  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;

  // rel_err < 0 means "not checked".
  template <typename Fn>
  void run(const std::string& values, int reps, double flops, double rel_err, Fn&& fn) {
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      fn();
      const auto t1 = std::chrono::steady_clock::now();
      const double s = std::chrono::duration<double>(t1 - t0).count();
      std::fprintf(file_, "%s,%d,%.9f,%.4f", values.c_str(), r, s, flops / s / 1e9);
      if (rel_err >= 0.0) std::fprintf(file_, ",%.3e\n", rel_err);
      else std::fprintf(file_, ",\n");
      std::fflush(file_);
      if (file_ != stdout) fsync(fileno(file_));
    }
  }

 private:
  std::FILE* file_ = stdout;
};
