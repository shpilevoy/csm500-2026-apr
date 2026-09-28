// Shows where full-matrix attention reads main memory. Runs the three phases
// one by one and reports how many times each phase read the score table.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gemm_packed.hpp"
#include "perf_counters.hpp"
#include "util.hpp"

namespace {

constexpr int H = 8, d = 64;

// The same estimate as dram_lines in scripts/plot_results.py.
double dram_lines(const PerfCounters::Counts& c) {
  const double l2_miss = c[2], l2pf_dram = c[4], dram_fills = c[5];
  return l2pf_dram + std::min(l2_miss, dram_fills);
}

}  // namespace

int main(int argc, char** argv) {
  const int L = argc > 1 ? std::atoi(argv[1]) : 8192;
  const int strip = argc > 2 ? std::atoi(argv[2]) : L;
  if (L <= 0 || strip <= 0) {
    std::fprintf(stderr, "usage: attn_phases [LEN] [STRIP]\n");
    return 2;
  }
  PerfCounters counters;
  const std::size_t head = static_cast<std::size_t>(L) * d, table = static_cast<std::size_t>(L) * L;
  auto Q = alloc_floats(H * head), K = alloc_floats(H * head), V = alloc_floats(H * head), O = alloc_floats(H * head);
  fill_uniform(Q.get(), H * head, 1);
  fill_uniform(K.get(), H * head, 2);
  fill_uniform(V.get(), H * head, 3);
  std::vector<FloatBuf> S(H), Kp(H), Vp(H);
#pragma omp parallel for
  for (int h = 0; h < H; ++h) {
    S[h] = alloc_floats(table);
    std::memset(S[h].get(), 0, table * sizeof(float));
    Kp[h] = alloc_floats(gemm_packed_b_size(L, d));
    Vp[h] = alloc_floats(gemm_packed_b_size(d, L));
    gemm_pack_b(L, d, K.get() + h * head, d, true, Kp[h].get());
    gemm_pack_b(d, L, V.get() + h * head, d, false, Vp[h].get());
  }
  const double table_bytes = static_cast<double>(H) * table * sizeof(float);

  std::printf("stores,phase,seconds,reads_of_table\n");
  for (const bool stream : {false, true}) {
    // The first round warms up. Only the second round is reported.
    for (int round = 0; round < 2; ++round) {
      auto report = [&](const char* phase, const PerfCounters::Sample& before, double seconds) {
        if (round == 0) return;
        const auto c = PerfCounters::delta(before, counters.read());
        std::printf("%s,%s,%.3f,%.2f\n", stream ? "streaming" : "regular", phase, seconds,
                    dram_lines(c) * 64.0 / table_bytes);
      };

      auto c0 = counters.read();
      double t0 = omp_get_wtime();
#pragma omp parallel for
      for (int h = 0; h < H; ++h) {
        for (int q0 = 0; q0 < L; q0 += strip) {
          gemm_packed(std::min(strip, L - q0), L, d, Q.get() + h * head + static_cast<std::size_t>(q0) * d, d,
                      nullptr, 0, S[h].get() + static_cast<std::size_t>(q0) * L, L,
                      {.stream_c = stream, .packed_b = Kp[h].get()});
        }
      }
      report("1 write scores", c0, omp_get_wtime() - t0);

      c0 = counters.read();
      t0 = omp_get_wtime();
#pragma omp parallel for
      for (int h = 0; h < H; ++h) {
        float* s = S[h].get();
        // This stands in for the softmax. It is one read-modify-write pass
        // with no exponential.
        for (std::size_t i = 0; i < table; ++i) s[i] *= 0.999f;
      }
      report("2 in-place pass", c0, omp_get_wtime() - t0);

      c0 = counters.read();
      t0 = omp_get_wtime();
#pragma omp parallel for
      for (int h = 0; h < H; ++h) {
        for (int q0 = 0; q0 < L; q0 += strip) {
          gemm_packed(std::min(strip, L - q0), d, L, S[h].get() + static_cast<std::size_t>(q0) * L, L, nullptr, 0,
                      O.get() + h * head + static_cast<std::size_t>(q0) * d, d, {.packed_b = Vp[h].get()});
        }
      }
      report("3 read scores", c0, omp_get_wtime() - t0);
    }
  }
  return 0;
}
