// Measures the roofline ceilings of the machine. fma gives the compute peak and
// triad the bandwidth. sum, update and fill have known memory traffic. They
// check the counter estimate.
#include <immintrin.h>
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "util.hpp"

namespace {

void triad(float* a, const float* b, const float* c, long n, float s) {
  const __m256 vs = _mm256_set1_ps(s);
  for (long i = 0; i < n; i += 8) {
    _mm256_store_ps(a + i, _mm256_fmadd_ps(vs, _mm256_load_ps(c + i), _mm256_load_ps(b + i)));
  }
}

float sum(const float* a, long n) {
  __m256 acc = _mm256_setzero_ps();
  for (long i = 0; i < n; i += 8) acc = _mm256_add_ps(acc, _mm256_load_ps(a + i));
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  return lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

void update(float* a, long n, float s) {
  const __m256 vs = _mm256_set1_ps(s);
  for (long i = 0; i < n; i += 8) _mm256_store_ps(a + i, _mm256_mul_ps(vs, _mm256_load_ps(a + i)));
}

void fill(float* a, long n, float s) {
  const __m256 vs = _mm256_set1_ps(s);
  for (long i = 0; i < n; i += 8) _mm256_store_ps(a + i, vs);
}

// 12 independent chains, as in the GEMM microkernel. Returns a value that
// depends on every chain, so no chain can be optimised away.
float fma_chains(long iters, float seed) {
  const __m256 x = _mm256_set1_ps(seed), y = _mm256_set1_ps(1.0f - seed * 1e-3f);
  __m256 acc[12];
  for (auto& v : acc) v = _mm256_setzero_ps();
  for (long i = 0; i < iters; ++i) {
    for (auto& v : acc) v = _mm256_fmadd_ps(x, y, v);
  }
  __m256 sum = acc[0];
  for (int k = 1; k < 12; ++k) sum = _mm256_add_ps(sum, acc[k]);
  return _mm256_cvtss_f32(sum);
}

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: bench_machine --kernel triad|sum|update|fill --elems N [--reps R] [--out FILE]\n"
               "       bench_machine --kernel fma [--reps R] [--out FILE]\n"
               "  --elems  floats per array, per thread (rounded up to 8)\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel, out;
  long elems = 0;
  int reps = 30;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--kernel") kernel = next();
    else if (a == "--out") out = next();
    else if (a == "--elems") elems = std::atol(next());
    else if (a == "--reps") reps = std::atoi(next());
    else usage();
  }
  PerfCounters counters;
  const int threads = omp_get_max_threads();
  Recorder rec(out, "kernel,params,elems,passes,threads");

  if (kernel == "fma") {
    const long iters = 20'000'000;
    volatile float seed = 0.5f, sink = 0.0f;
    float total = 0.0f;
    auto run = [&] {
#pragma omp parallel reduction(+ : total)
      total += fma_chains(iters, seed);
      sink = total;
    };
    run();
    rec.run(counters, "fma,,0,0," + std::to_string(threads), reps,
            2.0 * 8 * 12 * iters * threads, -1.0, run);
  } else if ((kernel == "triad" || kernel == "sum" || kernel == "update" || kernel == "fill") && elems > 0) {
    elems = (elems + 7) / 8 * 8;
    // Enough passes per repetition to move about 256 MiB per array.
    const long passes = std::max(1L, (64L << 20) / elems);
    std::vector<FloatBuf> bufs(3 * threads);
    std::vector<float> sums(threads);
#pragma omp parallel
    {
      // Allocated and first touched by the thread that uses it.
      const int t = omp_get_thread_num();
      for (int k = 0; k < 3; ++k) {
        bufs[3 * t + k] = alloc_floats(elems);
        fill_uniform(bufs[3 * t + k].get(), elems, 3 * t + k);
      }
    }
    auto run = [&] {
#pragma omp parallel
      {
        const int t = omp_get_thread_num();
        float* a = bufs[3 * t].get();
        float* b = bufs[3 * t + 1].get();
        const float* c = bufs[3 * t + 2].get();
        // Swapping a and b each pass keeps the values bounded. It also stops
        // the compiler from merging the passes.
        float total = 0.0f;
        for (long p = 0; p < passes; ++p) {
          if (kernel == "triad") triad(a, b, c, elems, 0.5f);
          else if (kernel == "sum") total += sum(a, elems);
          else if (kernel == "update") update(a, elems, p % 2 ? 2.0f : 0.5f);
          else fill(a, elems, static_cast<float>(p));
          std::swap(a, b);
        }
        sums[t] = total;  // keeps the read-only loop observable
      }
    };
    run();
    if (kernel == "sum") std::fprintf(stderr, "checksum %g\n", sums[0]);
    rec.run(counters,
            kernel + ",," + std::to_string(elems) + "," + std::to_string(passes) + "," +
                std::to_string(threads),
            reps, 2.0 * elems * passes * threads, -1.0, run);
  } else {
    usage();
  }
  return 0;
}
