// Times one point of decode-step attention. A point is a kernel, a shape and
// a thread count.
#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "decode.hpp"
#include "util.hpp"

namespace {

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: bench_decode --kernel NAME --heads H --len L --dim D\n"
               "                    [--reps R] [--warmup W] [--check] [--out FILE]\n"
               "  --len       number of cached keys and values per head\n"
               "  --check     also report the error against the reference implementation\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel, out;
  int H = 0, L = 0, d = 0, reps = 30, warmup = 5;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--kernel") kernel = next();
    else if (a == "--out") out = next();
    else if (a == "--heads") H = std::atoi(next());
    else if (a == "--len") L = std::atoi(next());
    else if (a == "--dim") d = std::atoi(next());
    else if (a == "--reps") reps = std::atoi(next());
    else if (a == "--warmup") warmup = std::atoi(next());
    else if (a == "--check") check = true;
    else usage();
  }
  if (kernel.empty() || H <= 0 || L <= 0 || d <= 0 || reps <= 0 || warmup < 0) usage();

  // Opened before any thread exists so worker threads inherit the counters.
  PerfCounters counters;

  const bool half = kernel == "fused_f16";
  const DecodeFn fn = find_decode(kernel);
  if (!fn && !half) {
    std::fprintf(stderr, "unknown kernel '%s'; available: fused_f16", kernel.c_str());
    for (const auto& k : decode_kernels()) std::fprintf(stderr, " %s", k.name);
    std::fprintf(stderr, "\n");
    return 1;
  }

  const std::size_t n = static_cast<std::size_t>(H) * L * d, m = static_cast<std::size_t>(H) * d;
  auto q = alloc_floats(m), K = alloc_floats(n), V = alloc_floats(n), o = alloc_floats(m);
  fill_uniform(q.get(), m, 1);
  fill_uniform(K.get(), n, 2);
  fill_uniform(V.get(), n, 3);
  for (std::size_t i = 0; i < m; ++i) q[i] *= 4.0f;

  std::vector<std::uint16_t> K16, V16;
  if (half) {
    K16.resize(n);
    V16.resize(n);
    to_f16(K.get(), n, K16.data());
    to_f16(V.get(), n, V16.data());
  }
  auto run = [&] {
    if (half) decode_fused_f16(H, L, d, q.get(), K16.data(), V16.data(), o.get());
    else fn(H, L, d, q.get(), K.get(), V.get(), o.get());
  };
  for (int r = 0; r < warmup; ++r) run();

  double err = -1.0;
  if (check) {
    if (warmup == 0) run();
    if (half) {
      // Compare against the reference on the rounded data.
      from_f16(K16.data(), n, K.get());
      from_f16(V16.data(), n, V.get());
    }
    auto ref = alloc_floats(m);
    decode_ref(H, L, d, q.get(), K.get(), V.get(), ref.get());
    err = rel_err(o.get(), ref.get(), m);
  }
  if (half) {
    // Free the 32 bit copies so that they cannot sit in cache during timing.
    K.reset();
    V.reset();
  }

  // 2*d operations per key for the score and 2*d per value for the weighted
  // sum. The softmax is not counted, as for full attention.
  const double flops = 4.0 * H * L * d;
  Recorder rec(out, "kernel,params,heads,len,dim,threads");
  const std::string values = kernel + ",," + std::to_string(H) + "," + std::to_string(L) + "," +
                             std::to_string(d) + "," + std::to_string(omp_get_max_threads());
  rec.run(counters, values, reps, flops, err, run);
  return 0;
}
