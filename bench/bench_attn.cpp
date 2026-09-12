// Times one attention point. A point is a kernel, a shape and a thread count.
#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "attention.hpp"
#include "bench_common.hpp"
#include "gemm.hpp"
#include "util.hpp"

namespace {

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: bench_attn --kernel NAME --heads H --len L --dim D\n"
               "                  [--reps R] [--warmup W] [--set PARAM=VALUE]...\n"
               "                  [--check] [--out FILE]\n"
               "  --set       override a tile size (qb tq tk, or GEMM mc kc nc)\n"
               "  --check     also report the error against the reference implementation\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel, out, params;
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
    else if (a == "--set") {
      const std::string kv = next();
      const auto eq = kv.find('=');
      if (eq == std::string::npos) usage();
      const std::string name = kv.substr(0, eq);
      const int value = std::atoi(kv.c_str() + eq + 1);
      if (!attn_set_param(name, value) && !gemm_set_param(name, value)) usage();
      params += (params.empty() ? "" : ";") + kv;
    } else usage();
  }
  if (kernel.empty() || H <= 0 || L <= 0 || d <= 0 || reps <= 0 || warmup < 0) {
    usage();
  }

  // Opened before any thread exists so worker threads inherit the counters.
  PerfCounters counters;

  const AttnFn fn = find_attn(kernel);
  if (!fn) {
    std::fprintf(stderr, "unknown kernel '%s'; available:", kernel.c_str());
    for (const auto& k : attn_kernels()) std::fprintf(stderr, " %s", k.name);
    std::fprintf(stderr, "\n");
    return 1;
  }

  const std::size_t n = static_cast<std::size_t>(H) * L * d;
  auto Q = alloc_floats(n), K = alloc_floats(n), V = alloc_floats(n), O = alloc_floats(n);
  fill_uniform(Q.get(), n, 1);
  fill_uniform(K.get(), n, 2);
  fill_uniform(V.get(), n, 3);
  // Scores are a few units wide, so the softmax is not flat and not one-hot.
  for (std::size_t i = 0; i < n; ++i) Q[i] *= 4.0f;
  fill_uniform(O.get(), n, 4);

  auto run = [&] { fn(H, L, d, Q.get(), K.get(), V.get(), O.get()); };
  for (int r = 0; r < warmup; ++r) run();

  double err = -1.0;
  if (check) {
    if (warmup == 0) run();
    auto R = alloc_floats(n);
    attn_ref(H, L, d, Q.get(), K.get(), V.get(), R.get());
    err = rel_err(O.get(), R.get(), n);
  }

  // By convention only the two matrix products count as FLOPs, with 2*d per
  // score each. The softmax does not count.
  const double scores = 1.0 * L * L;
  const double flops = 4.0 * H * scores * d;
  Recorder rec(out, "kernel,params,heads,len,dim,threads");
  const std::string values = kernel + "," + params + "," + std::to_string(H) + "," +
                             std::to_string(L) + "," + std::to_string(d) + "," +
                             std::to_string(omp_get_max_threads());
  rec.run(counters, values, reps, flops, err, run);
  return 0;
}
