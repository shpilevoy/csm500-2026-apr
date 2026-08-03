// Quick timing of one GEMM kernel. Prints GFLOP/s.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gemm.hpp"
#include "util.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: bench naive|reorder N [reps]\n");
    return 2;
  }
  const int n = std::atoi(argv[2]);
  const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
  GemmFn fn = nullptr;
  if (!std::strcmp(argv[1], "naive")) fn = gemm_naive;
  if (!std::strcmp(argv[1], "reorder")) fn = gemm_reorder;
  if (!fn || n <= 0) return 2;

  const std::size_t nn = static_cast<std::size_t>(n) * n;
  auto A = alloc_floats(nn), B = alloc_floats(nn), C = alloc_floats(nn);
  fill_uniform(A.get(), nn, 1);
  fill_uniform(B.get(), nn, 2);

  double best = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    fn(n, n, n, A.get(), n, B.get(), n, C.get(), n);
    const auto t1 = std::chrono::steady_clock::now();
    const double s = std::chrono::duration<double>(t1 - t0).count();
    const double gflops = 2.0 * n * n * n / s / 1e9;
    if (gflops > best) best = gflops;
    std::printf("%s n=%d rep=%d %.3f s %.2f GFLOP/s\n", argv[1], n, r, s, gflops);
  }
  std::printf("best %.2f GFLOP/s\n", best);
  return 0;
}
