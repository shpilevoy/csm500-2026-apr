// Checks every GEMM kernel against the reference implementation.
#include <cstdio>

#include "gemm.hpp"
#include "util.hpp"

int main() {
  // Odd and non-square shapes on purpose. They test the edge handling that
  // blocked and vectorised kernels get wrong first.
  const int shapes[][3] = {{1, 1, 1},   {1, 7, 5},    {7, 1, 5},    {5, 7, 1},
                           {8, 8, 8},   {17, 31, 13}, {64, 64, 64}, {100, 3, 257},
                           {3, 100, 257}, {129, 65, 33}, {255, 257, 256}, {150, 300, 260}};
  const double tol = 1e-4;
  int failures = 0;
  // The second pass uses tiny blocks. It hits every block boundary and edge tile.
  for (const bool tiny_blocks : {false, true}) {
    if (tiny_blocks) {
      for (const char* name : {"bk", "bn"}) gemm_set_param(name, 5);
      gemm_set_param("mc", 12);
      gemm_set_param("kc", 7);
      gemm_set_param("nc", 32);
    }
    for (const auto& s : shapes) {
      const int M = s[0], N = s[1], K = s[2];
      auto A = alloc_floats(static_cast<std::size_t>(M) * K);
      auto B = alloc_floats(static_cast<std::size_t>(K) * N);
      auto C = alloc_floats(static_cast<std::size_t>(M) * N);
      auto R = alloc_floats(static_cast<std::size_t>(M) * N);
      fill_uniform(A.get(), static_cast<std::size_t>(M) * K, 1);
      fill_uniform(B.get(), static_cast<std::size_t>(K) * N, 2);
      gemm_ref(M, N, K, A.get(), K, B.get(), N, R.get(), N);
      for (const auto& k : gemm_kernels()) {
        fill_uniform(C.get(), static_cast<std::size_t>(M) * N, 3);
        k.fn(M, N, K, A.get(), K, B.get(), N, C.get(), N);
        const double e = rel_err(C.get(), R.get(), static_cast<std::size_t>(M) * N);
        const bool ok = e < tol;
        failures += !ok;
        std::printf("%-4s %-8s %s %4dx%4dx%4d rel_err=%.2e\n", ok ? "ok" : "FAIL",
                    k.name, tiny_blocks ? "tiny" : "dflt", M, K, N, e);
      }
    }
  }
  return failures != 0;
}
