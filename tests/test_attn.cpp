// Checks every attention kernel against the reference implementation.
#include <cstdio>

#include "attention.hpp"
#include "gemm.hpp"
#include "util.hpp"

int main() {
  const int shapes[][3] = {{1, 1, 1},   {2, 7, 5},    {3, 33, 16},
                           {2, 130, 64}, {1, 300, 48}, {9, 65, 64}, {1, 700, 32}};
  const double tol = 1e-4;
  int failures = 0;
  // Tiny tiles hit every tile boundary.
  for (const int pass : {0, 1}) {
    const bool tiny_tiles = pass == 1;
    if (tiny_tiles) {
      attn_set_param("qb", 5);
      attn_set_param("tq", 7);
      attn_set_param("tk", 9);
      gemm_set_param("mc", 12);
      gemm_set_param("kc", 7);
      gemm_set_param("nc", 32);
    }
    // Score spread: normal, near one-hot softmax, and all scores equal.
    for (const float spread : {4.0f, 60.0f, 0.0f}) {
    for (const auto& s : shapes) {
      const int H = s[0], L = s[1], d = s[2];
      const std::size_t n = static_cast<std::size_t>(H) * L * d;
      auto Q = alloc_floats(n), K = alloc_floats(n), V = alloc_floats(n);
      auto O = alloc_floats(n), R = alloc_floats(n);
      fill_uniform(Q.get(), n, 1);
      fill_uniform(K.get(), n, 2);
      fill_uniform(V.get(), n, 3);
      for (std::size_t i = 0; i < n; ++i) Q[i] *= spread;
      attn_ref(H, L, d, Q.get(), K.get(), V.get(), R.get());
      for (const auto& k : attn_kernels()) {
        fill_uniform(O.get(), n, 4);
        k.fn(H, L, d, Q.get(), K.get(), V.get(), O.get());
        const double e = rel_err(O.get(), R.get(), n);
        const bool ok = e < tol;
        failures += !ok;
        std::printf("%-4s %-10s %s spread=%g H=%d L=%d d=%d rel_err=%.2e\n", ok ? "ok" : "FAIL", k.name,
                    tiny_tiles ? "tiny" : "dflt", spread, H, L, d, e);
      }
    }
    }
  }
  return failures != 0;
}
