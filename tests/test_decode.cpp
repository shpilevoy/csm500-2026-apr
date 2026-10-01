// Checks every decode kernel against the reference implementation.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "decode.hpp"
#include "util.hpp"

int main() {
  // The lengths are on both sides of the 256-key block of the fused kernel.
  // Some head dimensions are not multiples of the SIMD width.
  const int shapes[][3] = {{1, 1, 1},    {2, 7, 5},     {3, 255, 64}, {2, 256, 64},
                           {2, 257, 64}, {1, 1000, 48}, {9, 513, 33}, {4, 4096, 128}};
  const double tol = 1e-4;
  int failures = 0;
  // Score spread: normal, near one-hot softmax, and all scores equal.
  for (const float spread : {4.0f, 60.0f, 0.0f}) {
  for (const auto& s : shapes) {
    const int H = s[0], L = s[1], d = s[2];
    const std::size_t n = static_cast<std::size_t>(H) * L * d, m = static_cast<std::size_t>(H) * d;
    auto q = alloc_floats(m), K = alloc_floats(n), V = alloc_floats(n), o = alloc_floats(m), r = alloc_floats(m);
    fill_uniform(q.get(), m, 1);
    fill_uniform(K.get(), n, 2);
    fill_uniform(V.get(), n, 3);
    // Scores are a few units wide, so the softmax is not flat and not one-hot.
    for (std::size_t i = 0; i < m; ++i) q[i] *= spread;
    decode_ref(H, L, d, q.get(), K.get(), V.get(), r.get());
    for (const auto& k : decode_kernels()) {
      fill_uniform(o.get(), m, 4);
      k.fn(H, L, d, q.get(), K.get(), V.get(), o.get());
      const double e = rel_err(o.get(), r.get(), m);
      const bool ok = e < tol;
      failures += !ok;
      std::printf("%-4s %-6s spread=%g H=%d L=%d d=%d rel_err=%.2e\n", ok ? "ok" : "FAIL", k.name, spread, H, L, d, e);
    }
    // The reference runs on the rounded data. So the test checks the kernel and
    // not the rounding.
    std::vector<std::uint16_t> K16(n), V16(n);
    to_f16(K.get(), n, K16.data());
    to_f16(V.get(), n, V16.data());
    from_f16(K16.data(), n, K.get());
    from_f16(V16.data(), n, V.get());
    decode_ref(H, L, d, q.get(), K.get(), V.get(), r.get());
    fill_uniform(o.get(), m, 4);
    decode_fused_f16(H, L, d, q.get(), K16.data(), V16.data(), o.get());
    const double e = rel_err(o.get(), r.get(), m);
    failures += !(e < tol);
    std::printf("%-4s %-9s spread=%g H=%d L=%d d=%d rel_err=%.2e\n", e < tol ? "ok" : "FAIL", "fused_f16", spread, H, L, d,
                e);
  }
  }
  return failures != 0;
}
