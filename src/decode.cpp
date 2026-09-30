// Decode-step attention kernels with AVX2.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "decode.hpp"
#include "simd_math.hpp"

namespace {

// With 64 floats per key, 256 keys are 64 KiB. A block of K and its block of
// V then fit together in the L2 cache.
constexpr int kBlock = 256;

// Sets s[j] = K[j] . q for n keys. Four keys at a time, each with its own
// running total. This keeps both FMA units busy.
void scores(const float* K, const float* q, int n, int d, float* s) {
  const int d8 = d / 8 * 8;
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const float* k = K + static_cast<std::size_t>(j) * d;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int c = 0; c < d8; c += 8) {
      const __m256 x = _mm256_loadu_ps(q + c);
      a0 = _mm256_fmadd_ps(_mm256_loadu_ps(k + c), x, a0);
      a1 = _mm256_fmadd_ps(_mm256_loadu_ps(k + d + c), x, a1);
      a2 = _mm256_fmadd_ps(_mm256_loadu_ps(k + 2 * d + c), x, a2);
      a3 = _mm256_fmadd_ps(_mm256_loadu_ps(k + 3 * d + c), x, a3);
    }
    s[j] = hsum(a0), s[j + 1] = hsum(a1), s[j + 2] = hsum(a2), s[j + 3] = hsum(a3);
    for (int c = d8; c < d; ++c) {
      for (int t = 0; t < 4; ++t) s[j + t] += k[t * d + c] * q[c];
    }
  }
  for (; j < n; ++j) {
    const float* k = K + static_cast<std::size_t>(j) * d;
    float dot = 0.0f;
    for (int c = 0; c < d; ++c) dot += k[c] * q[c];
    s[j] = dot;
  }
}

// Adds sum_j p[j] * V[j] to out for n values.
void weighted_sum(const float* V, const float* p, int n, int d, float* out) {
  const int d8 = d / 8 * 8;
  for (int j = 0; j < n; ++j) {
    const float* v = V + static_cast<std::size_t>(j) * d;
    const __m256 w = _mm256_set1_ps(p[j]);
    for (int c = 0; c < d8; c += 8) {
      _mm256_storeu_ps(out + c, _mm256_fmadd_ps(w, _mm256_loadu_ps(v + c), _mm256_loadu_ps(out + c)));
    }
    for (int c = d8; c < d; ++c) out[c] += p[j] * v[c];
  }
}

}  // namespace

void decode_simd(int H, int L, int d, const float* q, const float* K,
                 const float* V, float* o) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(static)
  for (int h = 0; h < H; ++h) {
    static thread_local Scratch scratch;
    float* const s = scratch.get(L);
    float* out = o + h * d;
    scores(K + static_cast<std::size_t>(h) * L * d, q + h * d, L, d, s);
    const float sum = row_exp(s, L, scale, row_max(s, L) * scale);
    std::memset(out, 0, d * sizeof(float));
    weighted_sum(V + static_cast<std::size_t>(h) * L * d, s, L, d, out);
    row_scale(out, d, 1.0f / sum);
  }
}

namespace {
constexpr DecodeKernel kKernels[] = {
    {"naive", decode_naive},
    {"simd", decode_simd},
};
}  // namespace

std::span<const DecodeKernel> decode_kernels() { return kKernels; }

DecodeFn find_decode(std::string_view name) {
  for (const auto& k : kKernels) {
    if (name == k.name) return k.fn;
  }
  return nullptr;
}
