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

void decode_fused(int H, int L, int d, const float* q, const float* K,
                  const float* V, float* o) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(static)
  for (int h = 0; h < H; ++h) {
    float s[kBlock];
    float* out = o + h * d;
    std::memset(out, 0, d * sizeof(float));
    float run_max = -INFINITY, run_sum = 0.0f;
    for (int j0 = 0; j0 < L; j0 += kBlock) {
      const int n = std::min(kBlock, L - j0);
      const std::size_t offset = (static_cast<std::size_t>(h) * L + j0) * d;
      scores(K + offset, q + h * d, n, d, s);
      const float new_max = std::max(run_max, row_max(s, n) * scale);
      // Online softmax. Rescale the totals so far to the new maximum.
      if (j0 > 0 && new_max != run_max) {
        const float alpha = std::exp(run_max - new_max);
        run_sum *= alpha;
        row_scale(out, d, alpha);
      }
      run_max = new_max;
      run_sum += row_exp(s, n, scale, new_max);
      weighted_sum(V + offset, s, n, d, out);
    }
    row_scale(out, d, 1.0f / run_sum);
  }
}

void to_f16(const float* in, std::size_t n, std::uint16_t* out) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(in + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), h);
  }
  for (; i < n; ++i) out[i] = _cvtss_sh(in[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

void from_f16(const std::uint16_t* in, std::size_t n, float* out) {
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i))));
  }
  for (; i < n; ++i) out[i] = _cvtsh_ss(in[i]);
}

namespace {

inline __m256 load_f16(const std::uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

// Like scores() for half precision keys.
void scores_f16(const std::uint16_t* K, const float* q, int n, int d, float* s) {
  const int d8 = d / 8 * 8;
  int j = 0;
  for (; j + 4 <= n; j += 4) {
    const std::uint16_t* k = K + static_cast<std::size_t>(j) * d;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int c = 0; c < d8; c += 8) {
      const __m256 x = _mm256_loadu_ps(q + c);
      a0 = _mm256_fmadd_ps(load_f16(k + c), x, a0);
      a1 = _mm256_fmadd_ps(load_f16(k + d + c), x, a1);
      a2 = _mm256_fmadd_ps(load_f16(k + 2 * d + c), x, a2);
      a3 = _mm256_fmadd_ps(load_f16(k + 3 * d + c), x, a3);
    }
    s[j] = hsum(a0), s[j + 1] = hsum(a1), s[j + 2] = hsum(a2), s[j + 3] = hsum(a3);
    for (int c = d8; c < d; ++c) {
      for (int t = 0; t < 4; ++t) s[j + t] += _cvtsh_ss(k[t * d + c]) * q[c];
    }
  }
  for (; j < n; ++j) {
    const std::uint16_t* k = K + static_cast<std::size_t>(j) * d;
    float dot = 0.0f;
    for (int c = 0; c < d; ++c) dot += _cvtsh_ss(k[c]) * q[c];
    s[j] = dot;
  }
}

// Like weighted_sum() for half precision values.
void weighted_sum_f16(const std::uint16_t* V, const float* p, int n, int d, float* out) {
  const int d8 = d / 8 * 8;
  for (int j = 0; j < n; ++j) {
    const std::uint16_t* v = V + static_cast<std::size_t>(j) * d;
    const __m256 w = _mm256_set1_ps(p[j]);
    for (int c = 0; c < d8; c += 8) {
      _mm256_storeu_ps(out + c, _mm256_fmadd_ps(w, load_f16(v + c), _mm256_loadu_ps(out + c)));
    }
    for (int c = d8; c < d; ++c) out[c] += p[j] * _cvtsh_ss(v[c]);
  }
}

}  // namespace

void decode_fused_f16(int H, int L, int d, const float* q, const std::uint16_t* K,
                      const std::uint16_t* V, float* o) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(static)
  for (int h = 0; h < H; ++h) {
    float s[kBlock];
    float* out = o + h * d;
    std::memset(out, 0, d * sizeof(float));
    float run_max = -INFINITY, run_sum = 0.0f;
    for (int j0 = 0; j0 < L; j0 += kBlock) {
      const int n = std::min(kBlock, L - j0);
      const std::size_t offset = (static_cast<std::size_t>(h) * L + j0) * d;
      scores_f16(K + offset, q + h * d, n, d, s);
      const float new_max = std::max(run_max, row_max(s, n) * scale);
      if (j0 > 0 && new_max != run_max) {
        const float alpha = std::exp(run_max - new_max);
        run_sum *= alpha;
        row_scale(out, d, alpha);
      }
      run_max = new_max;
      run_sum += row_exp(s, n, scale, new_max);
      weighted_sum_f16(V + offset, s, n, d, out);
    }
    row_scale(out, d, 1.0f / run_sum);
  }
}

namespace {
constexpr DecodeKernel kKernels[] = {
    {"naive", decode_naive},
    {"simd", decode_simd},
    {"fused", decode_fused},
};
}  // namespace

std::span<const DecodeKernel> decode_kernels() { return kKernels; }

DecodeFn find_decode(std::string_view name) {
  for (const auto& k : kKernels) {
    if (name == k.name) return k.fn;
  }
  return nullptr;
}
