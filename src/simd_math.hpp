// AVX2 helpers shared by the attention and decode kernels.
#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "util.hpp"

// exp(x) for x <= 0. The relative error is about 2e-7.
// The constants are those of expf in the Cephes library.
// S. L. Moshier, "Cephes mathematical library", file single/expf.c, 2000.
// https://www.netlib.org/cephes (accessed 12 Oct 2026). The code is written new.
// Inputs below -87 are clamped. This keeps the result a normal float.
inline __m256 exp256(__m256 x) {
  x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
  const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  const __m256i pow2 = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2));
}

inline float hmax(__m256 v) {
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)));
}

inline float row_max(const float* s, int n) {
  float mx = -INFINITY;
  int j = 0;
  if (n >= 8) {
    __m256 v = _mm256_loadu_ps(s);
    for (j = 8; j + 8 <= n; j += 8) v = _mm256_max_ps(v, _mm256_loadu_ps(s + j));
    mx = hmax(v);
  }
  for (; j < n; ++j) mx = std::max(mx, s[j]);
  return mx;
}

// Sets s[j] = exp(s[j] * scale - shift) in place. Returns the sum of the results.
inline float row_exp(float* s, int n, float scale, float shift) {
  const __m256 vscale = _mm256_set1_ps(scale), vshift = _mm256_set1_ps(shift);
  __m256 vsum = _mm256_setzero_ps();
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256 e = exp256(_mm256_fmsub_ps(_mm256_loadu_ps(s + j), vscale, vshift));
    _mm256_storeu_ps(s + j, e);
    vsum = _mm256_add_ps(vsum, e);
  }
  float sum = hsum(vsum);
  for (; j < n; ++j) {
    s[j] = std::exp(s[j] * scale - shift);
    sum += s[j];
  }
  return sum;
}

inline void row_scale(float* x, int n, float f) {
  const __m256 vf = _mm256_set1_ps(f);
  int j = 0;
  for (; j + 8 <= n; j += 8) _mm256_storeu_ps(x + j, _mm256_mul_ps(_mm256_loadu_ps(x + j), vf));
  for (; j < n; ++j) x[j] *= f;
}
