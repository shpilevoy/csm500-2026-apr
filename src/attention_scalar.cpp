// Scalar attention. The reference implementation and the naive kernel.
#include <algorithm>
#include <cmath>
#include <vector>

#include "attention.hpp"

void attn_ref(int H, int L, int d, const float* Q, const float* K,
              const float* V, float* O) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  std::vector<double> s(L), o(d);
  for (int h = 0; h < H; ++h) {
    const float* q = Q + static_cast<std::size_t>(h) * L * d;
    const float* k = K + static_cast<std::size_t>(h) * L * d;
    const float* v = V + static_cast<std::size_t>(h) * L * d;
    float* out = O + static_cast<std::size_t>(h) * L * d;
    for (int i = 0; i < L; ++i) {
      const int n = L;
      double mx = -INFINITY;
      for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int c = 0; c < d; ++c) acc += static_cast<double>(q[i * d + c]) * k[j * d + c];
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
      }
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        s[j] = std::exp(s[j] - mx);
        sum += s[j];
      }
      std::fill(o.begin(), o.end(), 0.0);
      for (int j = 0; j < n; ++j) {
        for (int c = 0; c < d; ++c) o[c] += s[j] * v[j * d + c];
      }
      for (int c = 0; c < d; ++c) out[i * d + c] = static_cast<float>(o[c] / sum);
    }
  }
}

void attn_naive(int H, int L, int d, const float* Q, const float* K,
                const float* V, float* O) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(static)
  for (int h = 0; h < H; ++h) {
    const float* q = Q + static_cast<std::size_t>(h) * L * d;
    const float* k = K + static_cast<std::size_t>(h) * L * d;
    const float* v = V + static_cast<std::size_t>(h) * L * d;
    float* out = O + static_cast<std::size_t>(h) * L * d;
    std::vector<float> S(static_cast<std::size_t>(L) * L);
    for (int i = 0; i < L; ++i) {
      for (int j = 0; j < L; ++j) {
        float acc = 0.0f;
        for (int c = 0; c < d; ++c) acc += q[i * d + c] * k[j * d + c];
        S[static_cast<std::size_t>(i) * L + j] = acc * scale;
      }
    }
    for (int i = 0; i < L; ++i) {
      const int n = L;
      float* s = &S[static_cast<std::size_t>(i) * L];
      float mx = s[0];
      for (int j = 1; j < n; ++j) mx = std::max(mx, s[j]);
      float sum = 0.0f;
      for (int j = 0; j < n; ++j) {
        s[j] = std::exp(s[j] - mx);
        sum += s[j];
      }
      const float inv = 1.0f / sum;
      for (int j = 0; j < n; ++j) s[j] *= inv;
    }
    for (int i = 0; i < L; ++i) {
      const int n = L;
      const float* s = &S[static_cast<std::size_t>(i) * L];
      for (int c = 0; c < d; ++c) out[i * d + c] = 0.0f;
      for (int j = 0; j < n; ++j) {
        for (int c = 0; c < d; ++c) out[i * d + c] += s[j] * v[j * d + c];
      }
    }
  }
}
