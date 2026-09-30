// Scalar decode-step attention. The reference implementation and the naive
// kernel.
#include <algorithm>
#include <cmath>
#include <vector>

#include "decode.hpp"

void decode_ref(int H, int L, int d, const float* q, const float* K,
                const float* V, float* o) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(d));
  std::vector<double> s(L), acc(d);
  for (int h = 0; h < H; ++h) {
    const float* k = K + static_cast<std::size_t>(h) * L * d;
    const float* v = V + static_cast<std::size_t>(h) * L * d;
    double mx = -INFINITY;
    for (int j = 0; j < L; ++j) {
      double dot = 0.0;
      for (int c = 0; c < d; ++c) dot += static_cast<double>(k[static_cast<std::size_t>(j) * d + c]) * q[h * d + c];
      s[j] = dot * scale;
      mx = std::max(mx, s[j]);
    }
    double sum = 0.0;
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int j = 0; j < L; ++j) {
      const double p = std::exp(s[j] - mx);
      sum += p;
      for (int c = 0; c < d; ++c) acc[c] += p * v[static_cast<std::size_t>(j) * d + c];
    }
    for (int c = 0; c < d; ++c) o[h * d + c] = static_cast<float>(acc[c] / sum);
  }
}

void decode_naive(int H, int L, int d, const float* q, const float* K,
                  const float* V, float* o) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
#pragma omp parallel for schedule(static)
  for (int h = 0; h < H; ++h) {
    const float* k = K + static_cast<std::size_t>(h) * L * d;
    const float* v = V + static_cast<std::size_t>(h) * L * d;
    std::vector<float> s(L);
    float mx = -INFINITY;
    for (int j = 0; j < L; ++j) {
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) dot += k[static_cast<std::size_t>(j) * d + c] * q[h * d + c];
      s[j] = dot * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < L; ++j) {
      s[j] = std::exp(s[j] - mx);
      sum += s[j];
    }
    float* out = o + h * d;
    for (int c = 0; c < d; ++c) out[c] = 0.0f;
    for (int j = 0; j < L; ++j) {
      for (int c = 0; c < d; ++c) out[c] += s[j] * v[static_cast<std::size_t>(j) * d + c];
    }
    for (int c = 0; c < d; ++c) out[c] /= sum;
  }
}
