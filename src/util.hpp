// Small helpers for aligned buffers, random data and error measurement.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

struct FreeDeleter {
  void operator()(void* p) const { std::free(p); }
};
using FloatBuf = std::unique_ptr<float[], FreeDeleter>;

// Allocates n floats aligned to 64 bytes. aligned_alloc needs a size that is a
// multiple of the alignment, so the size is rounded up.
inline FloatBuf alloc_floats(std::size_t n) {
  const std::size_t bytes = (n * sizeof(float) + 63) / 64 * 64;
  void* p = std::aligned_alloc(64, std::max<std::size_t>(bytes, 64));
  if (!p) {
    std::fprintf(stderr, "out of memory: %zu bytes\n", bytes);
    std::abort();
  }
  return FloatBuf(static_cast<float*>(p));
}

inline void fill_uniform(float* x, std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (std::size_t i = 0; i < n; ++i) x[i] = dist(rng);
}

// Largest absolute error divided by the largest reference magnitude.
inline double rel_err(const float* got, const float* ref, std::size_t n) {
  double err = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    err = std::max(err, std::abs(static_cast<double>(got[i]) - ref[i]));
    scale = std::max(scale, std::abs(static_cast<double>(ref[i])));
  }
  return scale > 0.0 ? err / scale : err;
}
