// Attention kernels. O[h] = softmax(Q[h] K[h]^T / sqrt(d)) V[h] for H heads.
#pragma once

#include <span>
#include <string_view>

// Q, K, V and O are H x L x d, row-major FP32.
using AttnFn = void (*)(int H, int L, int d, const float* Q, const float* K,
                        const float* V, float* O);

// Reference used as ground truth. Scalar, sums in double, one thread.
void attn_ref(int H, int L, int d, const float* Q, const float* K,
              const float* V, float* O);

// Scalar loops over a full L x L score matrix.
void attn_naive(int H, int L, int d, const float* Q, const float* K,
                const float* V, float* O);
// Full L x L scores with packed GEMM and vectorised softmax.
void attn_mat(int H, int L, int d, const float* Q, const float* K,
              const float* V, float* O);
// Like attn_mat but writes the scores with non-temporal stores.
void attn_mat_stream(int H, int L, int d, const float* Q, const float* K,
                     const float* V, float* O);

// Tile sizes in rows.
struct AttnParams {
  int qb = 0;  // 0 means as many rows as keep the score tile within kScoreTileFloats
  int tq = 192;
  int tk = 256;
};
inline constexpr int kScoreTileFloats = 64 * 1024;  // 256 KiB, half of L2
AttnParams& attn_params();
bool attn_set_param(std::string_view name, int value);

struct AttnKernel {
  const char* name;
  AttnFn fn;
};
std::span<const AttnKernel> attn_kernels();
AttnFn find_attn(std::string_view name);
