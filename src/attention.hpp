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
