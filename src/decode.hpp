// Decode-step attention. One new query per head runs against L cached keys
// and values.
#pragma once

#include <span>
#include <string_view>

// q and o are H x d. K and V are H x L x d. Row-major FP32.
using DecodeFn = void (*)(int H, int L, int d, const float* q, const float* K,
                          const float* V, float* o);

// Reference used as ground truth. Scalar, sums in double.
void decode_ref(int H, int L, int d, const float* q, const float* K,
                const float* V, float* o);

// Scalar loops. First all scores, then softmax, then the weighted sum.
void decode_naive(int H, int L, int d, const float* q, const float* K,
                  const float* V, float* o);
// The same three passes with AVX2.
void decode_simd(int H, int L, int d, const float* q, const float* K,
                 const float* V, float* o);

struct DecodeKernel {
  const char* name;
  DecodeFn fn;
};
std::span<const DecodeKernel> decode_kernels();
DecodeFn find_decode(std::string_view name);
