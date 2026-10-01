// Decode-step attention. One new query per head runs against L cached keys
// and values.
#pragma once

#include <cstddef>
#include <cstdint>
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
// One pass over blocks of keys and values with online softmax.
void decode_fused(int H, int L, int d, const float* q, const float* K,
                  const float* V, float* o);

// Keys and values are stored as 16 bit floats. This halves the bytes read.
using DecodeF16Fn = void (*)(int H, int L, int d, const float* q, const std::uint16_t* K,
                             const std::uint16_t* V, float* o);

void to_f16(const float* in, std::size_t n, std::uint16_t* out);
void from_f16(const std::uint16_t* in, std::size_t n, float* out);

// Like decode_fused but reads half precision keys and values.
void decode_fused_f16(int H, int L, int d, const float* q, const std::uint16_t* K,
                      const std::uint16_t* V, float* o);

struct DecodeKernel {
  const char* name;
  DecodeFn fn;
};
std::span<const DecodeKernel> decode_kernels();
DecodeFn find_decode(std::string_view name);
