// The list of GEMM kernels.
#include "gemm.hpp"

namespace {
constexpr GemmKernel kKernels[] = {
    {"naive", gemm_naive},     {"reorder", gemm_reorder},
    {"blocked", gemm_blocked}, {"simd", gemm_simd},
};
}  // namespace

std::span<const GemmKernel> gemm_kernels() { return kKernels; }

GemmFn find_gemm(std::string_view name) {
  for (const auto& k : kKernels) {
    if (name == k.name) return k.fn;
  }
  return nullptr;
}
