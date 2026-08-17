// The list of GEMM kernels and their block size settings.
#include "gemm.hpp"

namespace {
constexpr GemmKernel kKernels[] = {
    {"naive", gemm_naive},     {"reorder", gemm_reorder},
    {"blocked", gemm_blocked}, {"simd", gemm_simd},
    {"goto", gemm_goto},
};
}  // namespace

std::span<const GemmKernel> gemm_kernels() { return kKernels; }

GemmFn find_gemm(std::string_view name) {
  for (const auto& k : kKernels) {
    if (name == k.name) return k.fn;
  }
  return nullptr;
}

GemmParams& gemm_params() {
  static GemmParams params;
  return params;
}

bool gemm_set_param(std::string_view name, int value) {
  if (value <= 0) return false;
  GemmParams& p = gemm_params();
  if (name == "bk") p.bk = value;
  else if (name == "bn") p.bn = value;
  else if (name == "mc") p.mc = value;
  else if (name == "kc") p.kc = value;
  else if (name == "nc") p.nc = value;
  else return false;
  return true;
}
