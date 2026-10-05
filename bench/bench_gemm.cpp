// Times one GEMM point. A point is a kernel, a shape and a thread count.
#include <dlfcn.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bench_common.hpp"
#include "gemm.hpp"
#include "util.hpp"

namespace {

using CblasSgemm = void (*)(int order, int transA, int transB, int M, int N,
                            int K, float alpha, const float* A, int lda,
                            const float* B, int ldb, float beta, float* C,
                            int ldc);
CblasSgemm g_cblas_sgemm = nullptr;

void gemm_blas(int M, int N, int K, const float* A, int lda, const float* B,
               int ldb, float* C, int ldc) {
  constexpr int kRowMajor = 101, kNoTrans = 111;
  g_cblas_sgemm(kRowMajor, kNoTrans, kNoTrans, M, N, K, 1.0f, A, lda, B, ldb,
                0.0f, C, ldc);
}

// The GEMM entry point of oneDNN. Row-major with 64-bit dimensions.
using DnnlSgemm = int (*)(char transA, char transB, long M, long N, long K,
                          float alpha, const float* A, long lda, const float* B,
                          long ldb, float beta, float* C, long ldc);
DnnlSgemm g_dnnl_sgemm = nullptr;

void gemm_dnnl(int M, int N, int K, const float* A, int lda, const float* B,
               int ldb, float* C, int ldc) {
  g_dnnl_sgemm('N', 'N', M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

void* open_library(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    std::exit(1);
  }
  return handle;
}

[[noreturn]] void usage() {
  std::fprintf(stderr,
               "usage: bench_gemm --kernel NAME --m M --n N --k K [--reps R]\n"
               "                  [--warmup W] [--blas-lib PATH | --dnnl-lib PATH]\n"
               "                  [--check] [--set PARAM=VALUE]... [--out FILE]\n"
               "  --blas-lib  time cblas_sgemm from this shared library. NAME\n"
               "              is then only the label written to the CSV\n"
               "  --dnnl-lib  the same for dnnl_sgemm from a oneDNN library\n"
               "  --set       override a block size (bk bn mc kc nc)\n"
               "  --check     also report the error against the reference implementation\n");
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  std::string kernel, blas_lib, dnnl_lib, out, params;
  int M = 0, N = 0, K = 0, reps = 30, warmup = 5;
  bool check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--kernel") kernel = next();
    else if (a == "--blas-lib") blas_lib = next();
    else if (a == "--dnnl-lib") dnnl_lib = next();
    else if (a == "--out") out = next();
    else if (a == "--m") M = std::atoi(next());
    else if (a == "--n") N = std::atoi(next());
    else if (a == "--k") K = std::atoi(next());
    else if (a == "--reps") reps = std::atoi(next());
    else if (a == "--warmup") warmup = std::atoi(next());
    else if (a == "--check") check = true;
    else if (a == "--set") {
      const std::string kv = next();
      const auto eq = kv.find('=');
      if (eq == std::string::npos ||
          !gemm_set_param(kv.substr(0, eq), std::atoi(kv.c_str() + eq + 1))) {
        usage();
      }
      params += (params.empty() ? "" : ";") + kv;
    }
    else usage();
  }
  if (kernel.empty() || M <= 0 || N <= 0 || K <= 0 || reps <= 0 || warmup < 0) {
    usage();
  }

  // Opened before any thread exists so worker threads inherit the counters.
  PerfCounters counters;

  GemmFn fn = nullptr;
  if (!blas_lib.empty()) {
    g_cblas_sgemm = reinterpret_cast<CblasSgemm>(dlsym(open_library(blas_lib), "cblas_sgemm"));
    if (!g_cblas_sgemm) {
      std::fprintf(stderr, "no cblas_sgemm in %s\n", blas_lib.c_str());
      return 1;
    }
    fn = gemm_blas;
  } else if (!dnnl_lib.empty()) {
    g_dnnl_sgemm = reinterpret_cast<DnnlSgemm>(dlsym(open_library(dnnl_lib), "dnnl_sgemm"));
    if (!g_dnnl_sgemm) {
      std::fprintf(stderr, "no dnnl_sgemm in %s\n", dnnl_lib.c_str());
      return 1;
    }
    fn = gemm_dnnl;
  } else {
    fn = find_gemm(kernel);
    if (!fn) {
      std::fprintf(stderr, "unknown kernel '%s'; available:", kernel.c_str());
      for (const auto& k : gemm_kernels()) std::fprintf(stderr, " %s", k.name);
      std::fprintf(stderr, "\n");
      return 1;
    }
  }

  const std::size_t a_n = static_cast<std::size_t>(M) * K;
  const std::size_t b_n = static_cast<std::size_t>(K) * N;
  const std::size_t c_n = static_cast<std::size_t>(M) * N;
  auto A = alloc_floats(a_n), B = alloc_floats(b_n), C = alloc_floats(c_n);
  fill_uniform(A.get(), a_n, 1);
  fill_uniform(B.get(), b_n, 2);
  std::memset(C.get(), 0, c_n * sizeof(float));

  for (int r = 0; r < warmup; ++r) fn(M, N, K, A.get(), K, B.get(), N, C.get(), N);

  double err = -1.0;
  if (check) {
    auto R = alloc_floats(c_n);
    gemm_ref(M, N, K, A.get(), K, B.get(), N, R.get(), N);
    err = rel_err(C.get(), R.get(), c_n);
  }

  Recorder rec(out, "kernel,params,m,n,k,threads");
  const std::string values = kernel + "," + params + "," + std::to_string(M) + "," +
                             std::to_string(N) + "," + std::to_string(K) + "," +
                             std::to_string(omp_get_max_threads());
  rec.run(counters, values, reps, 2.0 * M * N * K, err,
          [&] { fn(M, N, K, A.get(), K, B.get(), N, C.get(), N); });
  return 0;
}
