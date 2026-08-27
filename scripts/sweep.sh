#!/usr/bin/env bash
# Runs every kernel over a few sizes and appends to one CSV.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/sweep.csv}
mkdir -p "$(dirname "$OUT")"

for size in 256 512 1024 2048; do
  for kernel in naive reorder blocked simd goto; do
    # naive at 2048 takes minutes
    if [ "$kernel" = naive ] && [ "$size" -gt 1024 ]; then continue; fi
    build/bench_gemm --kernel $kernel --m $size --n $size --k $size --out "$OUT"
  done
  for threads in 2 4 8; do
    OMP_NUM_THREADS=$threads build/bench_gemm --kernel goto_mt --m $size --n $size --k $size --out "$OUT"
  done
done
