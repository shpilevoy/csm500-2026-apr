#!/usr/bin/env bash
# Short live demo for the video. Each step prints one readable line per kernel.
# Usage: scripts/demo.sh tests|gemm|attention|decode|all
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_PROC_BIND=true
CORES="{0},{2},{4},{6},{8},{10},{12},{14}"

# Runs one benchmark and prints the median GFLOP/s of its repetitions.
show() {
  local label=$1 threads=$2
  shift 2
  local median
  median=$(OMP_NUM_THREADS=$threads OMP_PLACES=$(echo "$CORES" | cut -d, -f1-"$threads") "$@" |
    tail -n +2 | cut -d, -f9 | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')
  printf '  %-34s %8.1f GFLOP/s\n' "$label" "$median"
}

tests() {
  echo "== Tests: every kernel against its reference implementation"
  ctest --test-dir build 2>&1 | tail -8
}

gemm() {
  echo "== Matrix multiplication 1024 x 1024 x 1024 on one thread"
  local size="--m 1024 --n 1024 --k 1024"
  show "naive loops" 1 build/bench_gemm --kernel naive $size --reps 1 --warmup 0
  show "AVX2 with cache blocking" 1 build/bench_gemm --kernel simd $size --reps 5
  show "packed microkernel (this work)" 1 build/bench_gemm --kernel goto $size --reps 9
  show "OpenBLAS" 1 build/bench_gemm --kernel openblas $size --reps 9 \
    --blas-lib /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0
}

attention() {
  echo "== Attention, 8 heads x 4096 tokens x 64 dims, 8 threads"
  local shape="--heads 8 --len 4096 --dim 64 --reps 5 --warmup 2"
  show "builds the full score table" 8 build/bench_attn --kernel mat_stream $shape
  show "tiled with online softmax" 8 build/bench_attn --kernel tiled $shape
  local out
  out=$(mktemp)
  OMP_NUM_THREADS=8 OMP_PLACES=$CORES .venv/bin/python scripts/bench_torch_attn.py $shape --out "$out"
  printf '  %-34s %8.1f GFLOP/s\n' "PyTorch built-in" \
    "$(tail -n +2 "$out" | cut -d, -f9 | sort -n | awk '{v[NR]=$1} END{print v[int((NR+1)/2)]}')"
  rm -f "$out"
}

decode() {
  echo "== Decode step against 65536 cached tokens, 8 threads"
  echo "   The roofline predicts 22 GFLOP/s for 32 bit data and 44 for 16 bit data."
  local shape="--heads 8 --len 65536 --dim 64 --reps 9 --warmup 3"
  show "scalar loops" 8 build/bench_decode --kernel naive $shape
  show "AVX2, one pass" 8 build/bench_decode --kernel fused $shape
  show "AVX2, one pass, 16 bit data" 8 build/bench_decode --kernel fused_f16 $shape
}

case "${1:-all}" in
  tests | gemm | attention | decode) "$1" ;;
  all) tests; gemm; attention; decode ;;
  *) echo "usage: scripts/demo.sh tests|gemm|attention|decode|all" >&2; exit 2 ;;
esac
