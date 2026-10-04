#!/usr/bin/env bash
# The full measurement campaign. It is safe to run again after an interruption
# because finished points are skipped. Usage: scripts/run_all.sh [results-subdir]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/${1:-$(hostname)}
run() { .venv/bin/python scripts/run_sweep.py "$@"; }
# A search is skipped when its confirmation file is already there.
tune() {
  local out
  out=$(printf '%s\n' "$@" | grep -A1 -x -- --out | tail -1)
  if [ -f "${out%.csv}_confirm.csv" ]; then echo "search already done: $out"; return; fi
  .venv/bin/python scripts/autotune.py "$@"
}

# Machine ceilings
run --op machine --kernels fma --shapes 0 --threads 1 2 4 8 --out "$OUT/machine.csv"
run --op machine --kernels triad --threads 1 8 --out "$OUT/machine.csv" --shapes \
  1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 16777216
# Loops with known memory traffic, to check the counter-based traffic estimate.
run --op machine --kernels sum update fill --threads 1 8 --out "$OUT/machine.csv" --shapes 16777216

# GEMM
# The reference implementation is too slow for the larger shapes.
SMALL="256 512 1024 512x4096x512 1024x768x3072 1024x3072x768 256x64x1024"
G="$OUT/gemm.csv"
run --kernels goto simd openblas blis --shapes $SMALL --threads 1 --check --out "$G"
run --kernels goto simd openblas blis --shapes 2048 4096 --threads 1 --out "$G"
# simd with blocking switched off, to separate the two effects.
run --kernels simd --set bk=1000000 bn=1000000 --shapes $SMALL --threads 1 --check --out "$G"
run --kernels simd --set bk=1000000 bn=1000000 --shapes 2048 4096 --threads 1 --out "$G"
run --kernels reorder blocked --shapes $SMALL --threads 1 --check --out "$G"
run --kernels reorder blocked --shapes 2048 --threads 1 --out "$G"
run --kernels naive --shapes 256 512 1024 512x4096x512 --threads 1 --reps 10 --warmup 2 --check --out "$G"
# One slow point so that every stage of the ladder is measured at the same size.
run --kernels naive --shapes 2048 --threads 1 --reps 3 --warmup 1 --out "$G"
# Block sizes worked out from the cache sizes, next to the defaults copied from BLIS.
run --kernels goto --set $(.venv/bin/python scripts/analytic_blocks.py --settings) \
  --shapes 2048 1024x768x3072 256x64x1024 --threads 1 --out "$G"
run --kernels goto_mt --set $(.venv/bin/python scripts/analytic_blocks.py --threads 8 --settings) \
  --shapes 2048 1024x768x3072 --threads 8 --out "$G"
run --kernels goto_mt openblas blis --shapes $SMALL 2048 4096 --threads 8 --out "$G"
run --kernels goto_mt openblas blis --shapes 2048 --threads 1 2 4 --out "$G"

# Attention
A="$OUT/attn.csv"
ATTN="mat mat_stream qblock tiled torch torch_direct"
run --op attn --kernels $ATTN --threads 1 8 --check --out "$A" --shapes \
  8x128x64 8x256x64 8x512x64 8x1024x64 8x2048x64
run --op attn --kernels $ATTN --threads 1 8 --out "$A" --shapes 8x4096x64
run --op attn --kernels naive --threads 1 8 --reps 10 --warmup 2 --check --out "$A" --shapes \
  8x128x64 8x256x64 8x512x64 8x1024x64
run --op attn --kernels $ATTN --threads 8 --reps 15 --warmup 3 --out "$A" --shapes 8x8192x64
run --op attn --kernels mat mat_stream qblock tiled torch --threads 1 --reps 10 --warmup 2 --out "$A" --shapes 8x8192x64
# The full score table needs 8.6 GB at this length, so only the blocked kernels run.
run --op attn --kernels qblock tiled --threads 8 --reps 10 --warmup 2 --out "$A" --shapes 8x16384x64
run --op attn --kernels mat mat_stream qblock tiled torch --threads 8 --reps 15 --warmup 3 --out "$A" --shapes \
  8x256x128 8x1024x128 8x2048x128 8x4096x128 8x8192x128
# Thread scaling.
run --op attn --kernels mat_stream tiled torch --threads 2 4 --out "$A" --shapes 8x2048x64
# Causal mask, as in a language model.
run --op attn --kernels mat_stream qblock tiled torch torch_direct --set causal=1 --threads 8 --check --out "$A" --shapes \
  8x512x64 8x1024x64 8x2048x64
run --op attn --kernels mat_stream qblock tiled torch torch_direct --set causal=1 --threads 8 --reps 15 --warmup 3 \
  --out "$A" --shapes 8x4096x64 8x8192x64
run --op attn --kernels naive --set causal=1 --threads 8 --reps 10 --warmup 2 --check --out "$A" --shapes 8x512x64 8x1024x64

# Decode-step attention. One query against L cached keys and values.
D="$OUT/decode.csv"
run --op decode --kernels naive simd fused fused_f16 torch --threads 1 8 --check --out "$D" --shapes \
  8x64x64 8x256x64 8x1024x64 8x4096x64 8x16384x64 8x65536x64 8x262144x64

# Where full-matrix attention reads memory. One row per phase.
OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14}" \
  build/attn_phases 8192 > "$OUT/attn_phases.csv"

# Autotuning. Each search ends by measuring its winner again against the defaults.
T="$OUT/tune"
mkdir -p "$T"
GEMM_GRID="mc=72,96,144,192,288,384 kc=128,192,256,384,512 nc=1024,2048,4080,8160"
tune --op gemm --kernel goto --shape 2048 --threads 1 --grid $GEMM_GRID --out "$T/gemm_2048_t1.csv" | tee -a "$T/gemm_2048_t1.txt"
tune --op gemm --kernel goto --shape 1024x768x3072 --threads 1 --grid $GEMM_GRID --out "$T/gemm_mlp_t1.csv" | tee -a "$T/gemm_mlp_t1.txt"
tune --op gemm --kernel goto_mt --shape 2048 --threads 8 --grid $GEMM_GRID --out "$T/gemm_2048_t8.csv" | tee -a "$T/gemm_2048_t8.txt"
ATTN_GRID="tq=48,96,192,384,768 tk=64,128,256,512,1024"
tune --op attn --kernel tiled --shape 8x2048x64 --threads 8 --grid $ATTN_GRID --out "$T/attn_2048_t8.csv" | tee -a "$T/attn_2048_t8.txt"
tune --op attn --kernel tiled --shape 8x8192x64 --threads 8 --grid $ATTN_GRID --out "$T/attn_8192_t8.csv" | tee -a "$T/attn_8192_t8.txt"
echo "campaign complete"
