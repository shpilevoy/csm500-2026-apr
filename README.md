# Roofline analysis of hand-written GEMM and attention kernels

## Reproducing the measurements

Requirements: Linux x86-64 with AVX2 and FMA, GCC, CMake, Python 3.

```bash
sudo apt-get install libopenblas-openmp-dev libblis-dev libdnnl-dev clang libomp-dev hwloc
sudo sysctl kernel.perf_event_paranoid=1      # hardware counters; resets on reboot

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

cmake -S . -B build && cmake --build build -j
ctest --test-dir build                        # 661 kernel cases against the reference implementations, 37 harness tests

scripts/run_all.sh <name>                     # full campaign into results/<name>/
.venv/bin/python scripts/plot_results.py results/<name>
.venv/bin/python scripts/summarise.py results/<name>/gemm.csv
```

`run_all.sh` can be re-run after an interruption: finished points are skipped.

Data-race check (clang, with the OpenMP-aware ThreadSanitizer plugin):

```bash
CXX=clang++ cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-fsanitize=thread
cmake --build build-tsan -j --target test_gemm test_attn
OMP_TOOL_LIBRARIES=/usr/lib/llvm-21/lib/libarcher.so TSAN_OPTIONS=ignore_noninstrumented_modules=1 \
  OMP_WAIT_POLICY=passive OMP_NUM_THREADS=4 ./build-tsan/test_attn
```

## What the campaign controls and what it only records

- Threads are pinned one per physical core (`OMP_PLACES`), points run in shuffled order.
- Frequency scaling, turbo and SMT are left at system defaults and recorded in
  `results/<name>/*.machine.json`. Cycle counts are recorded for every repetition, so
  throughput can be reported in FLOPs per core cycle, which does not depend on frequency.
- The cache-traffic counters (`l2_miss`, `l2pf_l3`, `l2pf_dram`, `dram_fills`) are raw
  AMD Zen codes; see `bench/perf_counters.hpp`. On another CPU family they must be replaced.
- Bytes read from main memory are estimated as `64 * (l2pf_dram + min(l2_miss, dram_fills))`
  (`dram_lines` in `scripts/plot_results.py`). The campaign checks this against four loops of
  known traffic; see Table C1b in the report.

## Layout

| Path | Contents |
|---|---|
| `src/gemm_*.cpp` | GEMM reference implementation and the optimisation ladder |
| `src/attention*.cpp` | Attention reference implementation and kernels (optional causal mask) |
| `src/decode*.cpp` | Decode-step attention: one query against cached keys and values |
| `bench/` | Benchmark programs: `bench_gemm`, `bench_attn`, `bench_decode`, `bench_machine`, and `attn_phases` |
| `tests/` | Correctness tests against the reference implementations; `test_harness.py` tests the harness |
| `scripts/run_sweep.py` | One sweep: shuffled, pinned, resumable |
| `scripts/autotune.py` | Coordinate-descent tuner over block and tile sizes |
| `scripts/run_all.sh` | The whole campaign |
| `results/<name>/` | Raw per-repetition CSVs, machine description, figures |

