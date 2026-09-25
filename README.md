# Roofline analysis of hand-written GEMM and attention kernels

## Reproducing the measurements

Requirements: Linux x86-64 with AVX2 and FMA, GCC, CMake, Python 3.

```bash
sudo apt-get install libopenblas-openmp-dev libblis-dev hwloc

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

cmake -S . -B build && cmake --build build -j
ctest --test-dir build

scripts/run_all.sh <name>                     # full campaign into results/<name>/
.venv/bin/python scripts/plot_results.py results/<name>
```

`run_all.sh` can be re-run after an interruption: finished points are skipped.
