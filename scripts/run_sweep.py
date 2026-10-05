"""Runs a benchmark sweep with one process per point.

Points run in shuffled order, so slow drift does not line up with one kernel.
Finished points are skipped, so a sweep can resume after an interruption.
"""
import argparse
import csv
import glob
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# For each operation the benchmark program and the shape names. The names
# are both CLI flags and CSV columns.
OPS = {
    "gemm": (ROOT / "build" / "bench_gemm", ("m", "k", "n")),
    "attn": (ROOT / "build" / "bench_attn", ("heads", "len", "dim")),
    "decode": (ROOT / "build" / "bench_decode", ("heads", "len", "dim")),
    "machine": (ROOT / "build" / "bench_machine", ("elems",)),
}
TORCH_ATTN = [sys.executable, str(ROOT / "scripts" / "bench_torch_attn.py")]
TORCH_KERNELS = ("torch", "torch_direct")
LIB_DIR = Path("/usr/lib/x86_64-linux-gnu")
# Library baselines. The flag tells bench_gemm how to call the library.
BLAS_LIBS = {
    "openblas": ("--blas-lib", LIB_DIR / "openblas-openmp" / "libopenblas.so.0"),
    "blis": ("--blas-lib", LIB_DIR / "blis-openmp" / "libblis.so.4"),
    "onednn": ("--dnnl-lib", LIB_DIR / "libdnnl.so.3.9"),
}


def physical_cores():
    """Returns the first hardware thread of every physical core."""
    firsts = set()
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
        firsts.add(int(Path(path).read_text().replace("-", ",").split(",")[0]))
    return sorted(firsts)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def machine_info():
    """Describes the machine and its settings. Saved next to the results."""
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), None)
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "cpu": cpu,
        "kernel": platform.release(),
        "physical_cores": physical_cores(),
        # These are only recorded. Frequency scaling is left at the system defaults.
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "boost": read("/sys/devices/system/cpu/cpufreq/boost"),
        "smt": read("/sys/devices/system/cpu/smt/control"),
        "compiler": subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "blas_libs": {name: os.path.realpath(path) for name, (_, path) in BLAS_LIBS.items()},
        "python": platform.python_version(),
    }


def parse_shape(text, rank):
    """Parses S or AxBxC into `rank` sizes. A single number is repeated."""
    dims = [int(x) for x in text.split("x")]
    if len(dims) == 1:
        dims *= rank
    if len(dims) != rank:
        raise SystemExit(f"shape {text!r} does not have {rank} dimensions")
    return tuple(dims)


def done_points(out, cols):
    """Counts the rows already in `out` for each point."""
    counts = Counter()
    if out.exists():
        with out.open() as f:
            for row in csv.DictReader(f):
                shape = tuple(int(row[c]) for c in cols)
                counts[(row["kernel"], row["params"], shape, int(row["threads"]))] += 1
    return counts


def main():
    cores = physical_cores()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=OPS, default="gemm")
    ap.add_argument("--kernels", nargs="+", required=True,
                    help=f"our kernel names or baselines. gemm baselines are {', '.join(BLAS_LIBS)}. "
                         f"attn and decode baselines are {', '.join(TORCH_KERNELS)}. "
                         "machine kernels are fma, triad, sum, update, fill")
    ap.add_argument("--shapes", nargs="+", required=True,
                    help="gemm takes S for a square or MxKxN. attn and decode take HEADSxLENxDIM. "
                         "machine takes floats per array")
    ap.add_argument("--set", nargs="+", default=[], metavar="PARAM=VALUE",
                    help="block or tile sizes for our kernels. causal=1 also applies to PyTorch")
    ap.add_argument("--threads", nargs="+", type=int, default=[1])
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="record error against the reference implementation (slow)")
    ap.add_argument("--out", type=Path, required=True, help="CSV to append to")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if max(args.threads) > len(cores):
        ap.error(f"at most {len(cores)} threads: one per physical core")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".machine.json").write_text(json.dumps(machine_info(), indent=2) + "\n")

    bench, cols = OPS[args.op]
    shapes = [parse_shape(text, len(cols)) for text in args.shapes]
    points = [(k, s, t) for k in args.kernels for s in shapes for t in args.threads]
    random.Random(args.seed).shuffle(points)
    done = done_points(args.out, cols)

    for i, (kernel, shape, threads) in enumerate(points, 1):
        torch = args.op in ("attn", "decode") and kernel in TORCH_KERNELS
        baseline = torch or (args.op == "gemm" and kernel in BLAS_LIBS)
        # A library ignores our block sizes. The mask is the only setting it shares.
        settings = [kv for kv in args.set if kv == "causal=1"] if torch else [] if baseline else args.set
        params = ";".join(settings)
        label = f"[{i}/{len(points)}] {kernel} {'x'.join(map(str, shape))} threads={threads} {params}"
        if done[(kernel, params, shape, threads)] >= args.reps:
            print(label, "- already done")
            continue
        print(label, flush=True)
        env = os.environ | {
            "OMP_NUM_THREADS": str(threads),
            "BLIS_NUM_THREADS": str(threads),
            "OMP_PROC_BIND": "true",
            "OMP_PLACES": ",".join(f"{{{c}}}" for c in cores[:threads]),
        }
        cmd = TORCH_ATTN + (["--decode"] if args.op == "decode" else []) if torch else [str(bench)]
        cmd = cmd + ["--kernel", kernel, "--reps", str(args.reps), "--out", str(args.out)]
        if args.op != "machine":
            cmd += ["--warmup", str(args.warmup)]
        for col, value in zip(cols, shape):
            cmd += [f"--{col}", str(value)]
        if args.op == "gemm" and baseline:
            flag, path = BLAS_LIBS[kernel]
            cmd += [flag, str(path)]
        for kv in settings:
            cmd += ["--set", kv]
        if args.check and not baseline:
            cmd.append("--check")
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
