"""Tunes block and tile sizes by coordinate descent from the defaults.

Every trial is a normal benchmark row in --out. The winner is then measured
again against the defaults and written to <out>_confirm.csv. Report that result.
"""
import argparse
import csv
import os
import statistics
import subprocess
from pathlib import Path

from run_sweep import OPS, parse_shape, physical_cores

DEFAULTS = {"bk": 64, "bn": 512, "mc": 144, "kc": 256, "nc": 4080, "tq": 192, "tk": 256}


def coordinate_descent(evaluate, grid, start, max_passes=3):
    """Maximises evaluate(params) one parameter at a time. Returns (best params, best score)."""
    current = dict(start)
    best = evaluate(current)
    for _ in range(max_passes):
        changed = False
        for name, values in grid.items():
            scores = {v: evaluate(current | {name: v}) for v in values}
            winner = max(scores, key=scores.get)
            if scores[winner] > best and winner != current[name]:
                current[name], best, changed = winner, scores[winner], True
        if not changed:
            break
    return current, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=("gemm", "attn"), required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--grid", nargs="+", required=True, metavar="PARAM=V1,V2,...")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--confirm-reps", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True, help="CSV file that gets every trial")
    args = ap.parse_args()

    bench, cols = OPS[args.op]
    shape = parse_shape(args.shape, len(cols))
    grid = {}
    for item in args.grid:
        name, values = item.split("=")
        grid[name] = sorted({int(v) for v in values.split(",")} | {DEFAULTS[name]})
    current = {name: DEFAULTS[name] for name in grid}

    cores = physical_cores()
    env = os.environ | {
        "OMP_NUM_THREADS": str(args.threads),
        "OMP_PROC_BIND": "true",
        "OMP_PLACES": ",".join(f"{{{c}}}" for c in cores[:args.threads]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}

    def measure(params, reps, out):
        """Median GFLOP/s of `reps` new repetitions appended to `out`."""
        tag = ";".join(f"{k}={v}" for k, v in params.items())
        cmd = [str(bench), "--kernel", args.kernel, "--reps", str(reps),
               "--warmup", str(args.warmup), "--out", str(out)]
        for col, value in zip(cols, shape):
            cmd += [f"--{col}", str(value)]
        for k, v in params.items():
            cmd += ["--set", f"{k}={v}"]
        subprocess.run(cmd, env=env, check=True)
        with out.open() as f:
            rows = [r for r in csv.DictReader(f) if r["params"] == tag]
        return statistics.median(float(r["gflops"]) for r in rows[-reps:])

    def evaluate(params):
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = measure(params, args.reps, args.out)
            print(f"  {';'.join(f'{k}={v}' for k, v in params.items())}: {cache[key]:.1f} GFLOP/s", flush=True)
        return cache[key]

    start = evaluate(current)
    current, best = coordinate_descent(evaluate, grid, current, args.max_passes)
    tag = " ".join(f"{k}={v}" for k, v in current.items())
    print(f"search:  defaults {start:.1f} -> best {best:.1f} GFLOP/s, {len(cache)} trials, best: {tag}")
    confirm = args.out.with_name(args.out.stem + "_confirm.csv")
    confirm.unlink(missing_ok=True)
    base = measure({name: DEFAULTS[name] for name in grid}, args.confirm_reps, confirm)
    tuned = measure(current, args.confirm_reps, confirm)
    print(f"confirm: defaults {base:.1f} -> tuned {tuned:.1f} GFLOP/s ({tuned / base - 1:+.1%}) "
          f"at {args.confirm_reps} repetitions")


if __name__ == "__main__":
    main()
