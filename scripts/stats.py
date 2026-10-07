"""Statistics for benchmark results.

  ci       bootstrap confidence interval of one median
  compare  ratio of two kernels at one point, with interval and permutation test
  noise    spread inside one run and between separate runs of the same point

Examples
  stats.py ci results/x/gemm.csv --kernel goto --where m=2048,k=2048,n=2048,threads=1
  stats.py compare results/x/attn.csv --a tiled --b torch --where len=8192,dim=64,threads=8
  stats.py noise results/x
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHAPE_COLUMNS = ("m", "k", "n", "heads", "len", "dim", "elems")
# The tuner writes the default sizes out in full. These rows are default runs.
DEFAULT_TAGS = ("mc=144;kc=256;nc=4080", "tq=192;tk=256")


def bootstrap_median_ci(values, level=0.95, resamples=5000, seed=0):
    """Percentile bootstrap interval for the median. Returns (low, high)."""
    values = np.asarray(values, dtype=float)
    # With fewer than 3 values there is not enough data. The range is used.
    if values.size < 3:
        return float(values.min()), float(values.max())
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[picks], axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_ratio_ci(a, b, level=0.95, resamples=5000, seed=0):
    """Confidence interval for median(a) / median(b)."""
    # Each sample is resampled on its own because they come from separate runs.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    med_a = np.median(a[rng.integers(0, a.size, size=(resamples, a.size))], axis=1)
    med_b = np.median(b[rng.integers(0, b.size, size=(resamples, b.size))], axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(med_a / med_b, [tail, 1.0 - tail])
    return float(low), float(high)


def permutation_test(a, b, resamples=10000, seed=0):
    """Two sided p value for "a and b come from the same distribution", using the difference in medians.
    It is the share of label shuffles with a difference at least as large as observed.
    It assumes the repetitions can be exchanged, which drift during a run can break."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = abs(np.median(a) - np.median(b))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(resamples):
        rng.shuffle(pooled)
        if abs(np.median(pooled[:a.size]) - np.median(pooled[a.size:])) >= observed:
            hits += 1
    # One is added to both counts so that the p value is never exactly zero.
    return (hits + 1) / (resamples + 1)


def relative_iqr(values):
    """Interquartile range divided by the median."""
    q1, q2, q3 = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return float((q3 - q1) / q2)


def load(path):
    data = pd.read_csv(path, dtype={"params": str}, keep_default_na=False, na_values=[""])
    data["params"] = data["params"].fillna("")
    return data


def select(data, kernel, where, params=""):
    """Rows of one kernel at one point. `where` maps column names to values."""
    rows = data[(data.kernel == kernel) & (data.params == params)]
    for column, value in where.items():
        rows = rows[rows[column] == value]
    return rows


def point_columns(data):
    return [c for c in SHAPE_COLUMNS if c in data.columns] + ["threads", "kernel", "params"]


def within_run_noise(data):
    """Relative IQR of every measured point in one file."""
    return data.groupby(point_columns(data)).gflops.apply(relative_iqr)


def between_run_spread(runs):
    """Spread of the median across runs of the same point, as (max - min) / min.
    `runs` is a list of data frames. A point counts when two or more runs have it."""
    medians = []
    for index, data in enumerate(runs):
        m = data.groupby(point_columns(data)).gflops.median().rename("gflops").reset_index()
        m["run"] = index
        medians.append(m)
    table = pd.concat(medians)
    keys = [c for c in table.columns if c not in ("gflops", "run")]
    table[keys] = table[keys].fillna(0)
    grouped = table.groupby(keys).gflops
    spread = (grouped.max() - grouped.min()) / grouped.min()
    return spread[grouped.count() >= 2]


def parse_where(text):
    where = {}
    for item in filter(None, text.split(",")):
        column, value = item.split("=")
        where[column] = int(value)
    return where


def command_ci(args):
    rows = select(load(args.file), args.kernel, parse_where(args.where), args.params)
    if rows.empty:
        sys.exit("no rows match")
    low, high = bootstrap_median_ci(rows.gflops)
    print(f"{args.kernel}: median {rows.gflops.median():.1f} GFLOP/s, 95% interval {low:.1f} to {high:.1f}, "
          f"{len(rows)} repetitions")


def command_compare(args):
    data, where = load(args.file), parse_where(args.where)
    a, b = select(data, args.a, where, args.params_a), select(data, args.b, where, args.params_b)
    if a.empty or b.empty:
        sys.exit("no rows match one of the kernels")
    ratio = a.gflops.median() / b.gflops.median()
    low, high = bootstrap_ratio_ci(a.gflops, b.gflops)
    p = permutation_test(a.gflops, b.gflops)
    print(f"{args.a} / {args.b} = {ratio:.3f}, 95% interval {low:.3f} to {high:.3f}, permutation p = {p:.4f}")
    print("Note: this covers the noise inside the two runs only. See the noise command for run to run changes.")


def command_noise(args):
    root = Path(args.directory)
    for name in ("gemm.csv", "attn.csv", "decode.csv"):
        if not (root / name).exists():
            continue
        noise = within_run_noise(load(root / name))
        print(f"{name}: within run IQR / median over {len(noise)} points: "
              f"median {noise.median():.1%}, 90th percentile {noise.quantile(0.9):.1%}, worst {noise.max():.1%}")
    # The tuner measures the default settings again in its own runs.
    for op, main in (("gemm", "gemm.csv"), ("attn", "attn.csv")):
        runs = [load(root / main)] + [load(p) for p in sorted((root / "tune").glob(f"{op}_*.csv"))]
        runs = [r.assign(params=r.params.where(~r.params.isin(DEFAULT_TAGS), "")) for r in runs]
        runs = [r[r.params == ""] for r in runs]
        spread = between_run_spread(runs)
        if len(spread):
            print(f"{op}: between run spread of the median over {len(spread)} repeated points: "
                  f"median {spread.median():.1%}, worst {spread.max():.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ci = commands.add_parser("ci", help="confidence interval of one median")
    ci.add_argument("file")
    ci.add_argument("--kernel", required=True)
    ci.add_argument("--where", default="", help="for example len=8192,threads=8")
    ci.add_argument("--params", default="")
    ci.set_defaults(run=command_ci)
    compare = commands.add_parser("compare", help="ratio of two kernels at one point")
    compare.add_argument("file")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--where", default="")
    compare.add_argument("--params-a", default="")
    compare.add_argument("--params-b", default="")
    compare.set_defaults(run=command_compare)
    noise = commands.add_parser("noise", help="spread inside runs and between runs")
    noise.add_argument("directory")
    noise.set_defaults(run=command_noise)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
