"""Prints the median of each point from one or more benchmark CSV files."""
import sys

import pandas as pd

d = pd.concat(pd.read_csv(p, keep_default_na=False, na_values=[""], dtype={"params": str}) for p in sys.argv[1:])
d["params"] = d["params"].fillna("")
shape = [c for c in ("m", "k", "n", "heads", "len", "dim") if c in d.columns]
flops = d.gflops * d.seconds * 1e9
# Cycles are summed over threads. So this is per core and does not depend on frequency.
d["flops_per_cycle"] = flops / d.cycles
d["ghz"] = d.cycles / d.seconds / d.threads / 1e9
g = d.groupby(shape + ["threads", "kernel", "params"]).agg(
    reps=("gflops", "size"),
    gflops=("gflops", "median"),
    q1=("gflops", lambda x: x.quantile(0.25)),
    q3=("gflops", lambda x: x.quantile(0.75)),
    flops_per_cycle=("flops_per_cycle", "median"),
    ghz=("ghz", "median"),
    rel_err=("rel_err", "max"),
)
pd.set_option("display.width", 200)
print(g.round({"gflops": 1, "q1": 1, "q3": 1, "flops_per_cycle": 2, "ghz": 2}).to_string())
