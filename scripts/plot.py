"""Quick look at a sweep: median GFLOP/s per kernel and size."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.read_csv(sys.argv[1])
d = d[d.threads == 1]
g = d.groupby(["kernel", "m"], as_index=False).gflops.median()
fig, ax = plt.subplots(figsize=(7, 4))
for kernel, rows in g.groupby("kernel"):
    ax.plot(rows.m, rows.gflops, marker="o", label=kernel)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("matrix size")
ax.set_ylabel("GFLOP/s")
ax.legend()
fig.tight_layout()
fig.savefig("sweep.png", dpi=150)
