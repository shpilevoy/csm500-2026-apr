"""Draws the figures for one results directory.

Run:  plot_results.py results/<name>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
})


def load(path):
    """Reads a results CSV and adds derived columns."""
    d = pd.read_csv(path, dtype={"params": str}, keep_default_na=False, na_values=[""])
    d["params"] = d["params"].fillna("")
    d["flops"] = d.gflops * d.seconds * 1e9
    d["flops_per_cycle"] = d.flops / d.cycles
    return d


def medians(d, keys):
    cols = ["gflops", "flops_per_cycle", "flops", "seconds"]
    return d.groupby(keys, as_index=False)[cols].median()


def fig_gemm_ladder(gemm, machine, out):
    """Bar chart of FLOPs per cycle for each GEMM stage."""
    g = medians(gemm[(gemm.threads == 1) & (gemm.m == 2048) & (gemm.k == 2048)], ["kernel", "params"])
    stages = [("naive", "", "naive (ijk)"), ("reorder", "", "+ loop reorder (ikj)"),
              ("blocked", "", "+ cache blocking"), ("simd", "bk=1000000;bn=1000000", "reorder + AVX2, no blocking"),
              ("simd", "", "+ cache blocking + AVX2"), ("goto", "", "+ packing, 6x16 microkernel"),
              ("blis", "", "BLIS"), ("onednn", "", "oneDNN"), ("openblas", "", "OpenBLAS")]
    rows = [(label, g[(g.kernel == k) & (g.params == p)].flops_per_cycle.iloc[0], k in LIBRARIES)
            for k, p, label in stages if not g[(g.kernel == k) & (g.params == p)].empty]
    peak_fpc = machine[(machine.kernel == "fma") & (machine.threads == 1)].flops_per_cycle.median()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [ORANGE if r[2] else BLUE for r in rows][::-1]
    ax.barh(labels, values, color=colors, height=0.62)
    for y, v in enumerate(values):
        ax.text(v + 0.4, y, f"{v:.2f}" if v < 10 else f"{v:.1f}", va="center", color=INK, fontsize=9)
    ax.axvline(peak_fpc, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(peak_fpc - 0.3, len(values) - 0.45, f"peak {peak_fpc:.1f}", color=INK2, fontsize=9, ha="right")
    ax.set_xlim(0, peak_fpc * 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("FLOPs per core cycle (single thread)")
    ax.set_title("GEMM 2048x2048x2048, one thread: optimisation stages")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["this work", "library"], loc="upper right", bbox_to_anchor=(0.9, 0.8))
    fig.tight_layout()
    fig.savefig(out / "gemm_ladder.png", dpi=200)


LIBRARIES = ("blis", "openblas", "onednn")

ATTN = [("mat_stream", "materialised", BLUE, "o"), ("qblock", "query-blocked", ORANGE, "s"),
        ("tiled", "tiled, online softmax", AQUA, "^"), ("torch", "PyTorch (blocked)", YELLOW, "D"),
        ("torch_direct", "PyTorch, direct method", MAGENTA, "v")]


DECODE = [("naive", "naive (scalar)", BLUE, "o"), ("simd", "AVX2, three passes", ORANGE, "s"),
          ("fused", "AVX2, one pass", AQUA, "^"), ("torch", "PyTorch", YELLOW, "D"),
          ("fused_f16", "AVX2, one pass, half precision", MAGENTA, "v")]


def main():
    root = Path(sys.argv[1])
    out = root / "figures"
    out.mkdir(exist_ok=True)
    machine, gemm = load(root / "machine.csv"), load(root / "gemm.csv")
    fig_gemm_ladder(gemm, machine, out)
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
