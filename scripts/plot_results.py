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
    d["dram_bytes"] = dram_lines(d) * 64
    return d


def dram_lines(d):
    """Cache lines read from main memory, estimated from the hardware counters."""
    # L2 prefetches that missed L3 plus demand misses of L2 filled from main memory.
    # min() is used because a line counts only if it both missed L2 and came from
    # main memory. The calibration table in make_tables.py checks this estimate.
    return d.l2pf_dram + d[["l2_miss", "dram_fills"]].min(axis=1)


def medians(d, keys):
    cols = ["gflops", "flops_per_cycle", "flops", "dram_bytes", "seconds"]
    return d.groupby(keys, as_index=False)[cols].median()


def ceilings(machine, threads):
    """Returns (compute peak in GFLOP/s, read bandwidth of main memory in GB/s)."""
    # The bandwidth is that of the read-only loop. Its traffic is known exactly.
    # It is the most a kernel can read and is reached only if nothing is written back.
    m = machine[machine.threads == threads]
    peak = m[m.kernel == "fma"].gflops.median()
    big = m[(m.kernel == "sum") & (m.elems == m.elems.max())]
    return peak, (big.elems * big.passes * big.threads * 4 / big.seconds).median() / 1e9


def roofline(ax, peak, bandwidth, x_lo, x_hi):
    """Draws the two roofline limits on log axes."""
    ridge = peak / bandwidth
    ax.plot([x_lo, ridge, x_hi], [x_lo * bandwidth, peak, peak], color=INK2, linewidth=1.5, zorder=2)
    ax.annotate(f"compute peak {peak:.0f} GFLOP/s", (x_hi, peak), xytext=(-4, 5),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.annotate(f"memory read bandwidth {bandwidth:.0f} GB/s", (x_lo * 2, x_lo * 2 * bandwidth), xytext=(6, -12),
                textcoords="offset points", color=INK2, fontsize=9, rotation=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte read from DRAM, measured)")
    ax.set_ylabel("GFLOP/s")


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


def fig_gemm_roofline(gemm, machine, out):
    """GEMM kernels on the roofline, one thread."""
    peak, bw = ceilings(machine, 1)
    g = medians(gemm[(gemm.threads == 1) & (gemm.m == 2048) & (gemm.k == 2048)], ["kernel", "params"])
    names = {("naive", ""): "naive", ("reorder", ""): "reorder", ("blocked", ""): "blocked", ("simd", "bk=1000000;bn=1000000"): "AVX2, no blocking",
             ("simd", ""): "AVX2 + blocking", ("goto", ""): "packed microkernel", ("blis", ""): "BLIS",
             ("openblas", ""): "OpenBLAS", ("onednn", ""): "oneDNN"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    roofline(ax, peak, bw, 0.05, 2e4)
    ax.set_ylim(0.2, peak * 2.5)
    for _, r in g.iterrows():
        name = names.get((r.kernel, r.params))
        if name is None:
            continue
        lib = r.kernel in LIBRARIES
        x = r.flops / max(r.dram_bytes, 1)
        ax.scatter([x], [r.gflops], color=ORANGE if lib else BLUE, s=40 if lib else 55, zorder=4 if lib else 3,
                   edgecolor=SURFACE, linewidth=1.2)
        offset, align = {"OpenBLAS": ((-8, 4), "right"), "BLIS": ((6, 10), "left"), "oneDNN": ((-8, -14), "right"),
                         "packed microkernel": ((9, -14), "left"),
                         "AVX2, no blocking": ((10, -4), "left")}.get(name, ((0, 9), "center"))
        ax.annotate(name, (x, r.gflops), xytext=offset, textcoords="offset points", ha=align,
                    fontsize=9, color=INK)
    ax.scatter([], [], color=BLUE, label="this work")
    ax.scatter([], [], color=ORANGE, label="library")
    ax.legend(loc="lower right")
    ax.set_title("GEMM 2048x2048x2048 on the roofline, single thread")
    fig.tight_layout()
    fig.savefig(out / "gemm_roofline.png", dpi=200)


LIBRARIES = ("blis", "openblas", "onednn")

ATTN = [("mat_stream", "materialised", BLUE, "o"), ("qblock", "query-blocked", ORANGE, "s"),
        ("tiled", "tiled, online softmax", AQUA, "^"), ("torch", "PyTorch (blocked)", YELLOW, "D"),
        ("torch_direct", "PyTorch, direct method", MAGENTA, "v")]


def fig_attn_scaling(attn, machine, out):
    """Attention throughput against sequence length."""
    a = medians(attn[(attn.threads == 8) & (attn.dim == 64) & (attn.params == "")], ["kernel", "len"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for kernel, label, color, marker in ATTN:
        s = a[a.kernel == kernel].sort_values("len")
        ax.plot(s.len, s.gflops, color=color, marker=marker, label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(a.len.unique()))
    ax.set_xticklabels([str(v) for v in sorted(a.len.unique())])
    ax.set_xlim(a.len.min() * 0.85, a.len.max() * 1.2)
    ax.set_ylim(0, None)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("GFLOP/s (matrix products only)")
    ax.set_title("Attention throughput by sequence length (8 heads x 64 dims, 8 threads)")
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    ax.set_ylim(0, a.gflops.max() * 1.35)
    fig.tight_layout()
    fig.savefig(out / "attn_scaling.png", dpi=200)


def fig_attn_roofline(attn, machine, out):
    """Attention kernels on the roofline, 8 threads."""
    peak, bw = ceilings(machine, 8)
    a = medians(attn[(attn.threads == 8) & (attn.dim == 64) & (attn.params == "") & (attn.len >= 1024)],
                ["kernel", "len"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    roofline(ax, peak, bw, 0.5, 2e4)
    ax.set_ylim(20, peak * 2.5)
    # Both materialised kernels are drawn here. Points only, since a line through lengths doubles back.
    series = [("mat", "materialised", "#104281", "o"), ("mat_stream", "materialised, streaming", BLUE, "o")] + ATTN[1:]
    for kernel, label, color, marker in series:
        s = a[a.kernel == kernel].sort_values("len")
        x = s.flops / s.dram_bytes.clip(lower=1)
        ax.scatter(x, s.gflops, color=color, marker=marker, label=label, s=45, edgecolor=SURFACE, linewidth=1.2,
                   zorder=3)
    ax.set_ylim(20, peak * 2.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Attention on the roofline, one point per length from 1,024 (8 threads)")
    fig.tight_layout()
    fig.savefig(out / "attn_roofline.png", dpi=200)


DECODE = [("naive", "naive (scalar)", BLUE, "o"), ("simd", "AVX2, three passes", ORANGE, "s"),
          ("fused", "AVX2, one pass", AQUA, "^"), ("torch", "PyTorch", YELLOW, "D"),
          ("fused_f16", "AVX2, one pass, half precision", MAGENTA, "v")]


def fig_decode(decode, machine, out):
    """Decode-step attention against the size of the cached keys and values."""
    peak, bw = ceilings(machine, 8)
    dcd = medians(decode[(decode.threads == 8) & (decode.dim == 64)], ["kernel", "heads", "len", "dim"])
    dcd["mib"] = dcd.heads * dcd.len * dcd.dim * 8 / 2**20  # keys and values, all heads
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for kernel, label, color, marker in DECODE:
        s = dcd[dcd.kernel == kernel].sort_values("len")
        ax.plot(s.mib, s.gflops, color=color, marker=marker, label=label, markeredgecolor=SURFACE, markeredgewidth=1.2)
    # Every byte of keys and values is read once for 0.5 operations.
    # Half precision reads half the bytes, so its limit is twice as high.
    for intensity, name, drop in ((0.5, "32 bit", -42), (1.0, "16 bit", 16)):
        roof = intensity * bw
        ax.axhline(roof, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(f"limit, {name} data: {intensity:g} FLOP/byte x {bw:.0f} GB/s = {roof:.0f}", (dcd.mib.max(), roof),
                    xytext=(0, drop), textcoords="offset points", color=INK2, fontsize=9, ha="right",
                    arrowprops={"arrowstyle": "-", "color": AXIS})
    for size, name in ((4, "all L2 caches"), (16, "L3 cache")):
        ax.axvline(size, color=AXIS, linewidth=1)
        ax.annotate(name, (size, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = sorted(dcd.mib.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticks([3, 5, 10, 20, 40, 80, 160])
    ax.set_yticklabels(["3", "5", "10", "20", "40", "80", "160"])
    ax.minorticks_off()
    ax.set_xlabel("Cached keys and values, all heads, as 32 bit floats (MiB)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Decode-step attention by size of the cached keys and values (8 threads)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "decode_cache.png", dpi=200)


def main():
    root = Path(sys.argv[1])
    out = root / "figures"
    out.mkdir(exist_ok=True)
    machine, gemm, attn = load(root / "machine.csv"), load(root / "gemm.csv"), load(root / "attn.csv")
    fig_gemm_ladder(gemm, machine, out)
    fig_gemm_roofline(gemm, machine, out)
    fig_attn_scaling(attn, machine, out)
    fig_attn_roofline(attn, machine, out)
    if (root / "decode.csv").exists():
        fig_decode(load(root / "decode.csv"), machine, out)
    print("wrote", *sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
