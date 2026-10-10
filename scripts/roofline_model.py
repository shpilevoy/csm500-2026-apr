"""Compares roofline limits with the measurements.

The limit is the smaller of the compute peak and intensity x bandwidth. For
decode step attention the intensity is known in advance, so it is a prediction.

Run:  roofline_model.py results/<name> [--op gemm|attn|decode] [--threads N]
"""
import argparse
from pathlib import Path

import pandas as pd

from plot_results import ceilings, load

# Bytes read for each float of keys and values in a decode step.
BYTES_FP32, BYTES_FP16 = 4, 2


def limit(intensity, peak, bandwidth):
    """Roofline limit in GFLOP/s. `bandwidth` is in GB/s."""
    return min(peak, intensity * bandwidth)


def regime(intensity, peak, bandwidth):
    """Says which of the two limits applies."""
    return "memory" if intensity * bandwidth < peak else "compute"


def ridge(peak, bandwidth):
    """Intensity at which the two limits meet."""
    return peak / bandwidth


def decode_intensity(bytes_per_float=BYTES_FP32):
    """Operations per byte for a decode step."""
    # Each key costs 2*d operations for its score and each value costs 2*d for
    # the weighted sum. Both are read once, which is 2*d floats.
    return 4.0 / (2.0 * bytes_per_float)


def with_limits(data, machine):
    """Adds intensity, limit, regime and share of the limit to measured rows."""
    rows = []
    for threads, group in data.groupby("threads"):
        peak, bandwidth = ceilings(machine, threads)
        group = group.copy()
        group["intensity"] = group.flops / group.dram_bytes.clip(lower=1)
        group["limit"] = [limit(i, peak, bandwidth) for i in group.intensity]
        group["regime"] = [regime(i, peak, bandwidth) for i in group.intensity]
        group["share"] = group.gflops / group.limit
        rows.append(group)
    return pd.concat(rows)


def summary(data, machine, shape):
    """Median of each point, with its limit."""
    table = with_limits(data, machine)
    keys = shape + ["threads", "kernel", "params"]
    out = table.groupby(keys).agg(gflops=("gflops", "median"), intensity=("intensity", "median"),
                                  limit=("limit", "median"), share=("share", "median"),
                                  regime=("regime", lambda r: r.mode().iloc[0])).reset_index()
    return out


def decode_prediction(decode, machine, threads):
    """Predicted and measured decode throughput once the data is out of cache."""
    # Only points whose keys and values are four times larger than the L3 cache
    # are used. The L3 size is taken as 16 MiB, the size on the test machine.
    peak, bandwidth = ceilings(machine, threads)
    rows = decode[(decode.threads == threads) & (decode.heads * decode.len * decode.dim * 8 >= 4 * 16 * 2 ** 20)]
    lines = []
    for kernel, group in rows.groupby("kernel"):
        size = BYTES_FP16 if kernel.endswith("f16") else BYTES_FP32
        predicted = limit(decode_intensity(size), peak, bandwidth)
        measured = group.groupby("len").gflops.median()
        lines.append((kernel, decode_intensity(size), predicted, measured.min(), measured.max()))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--op", choices=("gemm", "attn", "decode"), default="decode")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.directory)
    machine = load(root / "machine.csv")
    peak, bandwidth = ceilings(machine, args.threads)
    print(f"{args.threads} threads: compute peak {peak:.0f} GFLOP/s, read bandwidth {bandwidth:.1f} GB/s, "
          f"ridge at {ridge(peak, bandwidth):.1f} FLOP per byte")
    data = load(root / f"{args.op}.csv")
    if args.op == "decode":
        print("kernel      intensity  predicted  measured (GFLOP/s, data out of cache)")
        for kernel, intensity, predicted, low, high in decode_prediction(data, machine, args.threads):
            print(f"{kernel:10s}  {intensity:9.2f}  {predicted:9.1f}  {low:.1f} to {high:.1f}")
        return
    shape = ["m", "k", "n"] if args.op == "gemm" else ["heads", "len", "dim"]
    table = summary(data[data.threads == args.threads], machine, shape)
    print(table.round({"gflops": 1, "intensity": 2, "limit": 1, "share": 2}).to_string(index=False))


if __name__ == "__main__":
    main()
