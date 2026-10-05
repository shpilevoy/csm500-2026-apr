"""Shows which internal routine PyTorch uses for attention on this machine.

The report treats torch.nn.functional.scaled_dot_product_attention as a blocked
kernel. This script checks that for every shape that is measured.

Run:  check_torch_dispatch.py [--out FILE]
"""
import argparse
import resource

import torch
from torch.profiler import ProfilerActivity, profile

LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)
BLOCKED = "aten::_scaled_dot_product_flash_attention_for_cpu"


def operators(function):
    """Names of the aten operators that one call runs."""
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        function()
    return {event.key for event in prof.key_averages() if event.key.startswith("aten::")}


def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def check(heads, length, dim, causal):
    q, k, v = (torch.rand(1, heads, length, dim) for _ in range(3))
    before = peak_gib()
    ops = operators(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal))
    table_gib = heads * length * length * 4 / 2 ** 30
    return BLOCKED in ops, peak_gib() - before, table_gib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", help="also write the table to this file")
    args = parser.parse_args()
    lines = [f"torch {torch.__version__}", "heads,len,dim,causal,blocked_kernel,peak_memory_growth_gib,score_table_gib"]
    # Peak memory only grows. Short lengths run first so that each row shows new growth.
    for dim in (64, 128):
        for causal in (False, True):
            for length in LENGTHS:
                blocked, growth, table = check(8, length, dim, causal)
                lines.append(f"8,{length},{dim},{int(causal)},{int(blocked)},{growth:.3f},{table:.3f}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
