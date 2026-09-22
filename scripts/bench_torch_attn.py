"""PyTorch baseline for bench_attn and bench_decode. Same shapes and CSV columns.

torch uses scaled_dot_product_attention. On a CPU it is blocked and does not
build the full table of scores. torch_direct uses plain matrix products and
builds the full table. With --decode there is one query per head.
"""
import argparse
import math
import os
import time

from perf_counters import NAMES, PerfCounters

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--kernel", default="torch", choices=("torch", "torch_direct"))
ap.add_argument("--heads", type=int, required=True)
ap.add_argument("--len", type=int, required=True)
ap.add_argument("--dim", type=int, required=True)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--decode", action="store_true")
ap.add_argument("--set", action="append", default=[], help="only causal=1 has an effect here")
ap.add_argument("--out", required=True)
args = ap.parse_args()

# Opened before torch starts its worker threads, so that they inherit the counters.
counters = PerfCounters()
import torch  # noqa: E402

threads = int(os.environ.get("OMP_NUM_THREADS", torch.get_num_threads()))
torch.set_num_threads(threads)
H, L, d = args.heads, args.len, args.dim
causal = "causal=1" in args.set
gen = torch.Generator().manual_seed(0)
q = (torch.rand(1, H, 1 if args.decode else L, d, generator=gen) * 2 - 1) * 4
k = torch.rand(1, H, L, d, generator=gen) * 2 - 1
v = torch.rand(1, H, L, d, generator=gen) * 2 - 1

if args.decode:
    flops = 4.0 * H * L * d
elif causal:
    flops = 4.0 * H * d * L * (L + 1) / 2
else:
    flops = 4.0 * H * L * L * d
header = "kernel,params,heads,len,dim,threads,rep,seconds,gflops," + ",".join(NAMES) + ",rel_err\n"
params = "causal=1" if causal else ""


if args.kernel == "torch":
    def run():
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
else:
    mask = torch.full((L, L), -math.inf).triu(1) if causal else None

    def run():
        scores = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(d))
        if mask is not None:
            scores += mask
        return torch.softmax(scores, dim=-1) @ v


with torch.inference_mode(), open(args.out, "a") as f:
    if f.tell() == 0:
        f.write(header)
    for _ in range(args.warmup):
        run()
    for rep in range(args.reps):
        c0 = counters.read()
        t0 = time.perf_counter()
        run()
        s = time.perf_counter() - t0
        counts = ",".join(str(c) for c in PerfCounters.delta(c0, counters.read()))
        f.write(f"{args.kernel},{params},{H},{L},{d},{threads},{rep},{s:.9f},{flops / s / 1e9:.4f},{counts},\n")
        f.flush()
        os.fsync(f.fileno())
