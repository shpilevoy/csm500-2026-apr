"""Works out block sizes for the packed GEMM kernel from the cache sizes.

Each packed piece should stay in one cache level while it is reused. This is a
simplified form of the model of Low et al. (2016).

Run:  analytic_blocks.py [--threads N]
"""
import argparse
import glob
from pathlib import Path

MR, NR = 6, 16
FLOAT_BYTES = 4
# Sizes copied from the BLIS configuration for AMD Zen. They are the defaults of the kernel.
BLIS_DEFAULTS = {"mc": 144, "kc": 256, "nc": 4080}


def parse_size(text):
    """Turns a sysfs size such as 32K into bytes."""
    text = text.strip()
    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    if text[-1] in units:
        return int(text[:-1]) * units[text[-1]]
    return int(text)


def shared_cpu_count(text):
    """Counts the CPUs in a sysfs list such as 0-15 or 0-1,8-9."""
    count = 0
    for part in text.strip().split(","):
        if "-" in part:
            first, last = part.split("-")
            count += int(last) - int(first) + 1
        else:
            count += 1
    return count


def read_caches(cpu=0):
    """Returns the data caches of one CPU as {level: {"bytes", "ways", "shared_by"}}."""
    caches = {}
    for index in sorted(glob.glob(f"/sys/devices/system/cpu/cpu{cpu}/cache/index*")):
        index = Path(index)
        if (index / "type").read_text().strip() not in ("Data", "Unified"):
            continue
        caches[int((index / "level").read_text())] = {
            "bytes": parse_size((index / "size").read_text()),
            "ways": int((index / "ways_of_associativity").read_text()),
            "shared_by": shared_cpu_count((index / "shared_cpu_list").read_text()),
        }
    return caches


def usable(cache):
    """Bytes left when one way of the cache is kept free for other data."""
    return cache["bytes"] * (cache["ways"] - 1) // cache["ways"]


def round_down(value, step):
    return max(step, value // step * step)


def block_sizes(caches, threads=1, smt=2):
    """Returns {"kc", "mc", "nc"} for the given caches and number of GEMM threads."""
    # L1 holds one slice of A and one slice of B. L2 holds the packed block of A.
    # L3 holds the packed block of B. It is shared, so each thread gets an equal part.
    kc = round_down(usable(caches[1]) // ((MR + NR) * FLOAT_BYTES), 8)
    l2_left = usable(caches[2]) - kc * NR * FLOAT_BYTES
    mc = round_down(l2_left // (kc * FLOAT_BYTES), MR)
    cores_sharing = max(1, caches[3]["shared_by"] // smt)
    l3_share = usable(caches[3]) // min(threads, cores_sharing)
    nc = round_down(l3_share // (kc * FLOAT_BYTES), NR)
    return {"kc": kc, "mc": mc, "nc": nc}


def footprint(sizes):
    """Bytes of each packed piece for a set of block sizes."""
    return {
        "L1: slices of A and B": (MR + NR) * sizes["kc"] * FLOAT_BYTES,
        "L2: packed A": sizes["mc"] * sizes["kc"] * FLOAT_BYTES,
        "L3: packed B": sizes["nc"] * sizes["kc"] * FLOAT_BYTES,
    }


def as_settings(sizes):
    """The sizes in the form that run_sweep.py --set expects."""
    return " ".join(f"{name}={sizes[name]}" for name in ("mc", "kc", "nc"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--settings", action="store_true", help="print only the --set arguments")
    args = parser.parse_args()
    caches = read_caches()
    sizes = block_sizes(caches, args.threads)
    if args.settings:
        print(as_settings(sizes))
        return
    for level, cache in sorted(caches.items()):
        print(f"L{level}: {cache['bytes'] // 1024} KiB, {cache['ways']} ways, shared by {cache['shared_by']} CPUs")
    for label, chosen in (("from the cache sizes", sizes), ("BLIS defaults", BLIS_DEFAULTS)):
        print(f"\n{label}: {as_settings(chosen)}")
        for piece, size in footprint(chosen).items():
            print(f"  {piece}: {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
