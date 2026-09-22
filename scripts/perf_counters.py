"""Hardware counters for Python benchmarks. Linux x86-64 only.
Create the object before threads start so that they inherit the counters."""
import ctypes
import os
import struct

NAMES = ("cycles", "instructions", "l2_miss", "l2pf_l3", "l2pf_dram", "dram_fills")
_PERF_TYPE_HARDWARE, _PERF_TYPE_RAW = 0, 4
# Generic cycles and instructions, then raw AMD Zen cache events. The codes
# are explained in bench/perf_counters.hpp.
_EVENTS = ((_PERF_TYPE_HARDWARE, 0), (_PERF_TYPE_HARDWARE, 1), (_PERF_TYPE_RAW, 0x0964),
           (_PERF_TYPE_RAW, 0xFF71), (_PERF_TYPE_RAW, 0xFF72), (_PERF_TYPE_RAW, 0x0844))
_SYS_PERF_EVENT_OPEN = 298
_FORMAT_ENABLED_RUNNING = 1 | 2
_INHERIT, _EXCLUDE_KERNEL, _EXCLUDE_HV = 1 << 1, 1 << 5, 1 << 6


class PerfCounters:
    def __init__(self):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fds = []
        for kind, config in _EVENTS:
            # struct perf_event_attr in its first 64-byte layout.
            attr = struct.pack("<IIQQQQQIIQ", kind, 64, config, 0, 0, _FORMAT_ENABLED_RUNNING,
                               _INHERIT | _EXCLUDE_KERNEL | _EXCLUDE_HV, 0, 0, 0)
            self.fds.append(libc.syscall(_SYS_PERF_EVENT_OPEN, ctypes.c_char_p(attr), 0, -1, -1, 0))

    def read(self):
        """Returns one (value, time enabled, time running) triple per event. Zeros if unavailable."""
        sample = []
        for fd in self.fds:
            data = os.read(fd, 24) if fd >= 0 else b""
            sample.append(struct.unpack("<QQQ", data) if len(data) == 24 else (0, 0, 0))
        return sample

    @staticmethod
    def delta(before, after):
        """Counts between two samples, scaled up if the event ran only part of the time. -1 if unknown."""
        counts = []
        for (v0, e0, r0), (v1, e1, r1) in zip(before, after):
            counts.append(int((v1 - v0) * (e1 - e0) / (r1 - r0)) if r1 > r0 else -1)
        return counts
