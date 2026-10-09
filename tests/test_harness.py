"""Tests of the measurement harness. Run with  .venv/bin/python -m pytest tests

A wrong harness gives wrong numbers without any error, so it is tested too.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from autotune import coordinate_descent  # noqa: E402
from perf_counters import PerfCounters  # noqa: E402
import run_sweep  # noqa: E402
from run_sweep import done_points, parse_shape, physical_cores  # noqa: E402

BUILD = ROOT / "build"
needs_build = pytest.mark.skipif(not (BUILD / "bench_gemm").exists(), reason="build the project first")


# Sweep runner
def test_parse_shape_expands_a_single_number():
    assert parse_shape("512", 3) == (512, 512, 512)
    assert parse_shape("8x1024x64", 3) == (8, 1024, 64)
    assert parse_shape("4096", 1) == (4096,)


def test_parse_shape_rejects_the_wrong_rank():
    with pytest.raises(SystemExit):
        parse_shape("8x1024", 3)


def test_physical_cores_are_distinct_and_plausible():
    cores = physical_cores()
    assert cores == sorted(set(cores))
    assert 1 <= len(cores) <= os.cpu_count()


def test_done_points_separates_block_size_settings(tmp_path):
    """Resuming must not treat a tuned run as the default run of the same point."""
    out = tmp_path / "r.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "params", "m", "n", "k", "threads", "rep"])
        w.writerows([["goto", "", 8, 8, 8, 1, 0], ["goto", "", 8, 8, 8, 1, 1], ["goto", "mc=12", 8, 8, 8, 1, 0]])
    done = done_points(out, ("m", "k", "n"))
    assert done[("goto", "", (8, 8, 8), 1)] == 2
    assert done[("goto", "mc=12", (8, 8, 8), 1)] == 1
    assert done[("goto", "", (8, 8, 8), 8)] == 0


def test_resume_refuses_results_made_by_other_code(tmp_path):
    out = tmp_path / "g.csv"
    out.write_text("kernel\n")
    out.with_suffix(".machine.json").write_text('{"source_sha256": "0000"}')
    with pytest.raises(SystemExit):
        run_sweep.check_provenance(out, {"source_sha256": "1111"})
    run_sweep.check_provenance(out, {"source_sha256": "0000"})
    run_sweep.check_provenance(tmp_path / "new.csv", {"source_sha256": "1111"})
    assert (tmp_path / "new.machine.json").exists()


# Autotuner
def test_coordinate_descent_finds_a_separable_optimum():
    calls = []

    def score(p):
        calls.append(dict(p))
        return -((p["a"] - 3) ** 2) - (p["b"] - 20) ** 2

    best, value = coordinate_descent(score, {"a": [1, 2, 3, 4], "b": [10, 20, 30]}, {"a": 1, "b": 10})
    assert best == {"a": 3, "b": 20}
    assert value == 0


def test_coordinate_descent_keeps_the_start_when_nothing_is_better():
    best, value = coordinate_descent(lambda p: -abs(p["a"] - 5), {"a": [1, 5, 9]}, {"a": 5})
    assert best == {"a": 5} and value == 0


def test_coordinate_descent_needs_several_passes_when_parameters_interact():
    # The best value of each parameter depends on the other. So one pass is not
    # enough and more passes keep improving the score.
    score = lambda p: -((p["a"] - 4) ** 2) - (p["b"] - p["a"]) ** 2  # noqa: E731
    grid = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
    one, score_one = coordinate_descent(score, grid, {"a": 1, "b": 1}, max_passes=1)
    many, score_many = coordinate_descent(score, grid, {"a": 1, "b": 1}, max_passes=10)
    assert one == {"a": 2, "b": 2}
    assert score_many > score_one
    # It stops at (3, 3), one step short of the true optimum (4, 4). Moving one
    # parameter alone no longer helps. This is the known limit of the method.
    assert many == {"a": 3, "b": 3}


# Hardware counters
def test_counter_delta_scales_time_sliced_events():
    before = [(100, 1000, 1000)]
    after = [(300, 3000, 2000)]  # counted during half of the interval
    assert PerfCounters.delta(before, after) == [400]


def test_counter_delta_reports_unknown_as_minus_one():
    assert PerfCounters.delta([(0, 0, 0)], [(0, 0, 0)]) == [-1]


def test_live_cycle_counter_advances_if_available():
    counters = PerfCounters()
    if counters.fds[0] < 0:
        pytest.skip("hardware counters are not accessible (kernel.perf_event_paranoid)")
    before = counters.read()
    sum(range(200_000))
    assert PerfCounters.delta(before, counters.read())[0] > 0


# Benchmark programs
def run(*args):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True)


@needs_build
@pytest.mark.parametrize("args", [
    ["bench_gemm", "--kernel", "no_such_kernel", "--m", 4, "--n", 4, "--k", 4],
    ["bench_gemm", "--kernel", "goto", "--m", 0, "--n", 4, "--k", 4],
    ["bench_gemm", "--kernel", "goto", "--m", 4, "--n", 4],
    ["bench_gemm", "--kernel", "goto", "--m", 4, "--n", 4, "--k", 4, "--set", "mc=-1"],
    ["bench_gemm", "--kernel", "goto", "--m", 4, "--n", 4, "--k", 4, "--set", "bogus=3"],
    ["bench_attn", "--kernel", "tiled", "--heads", 1, "--len", -3, "--dim", 4],
    ["bench_attn", "--kernel", "tiled", "--heads", 1, "--len", 8, "--dim", 4, "--set", "causal=2"],
    ["bench_decode", "--kernel", "no_such_kernel", "--heads", 1, "--len", 8, "--dim", 4],
    ["bench_machine", "--kernel", "triad"],
])
def test_invalid_input_is_rejected(args):
    result = run(BUILD / args[0], *args[1:])
    assert result.returncode != 0
    assert result.stderr


@needs_build
def test_results_file_gets_one_header_and_one_row_per_repetition(tmp_path):
    out = tmp_path / "g.csv"
    for _ in range(2):  # the second run appends
        assert run(BUILD / "bench_gemm", "--kernel", "goto", "--m", 32, "--n", 32, "--k", 32, "--reps", 3,
                   "--warmup", 1, "--check", "--out", out).returncode == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 6
    assert [r["rep"] for r in rows] == ["0", "1", "2"] * 2
    assert all(float(r["rel_err"]) < 1e-4 and float(r["gflops"]) > 0 for r in rows)


@needs_build
def test_causal_setting_is_recorded_and_halves_the_counted_work(tmp_path):
    out = tmp_path / "a.csv"
    base = [BUILD / "bench_attn", "--kernel", "tiled", "--heads", 1, "--len", 64, "--dim", 16, "--reps", 1,
            "--warmup", 0, "--check", "--out", out]
    assert run(*base).returncode == 0
    assert run(*base, "--set", "causal=1").returncode == 0
    full, causal = list(csv.DictReader(out.open()))
    assert (full["params"], causal["params"]) == ("", "causal=1")
    work = lambda r: float(r["gflops"]) * float(r["seconds"])  # noqa: E731
    assert work(causal) / work(full) == pytest.approx(65 / 128, rel=1e-2)
    assert float(causal["rel_err"]) < 1e-4


# Statistics
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analytic_blocks  # noqa: E402
import roofline_model  # noqa: E402
import stats  # noqa: E402


def test_bootstrap_interval_contains_the_median_and_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small, large = rng.normal(100, 5, 10), rng.normal(100, 5, 400)
    low_s, high_s = stats.bootstrap_median_ci(small)
    low_l, high_l = stats.bootstrap_median_ci(large)
    assert low_s <= np.median(small) <= high_s
    assert low_l <= np.median(large) <= high_l
    assert high_l - low_l < high_s - low_s


def test_bootstrap_interval_of_a_tiny_sample_is_its_range():
    assert stats.bootstrap_median_ci([3.0, 7.0]) == (3.0, 7.0)


def test_bootstrap_interval_is_repeatable():
    values = np.random.default_rng(2).normal(50, 3, 30)
    assert stats.bootstrap_median_ci(values) == stats.bootstrap_median_ci(values)


def test_ratio_interval_covers_a_known_ratio():
    rng = np.random.default_rng(3)
    low, high = stats.bootstrap_ratio_ci(rng.normal(200, 4, 30), rng.normal(100, 2, 30))
    assert low < 2.0 < high
    assert high - low < 0.1


def test_permutation_test_separates_different_samples_only():
    rng = np.random.default_rng(4)
    same_a, same_b = rng.normal(100, 5, 30), rng.normal(100, 5, 30)
    assert stats.permutation_test(same_a, same_b, resamples=2000) > 0.05
    assert stats.permutation_test(same_a, same_b + 20, resamples=2000) < 0.01


def test_relative_iqr():
    assert stats.relative_iqr([90, 100, 110]) == pytest.approx(0.1)


def frame(kernel, gflops, threads=1):
    return pd.DataFrame({"kernel": kernel, "params": "", "m": 8, "k": 8, "n": 8, "threads": threads,
                         "gflops": gflops})


def test_select_filters_by_kernel_and_point():
    data = pd.concat([frame("a", [1.0, 2.0]), frame("b", [3.0]), frame("a", [9.0], threads=8)])
    assert list(stats.select(data, "a", {"threads": 1}).gflops) == [1.0, 2.0]
    assert stats.select(data, "c", {}).empty


def test_between_run_spread_needs_two_runs_of_a_point():
    runs = [pd.concat([frame("a", [100.0, 100.0]), frame("b", [50.0])]), frame("a", [110.0, 110.0])]
    spread = stats.between_run_spread(runs)
    assert len(spread) == 1
    assert spread.iloc[0] == pytest.approx(0.1)


def test_parse_where():
    assert stats.parse_where("len=8192,threads=8") == {"len": 8192, "threads": 8}
    assert stats.parse_where("") == {}


# Block sizes from the cache sizes
CACHES = {1: {"bytes": 32 * 1024, "ways": 8, "shared_by": 2},
          2: {"bytes": 512 * 1024, "ways": 8, "shared_by": 2},
          3: {"bytes": 16 * 1024 ** 2, "ways": 16, "shared_by": 16}}


def test_sysfs_values_are_parsed():
    assert analytic_blocks.parse_size("32K") == 32768
    assert analytic_blocks.parse_size("16M") == 16 * 1024 ** 2
    assert analytic_blocks.parse_size("512") == 512
    assert analytic_blocks.shared_cpu_count("0-15") == 16
    assert analytic_blocks.shared_cpu_count("0-1,8-9") == 4
    assert analytic_blocks.shared_cpu_count("3") == 1


def test_block_sizes_fit_their_cache_and_respect_the_tile():
    sizes = analytic_blocks.block_sizes(CACHES)
    pieces = analytic_blocks.footprint(sizes)
    assert pieces["L1: slices of A and B"] <= 32 * 1024
    assert pieces["L2: packed A"] <= 512 * 1024
    assert pieces["L3: packed B"] <= 16 * 1024 ** 2
    assert sizes["mc"] % analytic_blocks.MR == 0
    assert sizes["nc"] % analytic_blocks.NR == 0
    assert sizes["kc"] % 8 == 0


def test_more_threads_share_the_last_cache():
    one, eight = analytic_blocks.block_sizes(CACHES, threads=1), analytic_blocks.block_sizes(CACHES, threads=8)
    assert eight["nc"] < one["nc"]
    assert (eight["kc"], eight["mc"]) == (one["kc"], one["mc"])


def test_the_machine_reports_three_cache_levels():
    caches = analytic_blocks.read_caches()
    if set(caches) != {1, 2, 3}:
        pytest.skip("this machine does not have three levels of data cache")
    assert caches[1]["bytes"] < caches[2]["bytes"] < caches[3]["bytes"]
    assert analytic_blocks.as_settings(analytic_blocks.block_sizes(caches)).startswith("mc=")


# Roofline model
def test_limit_is_the_smaller_of_the_two_roofs():
    assert roofline_model.limit(0.5, peak=1000, bandwidth=40) == 20
    assert roofline_model.limit(100, peak=1000, bandwidth=40) == 1000
    assert roofline_model.regime(0.5, 1000, 40) == "memory"
    assert roofline_model.regime(100, 1000, 40) == "compute"
    assert roofline_model.ridge(1000, 40) == 25


def test_decode_intensity_doubles_with_half_precision():
    assert roofline_model.decode_intensity(4) == 0.5
    assert roofline_model.decode_intensity(2) == 1.0
