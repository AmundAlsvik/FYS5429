"""Runtime benchmarks and speedup and break even helpers.

Adapted from original timing experiments. Codex and ChatGPT helped create the
break even calculation and refine the timing summary helpers.
"""

from __future__ import annotations

import time
import numpy as np

from common import BENCH_N, BENCH_WARMUP


def _time_once(fn):
    """Run a zero argument callable once and return elapsed seconds as a float."""
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_deploy_times(M0, M1, Vmin, Vmax, n_deploy, k, B_rstar=None):
    """Time full and reduced PMM scans over a scalar grid and return median runtimes.

    This timing helper was organized with Codex and ChatGPT from the original
    notebook timing cells.
    """
    Vg = np.linspace(Vmin, Vmax, int(n_deploy))

    def full_run():
        """Run one full eigensolve scan over the deployment grid and return None."""
        for V in Vg:
            H = M0 + V * M1
            _ = np.linalg.eigvalsh(H)[:k]

    def red_run(B):
        """Run one reduced eigensolve scan for basis B and return None."""
        for V in Vg:
            H = M0 + V * M1
            Hr = B.T @ H @ B
            _ = np.linalg.eigvalsh(Hr)[:k]

    for _ in range(BENCH_WARMUP):
        full_run()
        if B_rstar is not None:
            red_run(B_rstar)

    full_times = [_time_once(full_run) for _ in range(BENCH_N)]
    t_full = float(np.median(full_times))

    t_rstar = None
    if B_rstar is not None:
        times_r = [_time_once(lambda: red_run(B_rstar)) for _ in range(BENCH_N)]
        t_rstar = float(np.median(times_r))

    return t_full, t_rstar


def break_even_count(t_train, t_full, t_red):
    """Estimate how many deployments amortize preprocessing time and return a float count.

    Codex and ChatGPT helped add this summary metric for the compression
    runtime comparisons.
    """
    if (t_red is None) or (t_full <= t_red):
        return float("inf")
    return float(t_train / (t_full - t_red))


def pmm_runtime_row(case, compression, regime, n_deploy):
    """Benchmark one PMM and compressed PMM deployment setting and return one table row.

    This helper gathers the timing and break even quantities that are reported
    in the final PMM compression tables. It was suggested by Codex and implemented by me.
    """
    t_pmm, t_red = benchmark_deploy_times(
        case["M0"],
        case["M1"],
        -1.0,
        1.0,
        n_deploy,
        int(case["k"]),
        B_rstar=compression["B_rstar"],
    )
    preprocess = float(compression["compression_preprocess_time_s"])
    total_reduced = preprocess + float(t_red)
    return {
        "regime": regime,
        "n": int(case["n_pmm"]),
        "seed": int(case["seed"]),
        "epochs": int(case["epochs"]),
        "n_deploy": int(n_deploy),
        "r_star": int(compression["r_star"]),
        "r_star_over_k": float(compression["r_star_over_k"]),
        "e_rstar": float(compression["e_rstar"]),
        "pmm_scan_time_s": float(t_pmm),
        "pmm_compressed_scan_time_s": float(t_red),
        "compression_preprocess_time_s": preprocess,
        "total_reduced_time_s": total_reduced,
        "break_even_scan_count": break_even_count(preprocess, t_pmm, t_red),
        "raw_speedup": float(t_pmm / t_red) if t_red and t_red > 0 else np.inf,
        "amortized_speedup": float(t_pmm / total_reduced) if total_reduced > 0 else np.inf,
    }
