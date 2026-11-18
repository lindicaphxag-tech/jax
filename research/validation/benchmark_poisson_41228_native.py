"""JAX #41228 native-source sampling and numerical benchmark.

Run against *two independent source checkouts* of exact Git SHAs:
  python path/to/benchmark_poisson_41228_native.py --out baseline.json
  python path/to/benchmark_poisson_41228_native.py --out candidate.json

The Python interpreter must import jax from the corresponding checkout.
Never compare two implementations in the same interpreter through sys.path
swaps; imported JAX modules persist in sys.modules.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import poisson as scipy_poisson


def summary_ms(call, *, warmups=5, repeats=40):
    # The compile+first-execution measurement is separate from warmed execution.
    start = time.perf_counter_ns()
    call().block_until_ready()
    first_ms = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmups):
        call().block_until_ready()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call().block_until_ready()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "first_call_including_compile_ms": first_ms,
        "median_ms": statistics.median(timings),
        "p90_ms": float(np.percentile(timings, 90)),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "warmups": warmups,
        "repeats": repeats,
    }


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_case(dtype, n, rate, *, seed):
    # Int32 supports our selected rates. Float32 rate is the critical case.
    if rate > np.iinfo(np.int32).max:
        raise ValueError("rate exceeds int32")
    rates = jnp.full((n,), rate, dtype=dtype)
    key = jax.random.key(seed)
    sampler = jax.jit(lambda k, lam: jax.random.poisson(k, lam, dtype=jnp.int32))
    sample_call = lambda: sampler(key, rates)
    sample_timing = summary_ms(sample_call)
    samples = np.asarray(sample_call(), dtype=np.float64)
    mean = float(np.mean(samples))
    variance = float(np.var(samples))
    expected_variance = float(np.asarray(rates[0]))
    # This fixed-seed sample is a correctness smoke test, NOT a distribution
    # validation: 1024 samples at large rate are noisy by design.
    pmf_offsets = jnp.asarray([-3., -1., 0., 1., 3.], dtype=dtype)
    k = jnp.round(rates[:5] + pmf_offsets * jnp.sqrt(rates[:5]))
    pmf = jax.jit(lambda counts, lam: jax.scipy.stats.poisson.logpmf(counts, lam))
    rates5 = rates[:5]
    pmf_call = lambda: pmf(k, rates5)
    pmf_timing = summary_ms(pmf_call)
    counts_numpy = np.asarray(k, dtype=np.float64)
    rates_numpy = np.asarray(rates5, dtype=np.float64)
    reference = scipy_poisson.logpmf(counts_numpy, rates_numpy)
    estimate = np.asarray(pmf_call(), dtype=np.float64)
    return {
        "dtype": str(np.dtype(dtype)),
        "n_rates": n,
        "requested_rate": rate,
        "effective_rate": expected_variance,
        "sampler": sample_timing,
        "logpmf": pmf_timing,
        "samples_mean": mean,
        "samples_variance": variance,
        "sample_mean_z": (
            (mean - expected_variance) / np.sqrt(expected_variance / n)
        ),
        "sample_variance_ratio": variance / expected_variance,
        "logpmf_max_abs_scipy": float(np.max(np.abs(estimate - reference))),
        "logpmf_counts": counts_numpy.tolist(),
        "logpmf_estimates": estimate.tolist(),
        "logpmf_scipy_reference": reference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-sha", required=True,
                        help="exact source commit expected for this execution")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat whole experiment with different deterministic seeds")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    sha = git_sha()
    if sha != args.require_sha:
        parser.error(f"checkout SHA {sha} != expected {args.require_sha}")
    source_path = Path(jax.__file__).resolve()
    checkout = Path.cwd().resolve()
    if not source_path.is_relative_to(checkout):
        parser.error(f"imported JAX {source_path} is outside checkout {checkout}")
    # Disable x64 *explicitly* for the float32 rejection hot path. The
    # experiment covers float64 separately where the JAX build supports it.
    records = []
    for iteration in range(args.repeats):
        for dtype in (np.float32, np.float64):
            jax.config.update("jax_enable_x64", dtype == np.float64)
            for rate in (3e6, 1e8):
                for n in (256, 2048):
                    records.append(run_case(dtype, n, rate, seed=iteration + 13))
    payload = {
        "git_sha": sha,
        "source_file": str(source_path),
        "jax_version": jax.__version__,
        "jax_file": jax.__file__,
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "cpu_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "XLA_FLAGS")
        },
        "warning": (
            "Fixed-key timing and sample moment smoke test. Does not establish "
            "rare-tail behavior, multi-GPU throughput, or end-to-end training."
        ),
        "records": records,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "git_sha": payload["git_sha"],
        "jax_version": payload["jax_version"],
        "backend": payload["backend"],
        "cases": len(records),
        "output": args.out,
    }))


if __name__ == "__main__":
    main()
