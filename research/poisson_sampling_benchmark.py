"""End-to-end JAX Poisson sampler benchmark for review of PR #41228.

Run this SAME script in separate clean source checkouts of JAX main and the PR.
Outputs are deliberately not aggregated across different devices or builds.
Usage:
  python research/poisson_sampling_benchmark.py --output /tmp/poisson-main.json
  python research/poisson_sampling_benchmark.py --output /tmp/poisson-pr.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sample_case(rate, size, dtype, seeds, warmup, repetitions):
    mu = jnp.asarray(rate, dtype=dtype)
    fn = jax.jit(
        lambda key: jax.random.poisson(
            key, mu, shape=(size,), dtype=jnp.int32, method="exact"
        )
    )
    keys = [jax.random.key(seed) for seed in range(seeds)]
    for _ in range(warmup):
        fn(keys[0]).block_until_ready()
    samples = [np.asarray(fn(key).block_until_ready(), np.float64)
               for key in keys]
    moments = [
        {"mean": float(x.mean()), "variance": float(x.var(ddof=1))}
        for x in samples
    ]
    timings = []
    for iteration in range(repetitions):
        key = keys[iteration % seeds]
        start = time.perf_counter_ns()
        fn(key).block_until_ready()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "rate": float(rate),
        "size": size,
        "dtype": str(np.dtype(dtype)),
        "moment_estimates": moments,
        "median_ms": statistics.median(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "timings_ms": timings,
        "note": "JIT compilation excluded; deterministic keys, execution synchronized",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--rates", type=float, nargs="+", default=[3e6, 1e8])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.size < 512 or args.seeds < 2 or args.repetitions < 3:
        parser.error("require size >=512, seeds >=2, repetitions >=3")
    if any(not np.isfinite(x) or x <= 0 or x >= 2e9 for x in args.rates):
        parser.error("rates must be finite and within (0, 2e9) for int32 samples")
    jax.config.update("jax_enable_x64", bool(args.x64))
    result = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platform": platform.platform(),
        "parameters": vars(args),
        "cases": [],
    }
    dtypes = (jnp.float32, jnp.float64) if args.x64 else (jnp.float32,)
    for dtype in dtypes:
        for rate in args.rates:
            case = _sample_case(
                rate, args.size, dtype, args.seeds,
                args.warmup, args.repetitions,
            )
            result["cases"].append(case)
            print(
                f"{case['dtype']} rate={rate:g}: "
                f"median {case['median_ms']:.3f} ms, "
                f"variance/rate="
                f"{statistics.mean(m['variance'] for m in case['moment_estimates'])/rate:.4f}"
            )
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
