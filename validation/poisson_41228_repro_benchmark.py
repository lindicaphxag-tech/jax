#!/usr/bin/env python3
"""Independent same-host old-vs-patched JAX Poisson performance/variance audit.

This is NOT a benchmark result by itself. Use the separate GitHub Actions
workflow to run both commits with identical jaxlib, seeds, shape and CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import jaxlib


def run_case(rate: float, batch_size: int, repeats: int) -> dict:
    # The compilation and a warm-up execution occur OUTSIDE the timer.
    f = jax.jit(lambda key: jax.random.poisson(
        key, lam=jnp.asarray(rate, dtype=jnp.float32), shape=(batch_size,)))
    seeds = [jax.random.key(20261120 + i) for i in range(repeats)]
    f(seeds[0]).block_until_ready()
    durations = []
    samples = []
    for key in seeds:
        start = time.perf_counter_ns()
        y = f(key)
        y.block_until_ready()
        durations.append((time.perf_counter_ns() - start) / 1e6)
        samples.append(np.asarray(y, dtype=np.float64))
    observed = np.concatenate(samples)
    if observed.size != batch_size * repeats:
        raise RuntimeError("sample count changed unexpectedly")
    if not np.isfinite(observed).all():
        raise RuntimeError("non-finite sample values")
    return {
        "rate": rate,
        "count": int(observed.size),
        "sample_mean": float(observed.mean()),
        "sample_variance": float(observed.var(ddof=1)),
        "variance_ratio_to_rate": float(observed.var(ddof=1) / rate),
        "elapsed_ms_each": durations,
        "median_elapsed_ms": float(statistics.median(durations)),
        "mean_elapsed_ms": float(statistics.mean(durations)),
        "batch_size": batch_size,
        "repeats": repeats,
        "key_seed_first": 20261120,
        "compile_excluded": True,
        "synchronize_every_call": True,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path,
                   help="Expected JAX checkout root; fail if a wheel was imported")
    p.add_argument("--revision", required=True, choices=("base", "patched"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    if args.batch_size < 1024 or args.repeats < 3:
        raise ValueError("need enough independently sampled observations")
    imported = Path(jax.__file__).resolve()
    expected = (args.source.resolve() / "jax").resolve()
    if expected not in imported.parents:
        raise RuntimeError(
            f"wrong JAX checkout: imported={imported}; expected under={expected}")
    if jax.default_backend() != "cpu":
        raise RuntimeError("expected CPU-only matched benchmark")
    results = {
        "revision": args.revision,
        "jax_source": str(imported),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "platform": platform.platform(),
        "device": str(jax.devices()[0]),
        "cases": [run_case(rate, args.batch_size, args.repeats)
                  for rate in (1e4, 1e8)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
