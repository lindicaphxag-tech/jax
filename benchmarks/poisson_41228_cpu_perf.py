#!/usr/bin/env python3
"""Standalone opt-in CPU microbenchmark for upstream JAX PR #41228.

Run from this fork checkout, with a matching locally installed JAX build:
  python benchmarks/poisson_41228_cpu_perf.py --repeats 9 --out /tmp/poisson_perf.json

This benchmarks the log-PMF kernel ONLY, not end-to-end random.poisson.
No PR CI changes, no benchmark result files committed as upstream truth.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from jax._src.random.core import _poisson_log_pmf


def direct_log_pmf(k, lam):
    return -lam + k * lax.log(lam) - lax.lgamma(k + 1)


def measure(fn, k, lam, repeats):
    fn = jax.jit(fn)
    t0 = time.perf_counter()
    y = fn(k, lam)
    y.block_until_ready()
    first_ms = (time.perf_counter() - t0) * 1e3
    latency = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(k, lam).block_until_ready()
        latency.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(latency)), float(first_ms), np.asarray(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--out", type=Path, default=Path("poisson_41228_cpu_result.json"))
    args = p.parse_args()
    if args.repeats < 3:
        p.error("--repeats must be at least 3")
    cases = [
        (20.0, 1, "central"),
        (20.0, 4096, "central"),
        (1e5, 4096, "central"),
        (1e8, 4096, "central"),
        (1e8, 65536, "central"),
        (1e8, 4096, "tail"),
    ]
    output = []
    for rate, size, region in cases:
        offset = np.arange(size) % 31 - 15
        if region == "central":
            k = np.rint(rate + offset * np.sqrt(rate) / 8)
        else:
            k = np.rint(rate / 3 + offset * np.sqrt(rate) / 10)
        k = jnp.asarray(np.maximum(k, 1).astype(np.float32))
        lam = jnp.asarray(np.full(size, rate, dtype=np.float32))
        base_ms, base_first, base = measure(direct_log_pmf, k, lam, args.repeats)
        stable_ms, stable_first, stable = measure(_poisson_log_pmf, k, lam, args.repeats)
        output.append({
            "rate": rate, "size": size, "region": region,
            "direct_median_ms": base_ms, "stable_median_ms": stable_ms,
            "stable_over_direct": stable_ms / base_ms,
            "direct_first_call_ms": base_first,
            "stable_first_call_ms": stable_first,
            "max_abs_direct_stable_difference": float(np.max(np.abs(base - stable))),
            "stable_all_finite": bool(np.isfinite(stable).all()),
        })
        print(output[-1], flush=True)
    evidence = {
        "title": "JAX #41228 log-PMF kernel only, exact PR branch checkout required",
        "jax_version": jax.__version__, "jax_file": str(jax.__file__),
        "jax_backend": jax.default_backend(), "platform": platform.platform(),
        "repeats": args.repeats, "cases": output,
    }
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
