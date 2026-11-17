"""Exact-PR-head, paired CPU benchmark for JAX PR #41228.

Run from a checkout of jax-ml/jax at PR head 9428e21c:
    python research/poisson_41228_benchmark.py --n 65536 --seeds 9

The candidate uses the actual core._poisson_rejection at this branch HEAD;
the baseline reproduces the original pre-PR rejection sampler with its
cancellation-prone acceptance log-PMF. Neither compile time nor warm-up
is included in steady-state latency. No numerical result is pre-filled.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import statistics
import time

import jax
from jax import lax
from jax._src.random import core
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnums=(2, 3, 4))
def original_rejection(key, lam, shape, dtype, max_iters, counts=None):
    """Pre-#41228 sampler, adapted from the original core.py function."""
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)
    draw_u = (
        (lambda sk: core.uniform(sk, shape, lam.dtype))
        if counts is None
        else (lambda sk: core._threefry_uniform_f32(sk, *counts))
    )

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = core._split(key, 3)

        u = draw_u(subkey_0) - 0.5
        v = draw_u(subkey_1)
        u_shifted = 0.5 - abs(u)
        k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
        t = -lam + k * log_lam - lax.lgamma(k + 1)

        accept1 = (u_shifted >= 0.07) & (v <= v_r)
        reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
        accept2 = s <= t
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(accept & ~accepted, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, _, accepted, _ = carry
        return (~accepted).any() & (i < max_iters)

    k_init = lax.full_like(lam, -1, lam.dtype, shape)
    accepted = lax.full_like(lam, False, np.dtype("bool"), shape)
    return lax.while_loop(
        cond_fn, body_fn, (0, k_init, accepted, key)
    )[1].astype(dtype)


def bench_call(fn, key, lam):
    t0 = time.perf_counter_ns()
    result = fn(key, lam)
    result.block_until_ready()
    return (time.perf_counter_ns() - t0) / 1e6


def sample_stats(samples, rate):
    z = np.concatenate(samples).astype(np.float64)
    return {
        "n": int(z.size),
        "mean": float(z.mean()),
        "variance": float(z.var()),
        "variance_error_pct": 100 * (float(z.var()) / rate - 1),
    }


def run(args):
    dtype = np.dtype("int32")
    old = lambda key, lam: original_rejection(
        key, lam, (args.n,), dtype, args.max_iters
    )
    new = lambda key, lam: core._poisson_rejection(
        key, lam, (args.n,), dtype, args.max_iters
    )
    records = []

    for rate in args.rates:
        lam = jnp.full((args.n,), rate, dtype=jnp.float32)
        warm_key = jax.random.key(999)
        # Compile both variants and warm their caches outside the timed region.
        for fn in (old, new):
            fn(warm_key, lam).block_until_ready()
        for _ in range(4):
            old(warm_key, lam).block_until_ready()
            new(warm_key, lam).block_until_ready()

        old_times = []
        new_times = []
        old_samples = []
        new_samples = []
        for i in range(args.seeds):
            key = jax.random.key(i + 1000)
            # Balanced order with two repetitions of each variant per seed.
            order = [
                ("old", old), ("new", new), ("new", new), ("old", old)
            ]
            if i % 2:
                order.reverse()
            for name, fn in order:
                t = bench_call(fn, key, lam)
                (old_times if name == "old" else new_times).append(t)
            old_samples.append(np.asarray(old(key, lam)))
            new_samples.append(np.asarray(new(key, lam)))

        a = np.asarray(old_times).reshape(-1, 2).mean(axis=1)
        b = np.asarray(new_times).reshape(-1, 2).mean(axis=1)
        ratios = b / a
        rng = np.random.default_rng(2026)
        replicates = np.median(
            ratios[rng.integers(0, len(ratios), (10000, len(ratios)))],
            axis=1,
        )
        rec = {
            "rate": rate,
            "sample_count_per_variant": args.n * args.seeds,
            "old": sample_stats(old_samples, rate),
            "new": sample_stats(new_samples, rate),
            "old_latency_median_ms": statistics.median(old_times),
            "new_latency_median_ms": statistics.median(new_times),
            "paired_median_new_over_old": float(np.median(ratios)),
            "paired_bootstrap_95pct": [
                float(x) for x in np.quantile(replicates, [0.025, 0.975])
            ],
            "raw_old_ms": old_times,
            "raw_new_ms": new_times,
        }
        print(json.dumps({k: v for k, v in rec.items()
                          if not k.startswith("raw_")}), flush=True)
        records.append(rec)

    output = {
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "head_under_test": "9428e21c09e6a19cadbf9ed02d6ec78b165f6246",
        "method": "exact candidate vs emulated previous sampler, ABBA timed",
        "max_iters": args.max_iters,
        "rates": args.rates,
        "n": args.n,
        "seeds": args.seeds,
        "limitations": "CPU-only; test script must run on exact head; "
                       "no accelerator evidence; CI unaffected",
        "records": records,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=65536)
    parser.add_argument("--seeds", type=int, default=9)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[30.0, 10000.0, 1e8])
    parser.add_argument("--output", type=str,
                        default="poisson_41228_exact_head_results.json")
    run(parser.parse_args())
