"""Paired JAX #41228 sampler benchmark on the exact fork PR source.

The unchanged pre-fix rejection body is reproduced as a baseline. Candidate
uses the real jax._src.random.core._poisson_rejection. Fixed keys, warm JIT,
alternating order, and block_until_ready avoid asynchronous timing artifacts.
Results describe GitHub's CPU runner only; no performance assertions.
"""
import json
import math
import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.random import core as m


@partial(jax.jit, static_argnums=(2, 3, 4))
def old_rejection(key, lam, shape, dtype, max_iters):
    log_lam = lax.log(lam)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2)

    def body_fn(carry):
        i, k_out, accepted, key = carry
        key, subkey_0, subkey_1 = m._split(key, 3)
        u = m.uniform(subkey_0, shape, lam.dtype) - 0.5
        v = m.uniform(subkey_1, shape, lam.dtype)
        u_shifted = 0.5 - abs(u)
        k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
        t = -lam + k * log_lam - m.lax_special.lgamma(k + 1)
        accept1 = (u_shifted >= 0.07) & (v <= v_r)
        reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
        accept2 = s <= t
        accept = accept1 | (~reject & accept2)
        k_out = lax.select(accept, k, k_out)
        accepted |= accept
        return i + 1, k_out, accepted, key

    def cond_fn(carry):
        i, k_out, accepted, key = carry
        return (~accepted).any() & (i < max_iters)

    k_init = lax.full_like(lam, -1, lam.dtype, shape)
    accepted = lax.full_like(lam, False, np.dtype("bool"), shape)
    k = m.lax_control_flow.while_loop(
        cond_fn, body_fn, (0, k_init, accepted, key))[1]
    return k.astype(dtype)


def run_case(rate, n, blocks=10):
    lam = jnp.asarray(rate, dtype=jnp.float32)
    key = jax.random.key(437)
    fns = {"before": old_rejection, "after": m._poisson_rejection}
    for fn in fns.values():
        for b in range(4):
            fn(jax.random.fold_in(key, b), lam, (n,), np.dtype("int32"),
               100).block_until_ready()
    paired = []
    for b in range(blocks):
        block_key = jax.random.fold_in(key, 1000 + b)
        order = ("before", "after") if b % 2 == 0 else ("after", "before")
        values = {}
        for name in order:
            start = time.perf_counter_ns()
            fns[name](block_key, lam, (n,), np.dtype("int32"),
                      100).block_until_ready()
            values[name] = (time.perf_counter_ns() - start) / 1e6
        paired.append(values)
    metrics = {}
    for name, fn in fns.items():
        values = np.asarray(fn(jax.random.fold_in(key, 9000), lam, (n,),
                              np.dtype("int32"), 100), dtype=np.float64)
        metrics[name] = {
            "mean": float(values.mean()),
            "variance": float(values.var()),
            "variance_relative_error": float(abs(values.var() / rate - 1)),
            "median_ms": statistics.median(p[name] for p in paired),
        }
    return {
        "lambda": rate, "n": n, "pair_count": blocks,
        "median_paired_time_ratio_after_over_before": statistics.median(
            p["after"] / p["before"] for p in paired),
        "metrics": metrics, "pair_ms": paired,
    }


def main():
    print("JAX", jax.__version__, "DEVICES", jax.devices(), flush=True)
    results = []
    for rate in (1e4, 1e8):
        for n in (8192, 32768):
            row = run_case(rate, n)
            results.append(row)
            print(json.dumps(row), flush=True)
    print("SUMMARY_JSON", json.dumps({
        "jax": jax.__version__, "device": str(jax.devices()[0]),
        "note": "exact fork PR source, one CI CPU machine, not GPU",
        "results": results,
    }), flush=True)


if __name__ == "__main__":
    main()
