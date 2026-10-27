"""Audit the global all-central gate used by JAX #41228's Poisson BTRS fast path.

This is a source-matched CPU diagnostic, not an exact-upstream-checkout run.
It imports the installed JAX sampler body, replaces only its acceptance PMF
call, and compares the full stable PMF with the official PR's central-path
predicate/conditional. Run with JAX, NumPy installed:
    python ci/poisson_fastpath_gate_audit_20261028.py
"""
import inspect
import json
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lax import special as lax_special

try:
    from jax._src.random import core as source
    if not hasattr(source, "_poisson_rejection"):
        raise ImportError
except ImportError:
    from jax._src import random as source


def full(k, lam):
    safe_k = jnp.maximum(k, 1)
    delta = safe_k - lam
    relative_delta = delta / safe_k
    small_delta = jnp.abs(relative_delta) <= 0.01
    t = jnp.where(small_delta, relative_delta, 0.0)
    series = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
                * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
    central_deviance = -safe_k * t * t * series
    use_far_tail = lam / safe_k < 0.5
    regular_relative_delta = jnp.where(use_far_tail, 0.0, relative_delta)
    regular_deviance = safe_k * lax.log1p(-regular_relative_delta) + delta
    deviance = jnp.where(small_delta, central_deviance, regular_deviance)
    square = safe_k * safe_k
    tail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / square) / square) / safe_k
    stirling = jnp.where(
        safe_k <= 10.0, source._stirling_approx_tail(safe_k - 1.0), tail
    )
    normalizer = 0.5 * (lax.log(safe_k) + math.log(2 * math.pi)) + stirling
    near = deviance - normalizer
    safe_rate = jnp.where(k == 0, 1.0, lam)
    ratio = safe_rate / safe_k
    positive = ratio > 0.0
    safe_ratio = jnp.where(positive, ratio, 1.0)
    log_ratio = jnp.where(
        positive, lax.log(safe_ratio), lax.log(safe_rate) - lax.log(safe_k)
    )
    far = safe_k * log_ratio + delta - normalizer
    result = jnp.where(use_far_tail, far, near)
    result = jnp.where(k == 0, -lam, result)
    return jnp.where(lam < 0.0, jnp.nan, result)


def central(k, lam):
    safe_k = jnp.maximum(k, 1)
    t = (safe_k - lam) / safe_k
    series = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
                * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
    deviance = -safe_k * t * t * series
    square = safe_k * safe_k
    tail = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / square) / square) / safe_k
    normalizer = 0.5 * (lax.log(safe_k) + math.log(2 * math.pi)) + tail
    return deviance - normalizer


def central_gate(k, lam):
    safe_k = jnp.maximum(k, 1)
    relative_delta = (safe_k - lam) / safe_k
    return (
        (lam >= 1e6)
        & (jnp.abs(relative_delta) <= 0.01)
        & (safe_k > 10)
        & (lam > 0)
    )


def gated(k, lam):
    return lax.cond(
        jnp.all(central_gate(k, lam)),
        lambda _: central(k, lam),
        lambda _: full(k, lam),
        operand=None,
    )


def instrumented_sampler(pmf):
    source_text = inspect.getsource(source._poisson_rejection)
    old_call = "t = -lam + k * log_lam - lax_special.lgamma(k + 1)"
    new_call = "t = _audit_pmf(k, lam)"
    namespace = dict(vars(source), _audit_pmf=pmf)
    if old_call in source_text:
        source_text = source_text.replace(old_call, new_call)
        source_text = source_text.replace("  log_lam = lax.log(lam)\n", "")
    elif "t = _poisson_log_pmf(k, lam)" in source_text:
        source_text = source_text.replace("t = _poisson_log_pmf(k, lam)", new_call)
    else:
        raise RuntimeError("Unknown installed BTRS source; refusing to guess")
    exec(compile(source_text, "<source-matched-btrs-audit>", "exec"), namespace)
    return namespace["_poisson_rejection"]


def gate_activation_rate(rate, n, trials=80):
    lam = jnp.asarray(rate, jnp.float32)
    b = 0.931 + 2.53 * lax.sqrt(lam)
    a = -0.059 + 0.02483 * b
    hit = 0
    for seed in range(trials):
        key = jax.random.key(seed + 712)
        keys = jax.random.split(key, 3)
        u = jax.random.uniform(keys[1], (n,), dtype=jnp.float32) - 0.5
        u_shifted = 0.5 - jnp.abs(u)
        k = jnp.floor((2 * a / u_shifted + b) * u + lam + 0.43)
        hit += int(bool(jnp.all(central_gate(k, lam))))
    return {"hits": hit, "trials": trials, "fraction": hit / trials}


def benchmark(stable, fast, rate, n, repetitions=24):
    lam = jnp.asarray(rate, jnp.float32)
    stable_key = jax.random.key(20261028)
    args = (stable_key, lam, (n,), np.dtype("int32"), 100)
    stable(*args).block_until_ready()
    fast(*args).block_until_ready()
    agreement = []
    for seed in (7, 23, 71):
        test_args = (jax.random.key(seed), lam, (n,), np.dtype("int32"), 100)
        agreement.append(float(np.mean(
            np.asarray(stable(*test_args)) == np.asarray(fast(*test_args))
        )))
    timings = {"stable": [], "gated": []}
    for i in range(repetitions):
        order = [("stable", stable), ("gated", fast)]
        if i % 2:
            order.reverse()
        test_args = (jax.random.key(i + 200), lam, (n,), np.dtype("int32"), 100)
        for label, fn in order:
            start = time.perf_counter_ns()
            fn(*test_args).block_until_ready()
            timings[label].append((time.perf_counter_ns() - start) / 1e6)
    med = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "lambda": rate,
        "n": n,
        "same_key_sample_agreement_min": min(agreement),
        "stable_median_ms": med["stable"],
        "gated_median_ms": med["gated"],
        "gated_over_stable": med["gated"] / med["stable"],
        "stable_p90_ms": float(np.percentile(timings["stable"], 90)),
        "gated_p90_ms": float(np.percentile(timings["gated"], 90)),
    }


def main():
    stable_sampler = instrumented_sampler(full)
    gated_sampler = instrumented_sampler(gated)
    activation = []
    for rate in (1e6, 3e6, 1e7, 1e8, 1e9):
        for n in (256, 1024, 8192, 32768):
            row = {"lambda": rate, "n": n, **gate_activation_rate(rate, n)}
            activation.append(row)
            print("gate", json.dumps(row), flush=True)
    results = [
        benchmark(stable_sampler, gated_sampler, rate, n)
        for rate in (1e6, 1e7, 1e8, 1e9)
        for n in (1024, 8192, 32768)
    ]
    for row in results:
        print("benchmark", json.dumps(row), flush=True)
    payload = {
        "jax_version": jax.__version__,
        "source_module": source.__name__,
        "device": str(jax.devices()[0]),
        "scope": "source-matched installed JAX CPU; not exact upstream checkout or official CI",
        "gate_activation": activation,
        "timings": results,
    }
    with open("poisson_fastpath_gate_audit_20261028.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("PASS: audit completed; no universal speedup claim", flush=True)


if __name__ == "__main__":
    main()
