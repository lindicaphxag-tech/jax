"""CPU microbenchmark for the Poisson log-PMF in PR #41228.

Compares the previous direct expression with the proposed stable expression.
This is an isolated benchmark: it does not modify JAX source or sample RNGs.
"""
from __future__ import annotations

import json
import statistics
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import lax
from jax._src.random import core as random_core


def old_logpmf(k, lam):
    return -lam + k * jnp.log(lam) - jsp.special.gammaln(k + 1.0)


def proposed_logpmf(k, lam):
    # Kept in sync with jax/_src/random/core.py in PR #41228.
    safe_k = jnp.maximum(k, 1)
    delta = safe_k - lam
    relative_delta = delta / safe_k
    small_delta = jnp.abs(relative_delta) <= 0.01

    stirling_k_sq = safe_k * safe_k
    asymptotic_stirling_tail = (
        1.0 / 12
        - (1.0 / 360 - 1.0 / 1260 / stirling_k_sq) / stirling_k_sq
    ) / safe_k
    stirling_tail = jnp.where(
        safe_k <= 10.0,
        random_core._stirling_approx_tail(safe_k - 1.0),
        asymptotic_stirling_tail,
    )
    log_normalizer = (
        0.5 * (lax.log(safe_k) + np.log(2 * np.pi)) + stirling_tail
    )

    def central_path(_):
        t = relative_delta
        series = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
                    * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
        deviance = -safe_k * (t * t) * series
        return jnp.where(k == 0, -lam, deviance - log_normalizer)

    def general_path(_):
        t = jnp.where(small_delta, relative_delta, 0.0)
        series = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
                    * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
        central_deviance = -safe_k * (t * t) * series

        use_far_tail = lam / safe_k < 0.5
        regular_relative_delta = jnp.where(
            use_far_tail, 0.0, relative_delta
        )
        regular_deviance = (
            safe_k * lax.log1p(-regular_relative_delta) + delta
        )
        deviance = jnp.where(
            small_delta, central_deviance, regular_deviance
        )
        near_mean_log_pmf = deviance - log_normalizer

        safe_rate_for_log = jnp.where(k == 0, 1.0, lam)
        rate_ratio = safe_rate_for_log / safe_k
        positive_ratio = rate_ratio > 0.0
        safe_ratio = jnp.where(positive_ratio, rate_ratio, 1.0)
        log_ratio = jnp.where(
            positive_ratio,
            lax.log(safe_ratio),
            lax.log(safe_rate_for_log) - lax.log(safe_k),
        )
        far_tail_log_pmf = (
            safe_k * log_ratio + delta - log_normalizer
        )
        log_pmf = jnp.where(
            use_far_tail, far_tail_log_pmf, near_mean_log_pmf
        )
        return jnp.where(k == 0, -lam, log_pmf)

    result = lax.cond(
        jnp.all(small_delta), central_path, general_path, operand=None
    )
    return jnp.where(lam < 0.0, jnp.nan, result)


def median_ms(fn, k, lam, repeats=40):
    compiled = jax.jit(fn)
    compiled(k, lam).block_until_ready()  # compile and warm up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(k, lam)
        result.block_until_ready()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main():
    rng = np.random.default_rng(20261120)
    results = []
    for rate in (20.0, 1e3, 1e7, 1e8):
        for size in (1, 128, 1024, 8192):
            k_np = np.maximum(
                0.0,
                np.rint(rate + rng.normal(size=size) * np.sqrt(rate)),
            ).astype(np.float32)
            lam_np = np.full(k_np.shape, rate, dtype=np.float32)
            k = jnp.asarray(k_np)
            lam = jnp.asarray(lam_np)
            old_ms = median_ms(old_logpmf, k, lam)
            new_ms = median_ms(proposed_logpmf, k, lam)
            results.append({
                "rate": rate,
                "batch_size": size,
                "old_ms": old_ms,
                "proposed_ms": new_ms,
                "slowdown_ratio": new_ms / old_ms,
            })
    print(json.dumps({
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
