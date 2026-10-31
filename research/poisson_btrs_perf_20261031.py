"""Independent CPU benchmark for PR #41228's Poisson rejection hot path.

Run in an environment using this PR's exact JAX checkout, e.g.:
  python research/poisson_btrs_perf_20261031.py

Source-extracted rejection loop using public JAX PRNG helpers. This is *not*
upstream CI nor a production throughput benchmark. Warmup excludes JIT compile.
"""
import json
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln
from jax._src.random.core import _poisson_log_pmf

jax.config.update("jax_enable_x64", True)


def _old_log_pmf(k, lam):
    return -lam + k * jnp.log(lam) - gammaln(k + 1)


def _make_sampler(log_pmf, sample_count):
    def sample(key, lam):
        b = 0.931 + 2.53 * jnp.sqrt(lam)
        a = -0.059 + 0.02483 * b
        inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
        v_r = 0.9277 - 3.6224 / (b - 2)

        def body(carry):
            i, k_out, accepted, key = carry
            key, key0, key1 = jax.random.split(key, 3)
            u = jax.random.uniform(key0, (sample_count,), dtype=lam.dtype) - 0.5
            v = jax.random.uniform(key1, (sample_count,), dtype=lam.dtype)
            u_shifted = 0.5 - jnp.abs(u)
            k = jnp.floor((2 * a / u_shifted + b) * u + lam + 0.43)
            s = jnp.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
            t = log_pmf(k, lam)
            accept1 = (u_shifted >= 0.07) & (v <= v_r)
            reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
            accept = accept1 | (~reject & (s <= t))
            return i + 1, lax.select(accept, k, k_out), accepted | accept, key

        def cond(carry):
            i, _, accepted, _ = carry
            return (~accepted).any() & (i < 100)

        initial = (0, jnp.full((sample_count,), -1., dtype=lam.dtype),
                   jnp.zeros((sample_count,), dtype=bool), key)
        return lax.while_loop(cond, body, initial)[1].astype(jnp.int32)

    return jax.jit(sample)


def _measure(sample, key, lam, repetitions=15):
    sample(key, lam).block_until_ready()  # compiled first
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter_ns()
        sample(key, lam).block_until_ready()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    values = sample(key, lam)
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "mean": float(jnp.mean(values, dtype=jnp.float64)),
        "variance": float(jnp.var(values, dtype=jnp.float64)),
        "negative_sentinel_count": int(jnp.sum(values == -1)),
    }


if __name__ == "__main__":
    data = {
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "note": "source-extracted BTRS with public PRNG, not full upstream CI",
        "samples": 8192,
        "timing_repetitions": 15,
        "cases": [],
    }
    key = jax.random.PRNGKey(1651)
    for dtype in (jnp.float32, jnp.float64):
        for rate in (100., 1e6, 1e8):
            lam = jnp.asarray(rate, dtype=dtype)
            old = _measure(_make_sampler(_old_log_pmf, 8192), key, lam)
            new = _measure(_make_sampler(_poisson_log_pmf, 8192), key, lam)
            data["cases"].append({
                "dtype": str(dtype), "lambda": rate,
                "baseline": old, "proposed": new,
                "median_latency_ratio": new["median_ms"] / old["median_ms"],
            })
    print(json.dumps(data, indent=2))
