"""Fork-only native source gate for upstream JAX #41228 (current 9428e21 head).

Unlike a manually transcribed reproduction, imports the checked-out JAX
repository's own jax._src.random.core and public scipy.stats.poisson API.

Run: python research/poisson_41228_native_gate.py
The workflow also runs repository tests; none of these checks imply upstream
maintainer approval or accelerator throughput guarantees.
"""

from __future__ import annotations

import json

import jax
from jax._src.random import core
import jax.numpy as jnp
import numpy as np
import scipy.stats as scipy_stats


EXPECTED_PR_CORE_SHA = "52a1aba3a8d70166f19bad3dfc9d8eb9c71adbfb"
N = 32768
RATE = np.float32(1e8)


def check_public_logpmf() -> float:
    cases = []
    for rate in [1e7, 1e8, 1e9]:
        mu = np.float32(rate)
        offsets = np.asarray([-5, -3, -1, 0, 1, 3, 5], dtype=np.float32)
        k = np.rint(mu + offsets * np.sqrt(mu)).astype(np.float32)
        cases.extend([(float(x), float(mu)) for x in k])
    k = np.asarray([x[0] for x in cases], dtype=np.float32)
    mu = np.asarray([x[1] for x in cases], dtype=np.float32)
    actual = np.asarray(
        jax.jit(jax.scipy.stats.poisson.logpmf)(jnp.asarray(k), jnp.asarray(mu))
    )
    expected = scipy_stats.poisson.logpmf(k.astype(float), mu.astype(float))
    max_err = float(np.max(np.abs(actual.astype(float) - expected)))
    assert np.all(np.isfinite(actual)), "nonfinite central log-PMF"
    assert max_err < 5e-4, f"central log-PMF absolute error {max_err}"
    return max_err


def check_native_rejection() -> dict:
    lam = jnp.full((N,), RATE, dtype=jnp.float32)
    dtype = np.dtype("int32")
    fn = lambda key: core._poisson_rejection(key, lam, (N,), dtype, 100)
    samples = []
    # Each evaluation executes the *real checkout's* transformed-rejection
    # implementation, not a stand-in sampler.
    for seed in range(12):
        samples.append(np.asarray(fn(jax.random.key(seed))).astype(np.float64))
    data = np.concatenate(samples)
    result = {
        "rate": float(RATE),
        "samples": int(data.size),
        "mean": float(np.mean(data)),
        "variance": float(np.var(data)),
        "variance_relative_error": float(np.var(data) / float(RATE) - 1.0),
    }
    # 393,216 samples: 5% is far outside ordinary sampling variation. This
    # explicitly distinguishes the documented ~45% old variance inflation.
    assert abs(result["variance_relative_error"]) < 0.05, result
    assert abs(result["mean"] - float(RATE)) < 500, result
    return result


if __name__ == "__main__":
    assert jax.default_backend() == "cpu", jax.devices()
    error = check_public_logpmf()
    stats = check_native_rejection()
    print(json.dumps({
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "provenance": "actual checked-out JAX implementation",
        "max_central_logpmf_error_vs_scipy": error,
        "sample_statistics": stats,
    }, sort_keys=True))
