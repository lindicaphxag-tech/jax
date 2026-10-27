"""Adversarial batch-tail benchmark for guarded JAX Poisson BTRS log-PMF.

Execute on the fork branch with editable current JAX:
  python ci/poisson_guard_adversarial_cpu_20261027.py

The candidate is a fork-only optimization. Benchmarks are descriptive CPU
measurements, not a CI requirement that timings remain below a hard threshold.
"""
import json
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax._src.random import core

jax.config.update("jax_platform_name", "cpu")

FULL = jax.jit(core._poisson_log_pmf)
GUARDED = jax.jit(core._poisson_rejection_log_pmf)
assert FULL is not GUARDED

cases = []
rng = np.random.default_rng(742)
for n in (1024, 8192, 32768):
    for scenario in ("all_central", "one_outlier", "five_pct_outlier", "half_outlier"):
        rate = jnp.asarray(1e8, np.float32)
        k = np.float32(1e8) + rng.integers(-12000, 12000, size=n).astype(np.float32)
        if scenario == "one_outlier":
            k[0] = 100.
        if scenario == "five_pct_outlier":
            k[:max(1, n // 20)] = 100.
        if scenario == "half_outlier":
            k[:n // 2] = 100.
        k = jnp.asarray(k)
        old_values = np.asarray(FULL(k, rate))
        new_values = np.asarray(GUARDED(k, rate))
        np.testing.assert_allclose(
            new_values, old_values, rtol=0, atol=4e-5, equal_nan=True
        )
        if scenario != "all_central":
            # With one far-tail item the all-candidate guard is false,
            # and the implementation MUST run the unmodified full formula.
            np.testing.assert_array_equal(new_values, old_values)
        for fn in (FULL, GUARDED):
            fn(k, rate).block_until_ready()
        paired = []
        for i in range(24):
            sequence = (
                (("full", FULL), ("guarded", GUARDED)) if i % 2 == 0
                else (("guarded", GUARDED), ("full", FULL))
            )
            times = {}
            for label, fn in sequence:
                start = time.perf_counter_ns()
                fn(k, rate).block_until_ready()
                times[label] = (time.perf_counter_ns() - start) / 1e6
            paired.append(times)
        record = dict(
            count=n, scenario=scenario,
            paired_guard_full_ratio=statistics.median(
                x["guarded"] / x["full"] for x in paired
            ),
            full_median_ms=statistics.median(x["full"] for x in paired),
            guarded_median_ms=statistics.median(x["guarded"] for x in paired),
            max_abs_logpmf_difference=float(
                np.nanmax(np.abs(new_values - old_values))
            ),
            sample_equality=bool(np.array_equal(new_values, old_values)),
            runs=24,
        )
        cases.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
with open("poisson_guard_adversarial_cpu_results.json", "w") as f:
    json.dump({
        "jax_version": jax.__version__,
        "backend": str(jax.devices()[0]),
        "note": "Fork-only CPU exact-source benchmark; not official merge CI",
        "cases": cases,
    }, f, indent=2)
