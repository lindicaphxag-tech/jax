"""Benchmark the exact JAX #41228 source against old/stable/conditional BTRS.

Run from the checked-out PR source tree with CPU jaxlib available:
  python ci/poisson_sampler_fastpath_benchmark_20261027.py

The old and stable variants are created by changing only the acceptance-PMF
call in the actual checked-out _poisson_rejection source. The fast-path
variant is the actual source function. This is a diagnostic, not a CI
performance threshold: timings vary with CPU and runner load.
"""
import inspect
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.random import core as random_core


def _build_variant(source_text, replace_with, *, restore_log_lam=False):
    old_call = "t = _poisson_log_pmf_for_rejection(k, lam)"
    assert old_call in source_text, (
        "Expected fast-path call is missing; this benchmark must fail closed "
        "rather than silently benchmark a different source revision."
    )
    source_text = source_text.replace(old_call, replace_with, 1)
    if restore_log_lam:
        marker = "  b = 0.931 + 2.53 * lax.sqrt(lam)\n"
        assert marker in source_text and "log_lam = lax.log(lam)" not in source_text
        source_text = source_text.replace(
            marker, "  log_lam = lax.log(lam)\n" + marker, 1
        )
    namespace = dict(vars(random_core))
    exec(compile(source_text, "<exact-head-poisson-btrs-variant>", "exec"), namespace)
    return namespace["_poisson_rejection"]


def main():
    actual_source = inspect.getsource(random_core._poisson_rejection)
    old = _build_variant(
        actual_source,
        "t = -lam + k * log_lam - lax_special.lgamma(k + 1)",
        restore_log_lam=True,
    )
    stable = _build_variant(actual_source, "t = _poisson_log_pmf(k, lam)")
    fastpath = random_core._poisson_rejection

    variants = {"old": old, "stable": stable, "fastpath": fastpath}
    results = []
    for rate in (1e6, 3e6, 1e8):
        for size in (1024, 8192, 32768):
            lam = jnp.asarray(rate, dtype=jnp.float32)
            same_key = {}
            for name, fn in variants.items():
                output = fn(
                    jax.random.key(2026), lam, (size,), np.dtype("int32"), 100
                )
                output.block_until_ready()
                same_key[name] = np.asarray(output)
            np.testing.assert_array_equal(same_key["stable"], same_key["fastpath"])

            timings = {name: [] for name in variants}
            orders = (
                ("old", "stable", "fastpath"),
                ("stable", "fastpath", "old"),
                ("fastpath", "old", "stable"),
            )
            for iteration in range(24):
                key = jax.random.fold_in(jax.random.key(8888), iteration)
                for name in orders[iteration % len(orders)]:
                    start = time.perf_counter_ns()
                    output = variants[name](
                        key, lam, (size,), np.dtype("int32"), 100
                    )
                    output.block_until_ready()
                    timings[name].append((time.perf_counter_ns() - start) / 1e6)

            medians = {
                name: statistics.median(values)
                for name, values in timings.items()
            }
            results.append({
                "lambda": rate,
                "n": size,
                "median_ms": medians,
                "fastpath_over_old": medians["fastpath"] / medians["old"],
                "stable_over_old": medians["stable"] / medians["old"],
                "fastpath_over_stable": medians["fastpath"] / medians["stable"],
                "variance_over_lambda": {
                    name: float(samples.var() / rate)
                    for name, samples in same_key.items()
                },
                "same_key_stable_fastpath_exact": bool(np.array_equal(
                    same_key["stable"], same_key["fastpath"]
                )),
            })
            print(json.dumps(results[-1], sort_keys=True), flush=True)

    held_out = 0
    for rate in (10., 20., 1000., 1e6, 3e6, 1e8):
        for seed in (0, 1, 7, 37, 53, 113, 211):
            lam = jnp.asarray(rate, dtype=jnp.float32)
            key = jax.random.key(seed)
            old_output = np.asarray(
                stable(key, lam, (4096,), np.dtype("int32"), 100)
            )
            new_output = np.asarray(
                fastpath(key, lam, (4096,), np.dtype("int32"), 100)
            )
            np.testing.assert_array_equal(old_output, new_output)
            held_out += len(old_output)

    full_helper = jax.jit(random_core._poisson_log_pmf)
    conditional_helper = jax.jit(random_core._poisson_log_pmf_for_rejection)
    tail_results = []
    for fraction in (1 / 32768, 0.05):
        size = 32768
        lam = jnp.asarray(1e8, dtype=jnp.float32)
        counts = np.full(size, 1e8, dtype=np.float32)
        tail_count = max(1, int(size * fraction))
        counts[:tail_count] = 9e7
        counts = jnp.asarray(counts)
        expected = np.asarray(full_helper(counts, lam))
        actual = np.asarray(conditional_helper(counts, lam))
        np.testing.assert_array_equal(expected, actual)

        times = {}
        for name, fn in (("stable", full_helper), ("fastpath", conditional_helper)):
            fn(counts, lam).block_until_ready()
            measurements = []
            for _ in range(24):
                start = time.perf_counter_ns()
                fn(counts, lam).block_until_ready()
                measurements.append((time.perf_counter_ns() - start) / 1e6)
            times[name] = statistics.median(measurements)
        tail_results.append({
            "tail_fraction": fraction,
            "tail_count": tail_count,
            "n": size,
            "median_ms": times,
            "fastpath_over_stable": times["fastpath"] / times["stable"],
            "exact_fallback_match": True,
        })

    output = {
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "source": random_core.__file__,
        "protocol": "24 interleaved warmed measurements per cell; JIT compilation excluded",
        "scope": "Run on exact checked-out source; no performance threshold is asserted",
        "held_out_same_key_samples": held_out,
        "results": results,
        "mixed_tail_results": tail_results,
    }
    path = Path("poisson_sampler_fastpath_benchmark_20261027.json")
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("PASS; wrote", path, flush=True)


if __name__ == "__main__":
    main()
