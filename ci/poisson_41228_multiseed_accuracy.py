"""Independent-seed distribution evidence for JAX #41228 on matched source.

Use fork PR source's actual _poisson_rejection against an exact pre-patch
method-body reconstruction in poisson_41228_paired_perf.py.
This is distribution evidence, not an end-to-end training benchmark.
"""
import json
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np

from poisson_41228_paired_perf import old_rejection
from jax._src.random import core as random_core


def run(*, rate=1e8, size=32768, seeds=16):
    rate_arr = jnp.asarray(rate, dtype=jnp.float32)
    generator = jax.random.key(41228)
    funcs = {"original": old_rejection,
             "corrected": random_core._poisson_rejection}
    values = {name: [] for name in funcs}
    details = []
    # Both functions compile and warm before statistical data are recorded.
    for fn in funcs.values():
        fn(generator, rate_arr, (size,), np.dtype("int32"), 100).block_until_ready()
    for i in range(seeds):
        key = jax.random.fold_in(generator, i + 100)
        row = {"seed": i}
        for name, fn in funcs.items():
            sample = np.asarray(
                fn(key, rate_arr, (size,), np.dtype("int32"), 100),
                dtype=np.float64)
            variance = float(sample.var(ddof=1))
            relative = variance / rate - 1.0
            values[name].append(relative)
            row[name] = {
                "mean": float(sample.mean()),
                "sample_variance": variance,
                "signed_relative_variance_error": relative,
            }
        details.append(row)
    summary = {}
    for name, sample in values.items():
        a = np.asarray(sample)
        mean = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(seeds))
        summary[name] = {
            "mean_signed_relative_variance_error": mean,
            "se_across_seeds": se,
            "approx_95pct_ci": [mean - 1.96 * se, mean + 1.96 * se],
        }
    result = {
        "source": "current fork PR #41228 head under test",
        "jax_version": jax.__version__,
        "backend": str(jax.devices()[0]),
        "rate": rate, "sample_size_per_seed": size, "independent_seeds": seeds,
        "summary": summary, "records": details,
    }
    print("MULTISEED_JSON", json.dumps(result), flush=True)
    old_mean = summary["original"]["mean_signed_relative_variance_error"]
    new_mean = summary["corrected"]["mean_signed_relative_variance_error"]
    # This checks a very large regression (not statistical significance).
    if not (old_mean > 0.20 and abs(new_mean) < 0.08):
        raise AssertionError(
            f"large-lambda variance evidence unexpectedly changed: {old_mean=}, {new_mean=}")
    return result


if __name__ == "__main__":
    run()
