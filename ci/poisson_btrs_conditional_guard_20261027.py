"""Exact-checkout CPU gate for the fork-only guarded Poisson BTRS fast path.

Run from this branch after installing JAX editable:
    python ci/poisson_btrs_conditional_guard_20261027.py

Compiles the *actual* current source sampler and a comparator obtained by
replacing only the new acceptance log-PMF call with its previous implementation.
A passing run is not a proof of numerical correctness on every device or rate.
"""
from __future__ import annotations

import inspect
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")
from jax._src.random import core

sampler_source = inspect.getsource(core._poisson_rejection)
new_call = "t = _poisson_rejection_log_pmf(k, lam)"
old_call = "t = _poisson_log_pmf(k, lam)"
assert sampler_source.count(new_call) == 1, "Unexpected source head"
reference_source = sampler_source.replace(new_call, old_call)
reference_namespace = vars(core).copy()
exec(compile(reference_source, "<old-full-log-pmf-BTRS>", "exec"),
     reference_namespace)
reference_sampler = reference_namespace["_poisson_rejection"]
candidate_sampler = core._poisson_rejection


def sample(fn, rate, shape, seed):
  key = jax.random.key(seed)
  return np.asarray(fn(key, rate, shape, np.dtype("int32"), 100))


def check_log_pmf_contract():
  for dtype in (jnp.float32, jnp.float64):
    with jax.enable_x64(True):
      k = jnp.asarray(
          [0, 1, 2, 7, 11, 12, 20, 21, 100, 1000, 1e5, 1e8],
          dtype=dtype,
      )
      for mean in (0.0, 1e-8, 10.0, 20.0, 1000.0, 3e6, 1e8):
        lam = jnp.asarray(mean, dtype=dtype)
        for fn in (core._poisson_rejection_log_pmf,
                   jax.jit(core._poisson_rejection_log_pmf)):
          tested = np.asarray(fn(k, lam))
          expected = np.asarray(core._poisson_log_pmf(k, lam))
          np.testing.assert_allclose(
              tested, expected, rtol=1e-6, atol=5e-5,
              equal_nan=True,
          )
  print("PASS: public full-formula parity including small/zero and large k")


def check_source_sampler_parity():
  tested = 0
  for mean in (10., 20., 1000., 1e6, 3e6, 1e8):
    for seed in (0, 1, 7, 37, 53, 101, 211):
      rate = jnp.asarray(mean, jnp.float32)
      original = sample(reference_sampler, rate, (4096,), seed)
      trial = sample(candidate_sampler, rate, (4096,), seed)
      np.testing.assert_array_equal(original, trial)
      tested += len(trial)
  for means in (
      [20., 1000., 3e6, 1e8],
      [10., 11., 12., 13.],
      [1e8] * 4,
  ):
    rates = jnp.repeat(jnp.asarray(means, jnp.float32), 2048)
    for seed in (0, 1, 7, 37):
      original = sample(reference_sampler, rates, (8192,), seed)
      trial = sample(candidate_sampler, rates, (8192,), seed)
      np.testing.assert_array_equal(original, trial)
      tested += len(trial)
  print("PASS: same-key source-head CPU sampler parity", tested, "draws")


def paired_benchmark():
  measurements = []
  for mean in (20., 1000., 3e6, 1e8):
    for n in (1024, 8192, 32768):
      rate = jnp.asarray(mean, jnp.float32)
      args = (jax.random.key(191), rate, (n,), np.dtype("int32"), 100)
      for fn in (reference_sampler, candidate_sampler):
        fn(*args).block_until_ready()
      pairs = []
      for i in range(24):
        sequence = (
            (("old", reference_sampler), ("guarded", candidate_sampler))
            if i % 2 == 0 else
            (("guarded", candidate_sampler), ("old", reference_sampler))
        )
        times = {}
        for name, fn in sequence:
          start = time.perf_counter_ns()
          fn(*args).block_until_ready()
          times[name] = (time.perf_counter_ns() - start) * 1e-6
        pairs.append(times)
      ratios = [p["guarded"] / p["old"] for p in pairs]
      row = {
          "mean": mean, "n": n, "paired_ratio": statistics.median(ratios),
          "reference_ms": statistics.median(p["old"] for p in pairs),
          "guarded_ms": statistics.median(p["guarded"] for p in pairs),
          "n_pairs": len(pairs),
      }
      measurements.append(row)
      print("PAIR", json.dumps(row, sort_keys=True), flush=True)
  with open("poisson_btrs_guarded_cpu.json", "w", encoding="utf-8") as f:
    json.dump({
        "jax": jax.__version__, "device": str(jax.devices()[0]),
        "source": "current checked-out candidate, exact source-transplant reference",
        "results": measurements,
    }, f, indent=2)


if __name__ == "__main__":
  print("JAX", jax.__version__, "from", core.__file__, flush=True)
  check_log_pmf_contract()
  check_source_sampler_parity()
  paired_benchmark()
