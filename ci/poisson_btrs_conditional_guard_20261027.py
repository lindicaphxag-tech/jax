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



def check_both_conditional_branches():
  """Exercise BOTH sides of the runtime batch-wide guard, not only fallback.

  The older broad k-grid always included k=0, making jnp.all(central)
  false. A passing broad-grid test alone cannot validate this fast path.
  """
  for dtype in (jnp.float32, jnp.float64):
    with jax.enable_x64(True):
      for mean in (20.0, 1000.0, 3e6, 1e8):
        rate = jnp.asarray(mean, dtype=dtype)
        # All values lie in the polynomial/asymptotic-Stirling region.
        delta = jnp.asarray([0., 0., 0., 0., 0.], dtype=dtype)
        if mean >= 1000:
          delta = jnp.asarray([-2., -1., 0., 1., 2.], dtype=dtype)
        central_k = rate + delta
        safe_k = jnp.maximum(central_k, 1)
        central_mask = ((safe_k > 10.) & (rate > 0.)
                        & (jnp.abs((safe_k-rate)/safe_k) <= .01))
        assert bool(np.asarray(jnp.all(central_mask))), (
            "central guard unexpectedly false", dtype, mean)
        for jit in (False, True):
          candidate = (jax.jit(core._poisson_rejection_log_pmf)
                       if jit else core._poisson_rejection_log_pmf)
          actual = np.asarray(candidate(central_k, rate))
          expected = np.asarray(core._poisson_log_pmf(central_k, rate))
          np.testing.assert_allclose(actual, expected,
                                     rtol=1e-6, atol=5e-5)

        # A single out-of-support or noncentral element forces the entire
        # vector to use the full formula. Check both direct and JIT modes.
        mixed_k = central_k.at[0].set(0.)
        mixed_safe = jnp.maximum(mixed_k, 1)
        mixed_mask = ((mixed_safe > 10.) & (rate > 0.)
                      & (jnp.abs((mixed_safe-rate)/mixed_safe) <= .01))
        assert not bool(np.asarray(jnp.all(mixed_mask))), (
            "fallback guard unexpectedly true", dtype, mean)
        for jit in (False, True):
          candidate = (jax.jit(core._poisson_rejection_log_pmf)
                       if jit else core._poisson_rejection_log_pmf)
          actual = np.asarray(candidate(mixed_k, rate))
          expected = np.asarray(core._poisson_log_pmf(mixed_k, rate))
          np.testing.assert_allclose(actual, expected,
                                     rtol=1e-6, atol=5e-5)
  print("PASS: guarded CENTRAL and FULL fallback branches exercised on CPU")


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
  # Check adversarial mixtures: all-central, a single small-rate item,
  # and a batch with rates near the rejection-sampler threshold. A scalar-only
  # speedup is not evidence of real-world mixed-rate efficiency.
  mixed_scenarios = {
      "uniform_1000": [1000.],
      "uniform_1e8": [1e8],
      "mixed_central": [300., 1000., 3e6, 1e8],
      "one_near_small": [1e8] * 4095 + [10.],
      "mixed_near_boundary": [10., 11., 20., 1000., 1e8],
  }
  mixed_results = []
  for label, values in mixed_scenarios.items():
    rate = jnp.asarray(np.resize(np.asarray(values, np.float32), 4096))
    args = (jax.random.key(1027), rate, (4096,), np.dtype("int32"), 100)
    reference = np.asarray(reference_sampler(*args))
    candidate = np.asarray(candidate_sampler(*args))
    np.testing.assert_array_equal(reference, candidate)
    for fn in (reference_sampler, candidate_sampler):
      fn(*args).block_until_ready()
    pairs = []
    for i in range(12):
      sequence = (
          (("old", reference_sampler), ("guarded", candidate_sampler))
          if i % 2 == 0 else
          (("guarded", candidate_sampler), ("old", reference_sampler))
      )
      durations = {}
      for name, fn in sequence:
        start = time.perf_counter_ns()
        fn(*args).block_until_ready()
        durations[name] = (time.perf_counter_ns() - start) * 1e-6
      pairs.append(durations)
    row = {
        "scenario": label, "n": 4096,
        "paired_ratio": statistics.median(
            p["guarded"] / p["old"] for p in pairs),
        "reference_ms": statistics.median(p["old"] for p in pairs),
        "guarded_ms": statistics.median(p["guarded"] for p in pairs),
        "n_pairs": len(pairs), "exact_same_key": True,
    }
    mixed_results.append(row)
    print("MIXED", json.dumps(row, sort_keys=True), flush=True)
  with open("poisson_btrs_guarded_cpu.json", "w", encoding="utf-8") as f:
    json.dump({
        "jax": jax.__version__, "device": str(jax.devices()[0]),
        "source": "current checked-out candidate, exact source-transplant reference",
        "results": measurements,
        "mixed_scenarios": mixed_results,
    }, f, indent=2)


if __name__ == "__main__":
  print("JAX", jax.__version__, "from", core.__file__, flush=True)
  check_log_pmf_contract()
  check_both_conditional_branches()
  check_source_sampler_parity()
  paired_benchmark()
