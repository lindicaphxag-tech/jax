"""Review-only Poisson rejection benchmark on the exact checked-out JAX source.

The old body is reconstructed from the real current _poisson_rejection method
by replacing only the acceptance log-PMF and restoring the precomputed
log(lambda). The proposed function and public Poisson API run unmodified.
This script is CI-only and is not part of the upstream PR diff.
"""
import inspect
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from jax import random
from jax._src.random import core


def previous_sampler():
  source = inspect.getsource(core._poisson_rejection)
  assert "t = _poisson_log_pmf(k, lam)" in source
  assert "  b = 0.931 + 2.53 * lax.sqrt(lam)" in source
  source = source.replace(
      "def _poisson_rejection(", "def _poisson_rejection_before_review(")
  source = source.replace(
      "  b = 0.931 + 2.53 * lax.sqrt(lam)",
      "  log_lam = lax.log(lam)\n  b = 0.931 + 2.53 * lax.sqrt(lam)")
  source = source.replace(
      "t = _poisson_log_pmf(k, lam)",
      "t = -lam + k * log_lam - lax_special.lgamma(k + 1)")
  env = vars(core).copy()
  exec(compile(source, "<old-jax-poisson>", "exec"), env)
  return env["_poisson_rejection_before_review"]


def median_ms(fn, key, lam, size, repeats=21):
  # All timing excludes JIT compilation and blocks pending device work.
  fn(key, lam, (size,), jnp.int32, 100).block_until_ready()
  elapsed = []
  for _ in range(repeats):
    begin = time.perf_counter()
    fn(key, lam, (size,), jnp.int32, 100).block_until_ready()
    elapsed.append((time.perf_counter() - begin) * 1e3)
  return statistics.median(elapsed)


def main():
  print("JAX", jax.__version__, "devices", jax.devices())
  print("proposed_source", core.__file__)
  assert "/site-packages/" not in core.__file__, "Benchmark must import repository source"
  previous = previous_sampler()
  cases = [(1024, 1e4), (1024, 1e8), (8192, 1e8)]
  for size, rate in cases:
    key = random.key(8675309)
    rates = jnp.full((size,), rate, dtype=jnp.float32)
    old = previous(key, rates, (size,), jnp.int32, 100)
    new = core._poisson_rejection(key, rates, (size,), jnp.int32, 100)
    old_np = np.asarray(old, dtype=np.float64)
    new_np = np.asarray(new, dtype=np.float64)
    assert (new_np >= 0).all(), "Proposed sampler returned unfinished draws"
    old_time = median_ms(previous, key, rates, size)
    new_time = median_ms(core._poisson_rejection, key, rates, size)
    print("case", size, rate, "baseline_ms", round(old_time, 4),
          "patched_ms", round(new_time, 4),
          "time_ratio", round(new_time / old_time, 3),
          "baseline_std", round(old_np.std(), 2),
          "patched_std", round(new_np.std(), 2),
          "target_std", round(np.sqrt(rate), 2))
    if rate == 1e8 and size >= 8192:
      # Deliberately generous guard for a deterministic test seed.
      assert abs(new_np.std() / np.sqrt(rate) - 1) < 0.12
  mu = jnp.float32(1e8)
  count = mu + jnp.asarray([-1e4, 0.0, 1e4], dtype=jnp.float32)
  public = np.asarray(jax.scipy.stats.poisson.logpmf(count, mu), dtype=np.float64)
  expected = scipy.stats.poisson.logpmf(
      np.asarray(count, dtype=np.float64), 1e8)
  np.testing.assert_allclose(public, expected, rtol=0.0, atol=2e-3)
  print("PASS source-matched sampler benchmark and public scipy Poisson oracle")


if __name__ == "__main__":
  main()
