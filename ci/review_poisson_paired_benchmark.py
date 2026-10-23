"""Interleaved, paired sampling benchmark for JAX PR #41228.

Uses production Poisson rejection and reconstructs the exact preceding
acceptance expression from the same checked-out function. Runs both
kernels on identical arrays and PRNG keys with separate JIT warmup.
AB/BA order is alternated and blockwise log-ratios are reported with
deterministic bootstrap confidence intervals. A single shared-host
CI run remains descriptive, not a benchmark across CPU/GPU/TPU hardware.
"""
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax._src.random import core
from review_poisson_sampling_benchmark import previous_sampler


def sync_seconds(fn, key, rates, size):
  start = time.perf_counter()
  value = fn(key, rates, (size,), jnp.int32, 100)
  value.block_until_ready()
  return time.perf_counter() - start


def bootstrap_ci(log_ratios, seed=2718):
  rng = np.random.default_rng(seed)
  log_ratios = np.asarray(log_ratios)
  draws = rng.integers(0, len(log_ratios), size=(4096, len(log_ratios)))
  boot = np.exp(log_ratios[draws].mean(axis=1))
  return np.quantile(boot, [0.025, 0.975])


def benchmark(size, rate, old, new, blocks=16, reps=8):
  key = random.key(20261023)
  rates = jnp.full((size,), rate, dtype=jnp.float32)
  # Compile and drain both paths before any timed block.
  old(key, rates, (size,), jnp.int32, 100).block_until_ready()
  new(key, rates, (size,), jnp.int32, 100).block_until_ready()
  ratios = []
  raw_old = []
  raw_new = []
  for block in range(blocks):
    per_old = []
    per_new = []
    for rep in range(reps):
      # ABBA and BAAB alternating pair order, preventing monotonic
      # warmup/clock drift from favoring the same implementation.
      order = ("old", "new") if (block + rep) % 2 == 0 else ("new", "old")
      for kind in order:
        timing = sync_seconds(old if kind == "old" else new, key, rates, size)
        (per_old if kind == "old" else per_new).append(timing)
    med_old = statistics.median(per_old)
    med_new = statistics.median(per_new)
    ratios.append(np.log(med_new / med_old))
    raw_old.append(med_old)
    raw_new.append(med_new)
  interval = bootstrap_ci(ratios)
  geo_ratio = np.exp(np.mean(ratios))
  print(
      "PAIRED", "n", size, "lambda", rate,
      "blocks",blocks,"reps",reps,
      "old_median_ms",round(statistics.median(raw_old)*1000,4),
      "patched_median_ms",round(statistics.median(raw_new)*1000,4),
      "paired_geom_ratio",round(float(geo_ratio),4),
      "block_bootstrap_CI95",
      tuple(round(float(x),4) for x in interval),
      "range_ratio",
      tuple(round(float(x),4) for x in np.exp([min(ratios),max(ratios)])),
  )
  assert np.isfinite(geo_ratio)
  return geo_ratio


def main():
  print("JAX",jax.__version__,"runtime",jax.devices(),"source",core.__file__)
  assert "/site-packages/" not in core.__file__
  old = previous_sampler()
  # Each case uses same key across old/patched, so differences are from
  # acceptance math rather than random inputs.
  for case in [(1024, 1e4),(1024,1e8),(8192,1e8),(32768,1e8)]:
    benchmark(*case, old, core._poisson_rejection)
  print("PASS paired balanced-order JAX rejection-sampler benchmark")


if __name__ == "__main__":
  main()
