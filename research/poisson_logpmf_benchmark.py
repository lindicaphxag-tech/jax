"""CPU/GPU microbenchmark for JAX Poisson log-PMF review (#41228).

Run from a JAX source checkout whose PYTHONPATH resolves to that checkout.
This benchmark is not a substitute for random_lax_test/scipy_stats_test or
end-to-end Poisson sampler distribution checks. To compare PR versus upstream,
run the same script on both trees, adapting the candidate import on base.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import poisson

try:
  from jax._src.random.core import _poisson_log_pmf
except ImportError as error:
  raise SystemExit(
      "This script requires the #41228 source tree on PYTHONPATH."
  ) from error


def _legacy_logpmf(k, rate):
  return k * jnp.log(rate) - jax.scipy.special.gammaln(k + 1) - rate


def _time(fn, k, rate, repetitions):
  compiled = jax.jit(fn)
  compiled(k, rate).block_until_ready()  # Exclude compilation and first-run.
  timings = []
  for _ in range(repetitions):
    start = time.perf_counter()
    compiled(k, rate).block_until_ready()
    timings.append((time.perf_counter() - start) * 1_000)
  return statistics.median(timings), compiled


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", type=int, default=8192)
  parser.add_argument("--repeats", type=int, default=20)
  parser.add_argument("--seed", type=int, default=1729)
  args = parser.parse_args()
  rng = np.random.default_rng(args.seed)
  report = {"jax": jax.__version__, "platform": jax.default_backend(),
            "size": args.size, "repeats": args.repeats, "cases": []}

  # dtype is explicit: float32 cannot resolve all integer offsets at 1e9.
  for dtype in (jnp.float32, jnp.float64):
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
      continue
    for mu, tail_ratio in ((1e7, 1.0), (1e9, 1.0), (1e9, 0.3)):
      counts = np.round(mu + rng.normal(size=args.size) * mu ** 0.5)
      k = jnp.asarray(counts, dtype=dtype)
      rate = jnp.full(k.shape, mu * tail_ratio, dtype=dtype)
      ref = poisson.logpmf(np.asarray(k, np.float64),
                           np.asarray(rate, np.float64))
      measurements = {}
      for name, fn in (("candidate", _poisson_log_pmf),
                       ("legacy", _legacy_logpmf)):
        median_ms, compiled = _time(fn, k, rate, args.repeats)
        values = np.asarray(compiled(k, rate), dtype=np.float64)
        measurements[name] = {
            "median_ms": median_ms,
            "max_abs_error_vs_scipy": float(np.max(np.abs(values - ref))),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
        }
      report["cases"].append({
          "dtype": str(np.dtype(dtype)),
          "rate": mu * tail_ratio,
          "count_center": mu,
          "measurements": measurements,
      })
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
