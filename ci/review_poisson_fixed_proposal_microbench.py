"""Isolate JAX Poisson log-PMF arithmetic from rejection-loop dynamics.

Old direct formula and the proposed stable formula evaluate the identical,
frozen (k, lambda) arrays. This removes the confound in full rejection
sampling where changing accept/reject outcomes changes iteration counts.
Descriptive CPU CI, not a GPU or cross-hardware claim.
"""
import math
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.random import core


@jax.jit
def old_log_pmf(k, lam):
  return -lam + k * jax.lax.log(lam) - jax.lax.lgamma(k + 1)


@jax.jit
def stable_log_pmf(k, lam):
  return core._poisson_log_pmf(k, lam)


def timed(fn, k, rate):
  start = time.perf_counter()
  fn(k, rate).block_until_ready()
  return (time.perf_counter() - start) * 1e3


def case(size, lam, blocks=12, repeat=8):
  # Nonrandom fixed symmetric proposals around the Poisson mode.
  # k stays nonnegative and comparable across implementations.
  offsets = np.linspace(-2, 2, size, dtype=np.float32)
  rate = jnp.full((size,), lam, dtype=jnp.float32)
  k = jnp.floor(rate + jnp.asarray(offsets) * math.sqrt(lam))
  old = np.asarray(old_log_pmf(k, rate))
  stable = np.asarray(stable_log_pmf(k, rate))
  # At high rates, direct float32 cancellations must actually be detectable.
  if lam == 1e8:
    print("FIXED_INPUT",size,lam,
          "old_range",tuple(float(x) for x in (old.min(),old.max())),
          "stable_range",tuple(float(x) for x in (stable.min(),stable.max())),
          "abs_mean_disagreement",float(np.mean(np.abs(old-stable))))
    assert np.isfinite(stable).all()
    assert float(np.mean(np.abs(old-stable))) > 1.0
  log_ratios = []
  for b in range(blocks):
    a,btime = [],[]
    for rep in range(repeat):
      order = (old_log_pmf, stable_log_pmf) if (b+rep)%2 == 0 else (stable_log_pmf, old_log_pmf)
      for fn in order:
        elapsed=timed(fn,k,rate)
        (a if fn is old_log_pmf else btime).append(elapsed)
    log_ratios.append(math.log(statistics.median(btime)/statistics.median(a)))
  ratio=math.exp(statistics.mean(log_ratios))
  print("ARITHMETIC",size,lam,
        "patched/old_geom_ratio",round(ratio,4),
        "per_block_min_max",tuple(round(math.exp(x),4) for x in (min(log_ratios),max(log_ratios))))
  assert np.isfinite(ratio)


def main():
  print("JAX",jax.__version__,"devices",jax.devices(),"source",core.__file__)
  assert "/site-packages/" not in core.__file__, "JAX source must be checked out"
  for n,lam in [(1024,1e4),(1024,1e8),(8192,1e8),(32768,1e8)]:
    case(n,lam)
  print("PASS: fixed-proposal arithmetic isolated from dynamic rejection loops")


if __name__=="__main__":
  main()
