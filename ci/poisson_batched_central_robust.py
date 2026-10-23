"""Independent phase-two CPU repeatability and branch-boundary evaluation.

Uses exact JAX implementation and source-matched control from
poisson_batched_central_ab.py; all benchmarks are interleaved JIT timings,
not claims about GPU and full RL training throughput.
"""
import json
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax._src.random import core
from poisson_batched_central_ab import old_stable_sampler

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)


def compare_score_branches():
  cases = {
      "central_exact": (
          np.array([1e8, 1e8 + 1000, 1e8 - 1000, 1e8+1e5], np.float32),
          np.full(4, 1e8, np.float32)),
      "mixed_one_far": (
          np.array([1e8, 1e8 + 3e6, 1e8 - 1e3, 1e8], np.float32),
          np.full(4, 1e8, np.float32)),
      "mixed_rate": (
          np.array([15, 1e8, 500, 1e6], np.float32),
          np.array([10, 1e8, 1e3, 1e6], np.float32)),
      "negative_proposal": (
          np.array([-1, 1e8, 0, 1e8], np.float32),
          np.full(4, 1e8, np.float32)),
  }
  for title, (raw_k, raw_lam) in cases.items():
    certificate = bool(np.all(
        (raw_lam >= 1e6) & (raw_k > 10) &
        (np.abs((raw_k - raw_lam) / np.maximum(raw_k, 1)) <= 0.01)))
    expected_certificate = (title == "central_exact")
    assert certificate == expected_certificate, (title, certificate)
    k = jnp.asarray(raw_k)
    lam = jnp.asarray(raw_lam)
    expected = np.asarray(core._poisson_log_pmf(k, lam))
    observed = np.asarray(core._poisson_rejection_log_pmf(k, lam))
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-5,
                               equal_nan=True)
    print("BRANCH_CASE", json.dumps({"case": title, "ok": True,
          "max_finite_abs_diff": float(np.max(
              np.abs(observed[np.isfinite(observed)] -
                     expected[np.isfinite(expected)]))) if np.isfinite(expected).all() else None}),flush=True)


def repeated_timing(lam, n, repeats=21, blocks=5):
  old = old_stable_sampler()
  new = core._poisson_rejection
  shape = (n,)
  dtype = np.dtype("int32")
  iters = int(np.iinfo(np.int32).max)
  rates = jnp.full(shape, lam, dtype=jnp.float32)
  ratios = []
  raw = []
  for block in range(blocks):
    key = jax.random.key(10000 + 997*block)
    args = (key, rates, shape, dtype, iters)
    for fn in (old, new):
      fn(*args).block_until_ready()
    records = [[], []]
    for i in range(repeats):
      order = (0, 1) if (i + block) % 2 == 0 else (1, 0)
      for idx in order:
        start = time.perf_counter()
        (old, new)[idx](*args).block_until_ready()
        records[idx].append(1000*(time.perf_counter()-start))
    before = statistics.median(records[0])
    after = statistics.median(records[1])
    ratio = after/before
    ratios.append(ratio)
    raw.append({"block": block,"old_ms":before,"new_ms":after,"ratio":ratio})
  result={"lambda":lam,"n":n,"blocks":blocks,"repeats_per_block":repeats,
          "median_ratio":statistics.median(ratios),
          "min_ratio":min(ratios),"max_ratio":max(ratios),
          "block_results":raw}
  print("REPEATED_TIMING",json.dumps(result,sort_keys=True),flush=True)


def mixed_rate_sampling():
  old=old_stable_sampler()
  new=core._poisson_rejection
  rates=jnp.asarray(np.tile([10,1000,1e6,1e8],2048),dtype=jnp.float32)
  args=((8192,),np.dtype("int32"),int(np.iinfo(np.int32).max))
  for seed in [31,73,101,199,313,997]:
    key=jax.random.key(seed)
    a=np.asarray(old(key,rates,*args))
    b=np.asarray(new(key,rates,*args))
    mismatches=int(np.count_nonzero(a!=b))
    print("MIXED_RATE",json.dumps({"seed":seed,"mismatches":mismatches}),flush=True)
    assert mismatches==0


if __name__=="__main__":
  print("JAX_ENV",jax.__version__,jax.lib.__version__,jax.devices())
  compare_score_branches()
  mixed_rate_sampling()
  for lam in (10.,1000.,1e6,1e8):
    repeated_timing(lam,8192)
  print("PHASE_TWO_COMPLETE",flush=True)
