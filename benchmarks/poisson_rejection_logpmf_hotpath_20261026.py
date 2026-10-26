"""Benchmark original, PR #41228, and a research-only rate-gated acceptance PMF.

Run from an editable install of the Poisson PR branch. No changes to the
installed sampler or public poisson.logpmf are made by this script.
The *selective* arm is a hypothesis, not an approved production patch; it
uses the historical BTRS expression only if ALL lambda values are <1e6.
Verify accuracy/stochastic bias over more seeds before interpreting speed.
"""
import inspect
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.random import core as random_core

src = inspect.getsource(random_core._poisson_rejection)
needle = "t = _poisson_log_pmf(k, lam)"
assert needle in src, "Install the candidate JAX Poisson branch"
old_expr = "-lam + k * log_lam - lax_special.lgamma(k + 1)"
old_src = src.replace("  b = 0.931 + 2.53 * lax.sqrt(lam)",
                      "  log_lam = lax.log(lam)\n  b = 0.931 + 2.53 * lax.sqrt(lam)")
assert old_src != src
old_src = old_src.replace(needle, f"t = {old_expr}")
selective_src = old_src.replace(
    f"t = {old_expr}",
    "t = lax.cond(jnp.all(lam < 1e6), "
    f"lambda _: {old_expr}, "
    "lambda _: _poisson_log_pmf(k, lam), operand=None)",
)
assert selective_src != old_src
ns = dict(vars(random_core))
exec(compile(old_src, "<baseline-pmf>", "exec"), ns)
baseline = ns["_poisson_rejection"]
exec(compile(selective_src, "<selective-pmf-experiment>", "exec"), ns)
selective = ns["_poisson_rejection"]
candidate = random_core._poisson_rejection

KEY = jax.random.key(2026)
N = 20000
print("JAX", jax.__version__, "devices", jax.devices())
for rate in (20., 100., 1000., 1e5, 1e6, 1e8):
    lam = jnp.asarray(rate, dtype=jnp.float32)
    measurements = {}
    for label, fn in (("baseline", baseline), ("candidate", candidate),
                      ("selective_experimental", selective)):
        def run(i):
            return fn(jax.random.fold_in(KEY, i), lam, (N,), np.int32, 100)
        run(1).block_until_ready()
        durations = []
        for i in range(2, 9):
            before = time.perf_counter()
            sample = run(i)
            sample.block_until_ready()
            durations.append(1000 * (time.perf_counter() - before))
        samples = np.asarray(sample, dtype=np.float64)
        measurements[label] = {
            "median_ms": round(statistics.median(durations), 3),
            "mean": round(float(np.mean(samples)), 3),
            "variance_over_rate": round(float(np.var(samples) / rate), 4),
            "negative_samples": int(np.sum(samples < 0)),
        }
    print(f"lambda={rate:g} {measurements}", flush=True)
