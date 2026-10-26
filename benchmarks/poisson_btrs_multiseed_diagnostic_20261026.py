"""CPU diagnostic for JAX Poisson PR #41228 (fork research only).

Run from an editable checkout of the exact candidate PR head:
  python benchmarks/poisson_btrs_multiseed_diagnostic_20261026.py

This source-derives the pre-fix and experimental rate-gated samplers from
the *same* installed candidate's BTRS production method. No official PR
files or public distribution semantics are changed. The 1e6 crossover is
a research hypothesis, not an endorsed accuracy threshold.
"""

import inspect
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.random import core

src = inspect.getsource(core._poisson_rejection)
stable = "t = _poisson_log_pmf(k, lam)"
assert stable in src, "Install PR #41228 fork checkout before running"
baseline = "-lam + k * log_lam - lax_special.lgamma(k + 1)"

# Build from the exact installed method body and dependencies, so all other
# sampler control flow, PRNG, and jit configuration remain unchanged.
older = src.replace(
    "  b = 0.931 + 2.53 * lax.sqrt(lam)",
    "  log_lam = lax.log(lam)\n  b = 0.931 + 2.53 * lax.sqrt(lam)",
).replace(stable, "t = " + baseline)
g = dict(vars(core))
exec(compile(older, "<pre-fix-BTRS>", "exec"), g)
old_sampler = g["_poisson_rejection"]

# Evaluate the potential cost of a rate-based decision inside the loop.
# A general array of rates is sent to the stable branch if any rate >=1e6.
g = dict(vars(core))
selective = older.replace(
    "t = " + baseline,
    "t = lax.cond(jnp.all(lam < 1e6), "
    "lambda _: " + baseline + ", "
    "lambda _: _poisson_log_pmf(k, lam), operand=None)",
)
exec(compile(selective, "<experimental-BTRS>", "exec"), g)
research_sampler = g["_poisson_rejection"]

print("JAX", jax.__version__, "devices", jax.devices())
n = 20000
seeds = [37, 41, 43, 47, 53]
for rate in (20., 1e5, 1e6, 1e8):
    lam = jnp.asarray(rate, jnp.float32)
    result = {}
    for name, sampler in (
        ("baseline", old_sampler),
        ("candidate", core._poisson_rejection),
        ("experimental_gated", research_sampler),
    ):
        def sample(seed):
            return sampler(jax.random.key(seed), lam, (n,), np.int32, 100)
        sample(100).block_until_ready()  # exclude compilation
        durations = []
        samples = []
        for seed in seeds:
            start = time.perf_counter()
            out = sample(seed)
            out.block_until_ready()
            durations.append((time.perf_counter() - start) * 1000)
            samples.append(np.asarray(out, dtype=np.float64))
        values = np.concatenate(samples)
        result[name] = {
            "median_ms": round(statistics.median(durations), 3),
            "mean_over_rate": round(float(np.mean(values) / rate), 7),
            "variance_over_rate": round(float(np.var(values) / rate), 5),
            "negative_count": int(np.sum(values < 0)),
        }
    print(f"rate={rate:g}", result, flush=True)
