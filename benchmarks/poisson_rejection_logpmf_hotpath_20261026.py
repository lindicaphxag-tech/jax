"""Compare current Poisson rejection against its pre-fix acceptance log-PMF.

Run from an editable installation of the Poisson candidate JAX branch.
The generated baseline uses the same installed BTRS source, replacing only
its acceptance calculation with the original cancellation-prone expression.
Requires JAX, NumPy. Results are device-specific, not portable benchmarks.
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
assert needle in src, "Install the candidate JAX branch with _poisson_log_pmf"
src_old = src.replace("  b = 0.931 + 2.53 * lax.sqrt(lam)",
                      "  log_lam = lax.log(lam)\n  b = 0.931 + 2.53 * lax.sqrt(lam)")
src_old = src_old.replace(
    needle, "t = -lam + k * log_lam - lax_special.lgamma(k + 1)")
namespace = dict(vars(random_core))
exec(compile(src_old, "<pre-fix-acceptance-PMF>", "exec"), namespace)
baseline = namespace["_poisson_rejection"]
candidate = random_core._poisson_rejection

KEY = jax.random.key(2026)
N = 20000
print("JAX", jax.__version__, "devices", jax.devices())
for rate in (20., 100., 1000., 1e5, 1e8):
    lam = jnp.asarray(rate, dtype=jnp.float32)
    measurements = {}
    for label, fn in (("baseline", baseline), ("candidate", candidate)):
        def run(i):
            return fn(jax.random.fold_in(KEY, i), lam, (N,), np.int32, 100)
        run(1).block_until_ready()  # compilation excluded
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
