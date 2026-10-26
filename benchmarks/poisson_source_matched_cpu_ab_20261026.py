"""CPU regression for JAX Poisson BTRS log-PMF, without changing installed JAX.

Supports legacy jax._src.random (e.g. 0.9.0.1) and a newer
jax._src.random.core tree. Replace only the acceptance log-PMF inside the
production sampler method, preserving the rejection algorithm/PRNG.
"""
import inspect
import math
import statistics
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
try:
    from jax._src.random import core as rc
    if not hasattr(rc, "_poisson_rejection"):
        raise ImportError()
except ImportError:
    from jax._src import random as rc


def candidate_pmf(k, lam):
    # Mirrors jax-ml/jax#41228 head 3894c33, NOT a simplified approximation.
    safe_k = jnp.maximum(k, 1)
    delta = safe_k - lam
    relative_delta = delta / safe_k
    small_delta = jnp.abs(relative_delta) <= .01
    t = jnp.where(small_delta, relative_delta, 0.)
    poly = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
              * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
    central = -safe_k * (t * t) * poly
    far = lam / safe_k < .5
    rt = jnp.where(far, 0., relative_delta)
    regular = safe_k * lax.log1p(-rt) + delta
    dev = jnp.where(small_delta, central, regular)
    sq = safe_k * safe_k
    asymp = (1. / 12 - (1. / 360 - 1. / 1260 / sq) / sq) / safe_k
    tail = jnp.where(safe_k <= 10., rc._stirling_approx_tail(safe_k - 1.), asymp)
    norm = .5 * (lax.log(safe_k) + math.log(2 * math.pi)) + tail
    rate = jnp.where(k == 0, 1., lam)
    ratio = rate / safe_k
    positive = ratio > 0.
    safe_ratio = jnp.where(positive, ratio, 1.)
    log_ratio = jnp.where(positive, lax.log(safe_ratio),
                          lax.log(rate) - lax.log(safe_k))
    far_pmf = safe_k * log_ratio + delta - norm
    result = jnp.where(k == 0, -lam, jnp.where(far, far_pmf, dev - norm))
    return jnp.where(lam < 0., jnp.nan, result)


body = inspect.getsource(rc._poisson_rejection)
old = "t = -lam + k * log_lam - lax_special.lgamma(k + 1)"
new = "t = _poisson_log_pmf(k, lam)"
if old in body:
    baseline_body = body
    candidate_body = body.replace(old, new).replace("  log_lam = lax.log(lam)\n", "")
elif new in body:
    candidate_body = body
    baseline_body = body.replace(new, old).replace(
        "  b = 0.931", "  log_lam = lax.log(lam)\n  b = 0.931")
else:
    raise RuntimeError("Unsupported sampler source; refusing to benchmark")
ns = dict(vars(rc), _poisson_log_pmf=candidate_pmf)
exec(compile(baseline_body, "<original sampler>", "exec"), ns)
baseline = ns["_poisson_rejection"]
exec(compile(candidate_body, "<candidate sampler>", "exec"), ns)
candidate = ns["_poisson_rejection"]

def main():
    print("runtime:", jax.__version__, rc.__name__, jax.devices(), flush=True)
    for lam_value in (20., 100., 1000., 1e5, 1e6, 1e8):
        lam = jnp.asarray(lam_value, jnp.float32)
        result = {}
        for name, fn in (("original", baseline), ("candidate", candidate)):
            def sample(seed):
                return fn(jax.random.key(seed), lam, (6000,), np.dtype("int32"), 100)
            sample(1).block_until_ready()
            timings = []
            all_samples = []
            for seed in (37, 41, 43, 47, 53):
                start = time.perf_counter()
                s = sample(seed)
                s.block_until_ready()
                timings.append((time.perf_counter() - start) * 1000)
                all_samples.append(np.asarray(s, dtype=np.float64))
            vals = np.concatenate(all_samples)
            result[name] = {
                "mean": round(float(vals.mean()), 3),
                "variance_over_lambda": round(float(vals.var() / lam_value), 5),
                "median_ms": round(statistics.median(timings), 3),
            }
        print(f"lambda={lam_value:g}", result, flush=True)


if __name__ == "__main__":
    main()
