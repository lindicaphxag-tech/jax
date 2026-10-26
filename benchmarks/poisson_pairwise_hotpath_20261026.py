"""JAX Poisson source-matched paired CPU benchmark, independent fork research.

Run from a checkout with JAX installed. Reconstruct historical BTRS and
replace only the acceptance log PMF, retaining key, proposal and output.
Not the exact JAX #41228 runtime unless executed on that checkout.
"""
from __future__ import annotations

import inspect
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

try:
    from jax._src.random import core as rc
    if not hasattr(rc, "_poisson_rejection"):
        raise ImportError
except ImportError:
    from jax._src import random as rc


def stable_pmf(k, lam):
    # Mirrors jax-ml/jax PR #41228 at 280b8edf; shared log-PMF arithmetic.
    safe_k = jnp.maximum(k, 1)
    delta = safe_k - lam
    relative = delta / safe_k
    small = jnp.abs(relative) <= 0.01
    t = jnp.where(small, relative, 0.0)
    poly = ((((((t / 8 + 1 / 7) * t + 1 / 6) * t + 1 / 5)
              * t + 1 / 4) * t + 1 / 3) * t + 1 / 2)
    dev_central = -safe_k * (t * t) * poly
    far = lam / safe_k < 0.5
    regular = safe_k * lax.log1p(-jnp.where(far, 0.0, relative)) + delta
    dev = jnp.where(small, dev_central, regular)
    sq = safe_k * safe_k
    asym = (1 / 12 - (1 / 360 - 1 / 1260 / sq) / sq) / safe_k
    tail = jnp.where(safe_k <= 10,
                     rc._stirling_approx_tail(safe_k - 1), asym)
    norm = 0.5 * (lax.log(safe_k) + math.log(2 * math.pi)) + tail
    rate = jnp.where(k == 0, 1.0, lam)
    ratio = rate / safe_k
    positive = ratio > 0
    log_ratio = jnp.where(
        positive, lax.log(jnp.where(positive, ratio, 1)),
        lax.log(rate) - lax.log(safe_k))
    far_log_pmf = safe_k * log_ratio + delta - norm
    out = jnp.where(k == 0, -lam,
                    jnp.where(far, far_log_pmf, dev - norm))
    return jnp.where(lam < 0, jnp.nan, out)


source = inspect.getsource(rc._poisson_rejection)
historical = "t = -lam + k * log_lam - lax_special.lgamma(k + 1)"
candidate = "t = _poisson_log_pmf(k, lam)"
if historical in source:
    baseline_source = source
    stable_source = source.replace(historical, candidate).replace(
        "  log_lam = lax.log(lam)\n", "")
elif candidate in source:
    stable_source = source
    baseline_source = source.replace(candidate, historical).replace(
        "  b = 0.931", "  log_lam = lax.log(lam)\n  b = 0.931")
else:
    raise RuntimeError("Unknown sampler source: refusing misleading A/B")

ns = dict(vars(rc), _poisson_log_pmf=stable_pmf)
exec(compile(baseline_source, "<historical-btrs>", "exec"), ns)
baseline = ns["_poisson_rejection"]
exec(compile(stable_source, "<candidate-btrs>", "exec"), ns)
patched = ns["_poisson_rejection"]

results = []
for rate in (20.0, 1000.0, 3e6, 1e8):
    for n in (1024, 8192, 32768):
        lam = jnp.asarray(rate, dtype=jnp.float32)
        args = (jax.random.key(1803), lam, (n,), np.dtype("int32"), 100)
        for fn in (baseline, patched):
            fn(*args).block_until_ready()
        pairs = []
        for i in range(24):
            order = (("old", baseline), ("stable", patched))
            if i % 2:
                order = tuple(reversed(order))
            durations = {}
            for name, fn in order:
                start = time.perf_counter_ns()
                fn(*args).block_until_ready()
                durations[name] = (time.perf_counter_ns() - start) / 1e6
            pairs.append(durations)
        old = np.array([p["old"] for p in pairs])
        stable = np.array([p["stable"] for p in pairs])
        ratios = stable / old
        rng = np.random.default_rng(121)
        drawn = rng.integers(0, 24, (2000, 24))
        bootstrap = np.median(ratios[drawn], axis=1)
        record = {
            "rate": rate,
            "n": n,
            "old_median_ms": float(np.median(old)),
            "stable_median_ms": float(np.median(stable)),
            "paired_median_ratio": float(np.median(ratios)),
            "ratio_bootstrap_025": float(np.quantile(bootstrap, .025)),
            "ratio_bootstrap_975": float(np.quantile(bootstrap, .975)),
            "raw_pairs": pairs,
        }
        results.append(record)
        print(json.dumps({k: v for k, v in record.items()
                          if k != "raw_pairs"}), flush=True)

with open("poisson_pairwise_hotpath_20261026.json", "w") as fh:
    json.dump({
        "jax": jax.__version__,
        "backend": str(jax.devices()[0]),
        "note": "Independent source transplant; not a current-head JAX PR CI run",
        "results": results,
    }, fh, indent=2)
