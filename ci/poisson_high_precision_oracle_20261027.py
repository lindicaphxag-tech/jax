"""100 decimal digit validation for stable Poisson log-PMF at large rates.

Run in editable checkout of JAX #41228:
  pip install mpmath scipy
  python ci/poisson_high_precision_oracle_20261027.py

Uses the actual private JAX log-PMF and public scipy.stats API, not a
transcribed helper. mpmath sees *exact represented* float32/float64 input
integers, rather than the nominal decimal rates before floating rounding.
This is a fork-only numerical gate; not official upstream CI.
"""
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from scipy.stats import poisson
from jax._src.random import core as random_core


def trial(dtype):
    with jax.enable_x64(True):
        count, mean = [], []
        for rate in (10, 100, 1000, 100000, 10**7, 10**8, 10**9, 10**10, 10**12):
            for shift in (-20, -7, -1, 0, 1, 7, 20):
                rate_rounded = dtype(rate)
                k = dtype(max(0, round(
                    float(rate_rounded) + shift * math.sqrt(float(rate_rounded))
                )))
                assert np.isfinite(k)
                mean.append(rate_rounded)
                count.append(k)
        k = np.asarray(count, dtype=dtype)
        lam = np.asarray(mean, dtype=dtype)
        actual = np.asarray(
            jax.jit(random_core._poisson_log_pmf)(jnp.asarray(k), jnp.asarray(lam))
        )
        public = np.asarray(
            jax.jit(jax.scipy.stats.poisson.logpmf)(jnp.asarray(k), jnp.asarray(lam))
        )
        np.testing.assert_array_equal(public, actual)
        scipy = poisson.logpmf(k.astype(np.float64), lam.astype(np.float64))
        mp.mp.dps = 100
        rows = []
        for a, b, got, s in zip(k, lam, actual, scipy, strict=True):
            # Test-grid inputs are integer-valued exactly after dtype rounding.
            ai, bi = int(a), int(b)
            oracle = float(
                -mp.mpf(bi) + mp.mpf(ai) * mp.log(mp.mpf(bi))
                - mp.loggamma(mp.mpf(ai) + 1)
            )
            rows.append({
                "dtype": np.dtype(dtype).name,
                "rate": bi, "k": ai,
                "jax": float(got), "scipy": float(s), "mpmath100": oracle,
                "jax_error": abs(float(got) - oracle),
                "scipy_error": abs(float(s) - oracle),
            })
        maximum = max(item["jax_error"] for item in rows)
        assert maximum < (0.0005 if dtype is np.float32 else 1e-8), maximum
        return rows


def main():
    rows = trial(np.float32) + trial(np.float64)
    summary = []
    for dtype in ("float32", "float64"):
        sub = [x for x in rows if x["dtype"] == dtype]
        jmax = max(sub, key=lambda x: x["jax_error"])
        smax = max(sub, key=lambda x: x["scipy_error"])
        v = {"dtype": dtype, "cases": len(sub), "worst_jax": jmax,
             "worst_scipy": smax}
        summary.append(v)
        print(json.dumps(v, sort_keys=True), flush=True)
    result = {
        "jax_version": jax.__version__,
        "module": random_core.__file__,
        "reference": "mpmath 100 decimal digits on exact float-representable inputs",
        "summary": summary, "rows": rows,
    }
    path = Path("poisson_high_precision_oracle_20261027.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("PASS", len(rows), "cases")


if __name__ == "__main__":
    main()
