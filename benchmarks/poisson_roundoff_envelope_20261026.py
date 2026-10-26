"""Cancellation envelope for the historic float32 BTRS acceptance log-PMF.

Run on JAX CPU after installing dependencies (JAX, NumPy, SciPy).
No production patch is applied. Compare the exact old arithmetic grouping
to the double-precision SciPy reference at float32-representable k.
Rates only up to 1e8; this is not a general high-precision certification.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lax import special as lax_special
from scipy.special import gammaln

RATES = (1e3, 1e4, 3e4, 1e5, 2e5, 3e5, 5e5,
         7.5e5, 1e6, 2e6, 1e7, 1e8)
SIGMAS = np.asarray([-3., -1., 0., 1., 3.])

@jax.jit
def original_logpmf(k, lam):
    return -lam + k * jnp.log(lam) - lax_special.lgamma(k + 1)

print("JAX", jax.__version__, "devices", jax.devices())
for rate in RATES:
    k = jnp.asarray(rate + SIGMAS * np.sqrt(rate), dtype=jnp.float32)
    lam = jnp.asarray(rate, dtype=jnp.float32)
    old = np.asarray(original_logpmf(k, lam), dtype=np.float64)
    k64 = np.asarray(k, dtype=np.float64)
    lam64 = float(np.asarray(lam, dtype=np.float64))
    reference = -lam64 + k64 * np.log(lam64) - gammaln(k64 + 1.)
    err = np.abs(old - reference)
    print(f"rate={rate:10g} max_abs_error={err.max():10.5f} "
          f"old_at_center={old[2]:10.5f} ref_at_center={reference[2]:10.5f}",
          flush=True)
