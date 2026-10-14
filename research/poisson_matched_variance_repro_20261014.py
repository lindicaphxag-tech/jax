"""Independent matched-key CPU sampler stress test for jax-ml/jax#41228.

This source intentionally does NOT modify installed JAX and does NOT import
or validate the exact upstream PR head. It copies the installed 0.9.0.1 native
transformed-rejection implementation with precisely one numerical acceptance
log-PMF expression replaced by a standalone compensated expression.

Reproduction: Python 3.13, jax/jaxlib 0.9.0.1, numpy, scipy;
python research/poisson_matched_variance_repro_20261014.py

This is author-operated engineering evidence, not outside-lab academic review.
"""
import inspect
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import jax._src.random as jax_random_internal
import numpy as np


def stable_logpmf_central(k, lam):
    k = jnp.maximum(k, jnp.asarray(1.0, k.dtype))
    delta = k - lam
    t = delta / k
    small = jnp.abs(t) <= jnp.asarray(.01, k.dtype)
    t_small = jnp.where(small, t, 0.)
    # k*(log1p(-t) + t) = -k*sum(t**j/j, j=2..infinity).
    coeff = ((((((t_small / 8 + 1/7)*t_small + 1/6)*t_small + 1/5)
               *t_small + 1/4)*t_small + 1/3)*t_small + 1/2)
    dev_small = -k * t_small * t_small * coeff
    t_reg = jnp.where(small, 0., t)
    dev_reg = k * jnp.log1p(-t_reg) + delta
    dev = jnp.where(small, dev_small, dev_reg)
    inv_k = 1./k
    tail = inv_k*(1./12-inv_k*inv_k/360+inv_k**4/1260)
    log_norm = .5*(jnp.log(k)+jnp.log(jnp.asarray(2*np.pi,k.dtype)))+tail
    return dev-log_norm


def make_matched_sampler():
    source = inspect.getsource(jax_random_internal._poisson_rejection)
    needle = "t = -lam + k * log_lam - lax_special.lgamma(k + 1)"
    if source.count(needle) != 1:
        raise RuntimeError("Installed JAX internal acceptance expression changed")
    source = source.replace(needle, "t = stable_logpmf_central(k, lam)")
    environment = dict(vars(jax_random_internal))
    environment["stable_logpmf_central"] = stable_logpmf_central
    exec(source, environment)
    return environment["_poisson_rejection"]


def main():
    if jax.__version__ != "0.9.0.1":
        raise RuntimeError("Version-locked JAX 0.9.0.1 evidence, not exact PR-head validation")
    start = time.time()
    fixed = make_matched_sampler()
    result = {
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "cpu_backend": jax.default_backend(),
        "not_exact_upstream_pr_head": True,
        "one_change": "installed JAX _poisson_rejection t=logpmf expression",
        "per_rate": {}
    }
    if jax.default_backend() != "cpu":
        raise RuntimeError("This evidence was frozen for CPU")
    for rate in (1e6,1e7,1e8):
        lam = jnp.asarray(rate, dtype=jnp.float32)
        old_samples=[]
        fixed_samples=[]
        for seed in range(12345,12350):
            key = jax.random.key(seed)
            old_samples.append(np.asarray(
                jax.random.poisson(key, lam, shape=(5000,),dtype=jnp.int32),
                dtype=np.float64))
            fixed_samples.append(np.asarray(
                fixed(key,lam,(5000,),np.dtype("int32"),max_iters=10000),
                dtype=np.float64))
        old = np.concatenate(old_samples)
        new = np.concatenate(fixed_samples)
        result["per_rate"][str(int(rate))] = {
            "n_samples": int(len(old)),
            "original_variance_ratio": float(old.var(ddof=1)/rate),
            "stable_variance_ratio": float(new.var(ddof=1)/rate),
            "original_mean_error_over_sqrt_lam": float((old.mean()-rate)/np.sqrt(rate)),
            "stable_mean_error_over_sqrt_lam": float((new.mean()-rate)/np.sqrt(rate)),
            "fraction_exactly_same_samples": float(np.mean(old == new)),
        }
    result["elapsed_wall_s_not_performance_benchmark"] = time.time()-start
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
