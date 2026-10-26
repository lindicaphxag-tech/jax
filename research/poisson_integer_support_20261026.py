"""CPU reproduction: JAX Poisson exact sampler loses integer support at large lambda.

Run from an editable JAX checkout with this research file:
  python research/poisson_integer_support_20261026.py
  python research/poisson_integer_support_20261026.py --require-full-support

The second command is intentionally red until a support-preserving sampler
is implemented. This is independent of PR #41228's log-PMF variance fix.
"""
import argparse
import math
import numpy as np
import jax
import jax.numpy as jnp


def poisson_residue_probability_upper_bound(lam, modulus):
    """Root-of-unity upper bound on P(Poisson(lam) % modulus == 0)."""
    theta = 2 * math.pi / modulus
    return 1 / modulus + (1 - 1 / modulus) * math.exp(
        -lam * (1 - math.cos(theta))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-full-support", action="store_true")
    args = parser.parse_args()
    print("JAX", jax.__version__, "device", jax.devices(), flush=True)
    n = 12000
    rng = np.random.default_rng(2026)
    for lam in (2e7, 4e7, 7e7, 1e8, 1.5e8):
        modulus = int(np.spacing(np.float32(lam)))
        assert modulus in (2, 4, 8, 16)
        # Exercise the real, publicly exposed, default exact method.
        draws = np.asarray(
            jax.random.poisson(jax.random.key(2026), lam, shape=(n,),
                               dtype=jnp.int32),
            dtype=np.int64,
        )
        control = rng.poisson(lam, size=n)
        support_fraction = float(np.mean(draws % modulus == 0))
        control_fraction = float(np.mean(control % modulus == 0))
        var_ratio = float(np.var(draws.astype(float)) / lam)
        tv_if_always_on_grid = (
            1 - poisson_residue_probability_upper_bound(lam, modulus)
        )
        print(
            f"lambda={lam:g} float32_ulp={modulus} n={n} "
            f"JAX_on_lattice={support_fraction:.4f} "
            f"NumPy_on_lattice={control_fraction:.4f} "
            f"JAX_variance_over_lambda={var_ratio:.5f} "
            f"TV_lower_bound_if_always_on_lattice={tv_if_always_on_grid:.6f}",
            flush=True,
        )
        if args.require_full_support:
            assert support_fraction < 0.95, (
                "Missing integer residue classes", lam, modulus
            )
    print("Observing near-correct variance does NOT establish distributional parity.")


if __name__ == "__main__":
    main()
