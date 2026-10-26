"""JAX #41228: finite AD/transform contract on the stable log-PMF.

Independent fork-only CPU regression. This checks the mathematical candidate
and JAX transforms, not the full latest-main PR test suite.
Run: python benchmarks/poisson_ad_contract_cpu_20261026.py
"""
import math
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from poisson_source_matched_cpu_ab_20261026 import candidate_pmf


class PoissonLogPmfTransformTest(unittest.TestCase):
    def test_zero_count_zero_rate_jit_grad_hessian(self):
        k = jnp.float32(0.)
        for mu in (0., 1e-12, 1e-8):
            x = jnp.float32(mu)
            fn = lambda rate: candidate_pmf(k, rate)
            self.assertAlmostEqual(float(jax.jit(fn)(x)), -mu, delta=1e-6)
            self.assertEqual(float(jax.jit(jax.grad(fn))(x)), -1.)
            self.assertEqual(float(jax.jit(jax.grad(jax.grad(fn)))(x)), 0.)

    def test_vectorized_jit_autodiff(self):
        k = jnp.array([0., 0., 1., 2., 100.], dtype=jnp.float32)
        mu = jnp.array([0., 1e-12, 1e-8, 1e-6, 1e8], dtype=jnp.float32)
        scalar = lambda rate, count: candidate_pmf(count, rate)
        gradients = np.asarray(jax.jit(jax.vmap(jax.grad(scalar, argnums=0)))(mu, k))
        hessians = np.asarray(jax.jit(jax.vmap(jax.grad(jax.grad(scalar, argnums=0), argnums=0)))(mu, k))
        g_ref = np.array([-1., -1., 1e8-1., 2e6-1., 100./1e8-1.])
        h_ref = np.array([0., 0., -1e16, -2e12, -100./1e16])
        np.testing.assert_allclose(gradients, g_ref, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(hessians, h_ref, rtol=3e-5, atol=3e-5)
        values = np.asarray(jax.jit(jax.vmap(candidate_pmf))(k, mu))
        self.assertTrue(np.isfinite(values).all())

    def test_float32_extreme_center_stays_finite(self):
        for rate in (1e8, 1e15, 1e25, 1e35, 1e38):
            x = jnp.float32(rate)
            actual = float(jax.jit(candidate_pmf)(x, x))
            reference = -.5 * (math.log(2 * math.pi) + math.log(float(x)))
            self.assertTrue(math.isfinite(actual))
            self.assertAlmostEqual(actual, reference, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
