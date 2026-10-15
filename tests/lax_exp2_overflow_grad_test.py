# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overflow-safe exp2 JVP: a finite derivative can outlive its primal."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np


class Exp2DerivativeOverflowTest(absltest.TestCase):
  def test_overflowed_primal_has_finite_first_derivative(self):
    # https://github.com/jax-ml/jax/issues/40382
    with jax.enable_x64():
      x = jnp.asarray(1024.0000000000002, dtype=jnp.float64)
      reference = 1.2460659279419801e308
      for operation in (jax.lax.exp2, jnp.exp2):
        function = lambda z: jnp.sum(operation(z))
        self.assertTrue(bool(jnp.isinf(function(x))))
        for derivative in (
            jax.grad(function)(x),
            jax.jit(jax.grad(function))(x),
            jax.jvp(function, (x,), (jnp.ones_like(x),))[1],
        ):
          self.assertTrue(bool(jnp.isfinite(derivative)))
          np.testing.assert_allclose(derivative, reference, rtol=1e-12)

  def test_second_derivative_and_batched_gradients_remain_finite(self):
    with jax.enable_x64():
      x = jnp.asarray(1024.0000000000002, dtype=jnp.float64)
      reference = 8.637070847447417e307
      second = jax.jit(jax.grad(jax.grad(jax.lax.exp2)))(x)
      self.assertTrue(bool(jnp.isfinite(second)))
      np.testing.assert_allclose(second, reference, rtol=1e-12)

      inputs = jnp.asarray([-10, -1, 0, 1, 100, 1023, 1024],
                           dtype=jnp.float64)
      grads = jax.jit(jax.vmap(jax.grad(jax.lax.exp2)))(inputs)
      self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))
      np.testing.assert_allclose(
          np.asarray(grads[:5]),
          np.log(2) * np.exp2(np.asarray(inputs[:5])),
          rtol=1e-12,
      )

  def test_float16_and_float32_derivatives_outlive_their_primal(self):
    # Float16 and float32 have the same half-binade overflow window as
    # float64: the primal is infinite but ln(2) * 2**x is finite.
    for dtype, x, tolerance in (
        (jnp.float16, 16.0, 4e-3),
        (jnp.float32, 128.0, 2e-6),
    ):
      scalar = jnp.asarray(x, dtype=dtype)
      self.assertTrue(bool(jnp.isinf(jax.lax.exp2(scalar))))
      expected = np.log(2.0) * (2.0 ** x)
      for derivative in (
          jax.grad(jax.lax.exp2)(scalar),
          jax.jit(jax.grad(jnp.exp2))(scalar),
      ):
        self.assertTrue(bool(jnp.isfinite(derivative)))
        np.testing.assert_allclose(
            derivative, expected, rtol=tolerance,
        )

  def test_float64_finite_gradient_near_derivative_overflow_boundary(self):
    with jax.enable_x64():
      scalar = jnp.asarray(1024.5, dtype=jnp.float64)
      self.assertTrue(bool(jnp.isinf(jax.lax.exp2(scalar))))
      gradient = jax.jit(jax.grad(jax.lax.exp2))(scalar)
      self.assertTrue(bool(jnp.isfinite(gradient)))
      np.testing.assert_allclose(
          gradient, 1.7622033349063e308, rtol=1e-12,
      )

  def test_float32_and_complex_derivatives_remain_consistent(self):
    x = jnp.asarray([0., 1., 4., -4.], dtype=jnp.float32)
    grads = jax.vmap(jax.grad(jax.lax.exp2))(x)
    np.testing.assert_allclose(
        grads, np.log(2) * np.exp2(np.asarray(x)), rtol=2e-6,
    )
    z = jnp.asarray(1. + 2.j, dtype=jnp.complex64)
    _, tangent = jax.jvp(jax.lax.exp2, (z,), (jnp.ones_like(z),))
    np.testing.assert_allclose(
        tangent, np.log(2) * np.exp2(np.asarray(z)), rtol=2e-6,
    )


if __name__ == "__main__":
  absltest.main()
