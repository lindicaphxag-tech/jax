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

"""Softplus curvature must remain nonzero after sigmoid rounds to one."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np


class SoftplusHigherDerivativeTest(absltest.TestCase):
  def test_float64_tail_hessian_does_not_vanish(self):
    with jax.enable_x64():
      for x, expected in [
          (37.42994775023705, 5.551115123125775e-17),
          (50., 1.9287498479639178e-22),
          (-37.42994775023705, 5.551115123125775e-17),
          (-50., 1.9287498479639178e-22),
      ]:
        y = jnp.asarray(x, dtype=jnp.float64)
        second = jax.grad(jax.grad(jax.nn.softplus))(y)
        compiled = jax.jit(jax.grad(jax.grad(jax.nn.softplus)))(y)
        for actual in (second, compiled):
          self.assertTrue(bool(jnp.isfinite(actual)))
          np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)

  def test_float64_third_derivative_retains_signed_tail(self):
    with jax.enable_x64():
      xs = jnp.asarray([37.42994775023705, -37.42994775023705])
      third = jax.jit(jax.vmap(jax.grad(jax.grad(jax.grad(jax.nn.softplus)))))(xs)
      np.testing.assert_allclose(
          third, [-5.551115123125775e-17, 5.551115123125775e-17],
          rtol=1e-12, atol=0,
      )

  def test_first_derivative_and_forward_are_unchanged(self):
    with jax.enable_x64():
      xs = jnp.asarray([-50., -1., 0., 1., 37.42994775023705, 50.])
      values = jax.jit(jax.nn.softplus)(xs)
      np.testing.assert_allclose(
          values, np.logaddexp(np.asarray(xs), 0.), rtol=1e-12, atol=1e-14,
      )
      first = jax.jit(jax.vmap(jax.grad(jax.nn.softplus)))(xs)
      expected_first = np.exp(-np.logaddexp(-np.asarray(xs), 0.))
      np.testing.assert_allclose(first, expected_first, rtol=1e-12)

  def test_midrange_float32_curvature(self):
    xs = jnp.asarray([-10., -1., 0., 1., 10.], dtype=jnp.float32)
    curvatures = jax.jit(jax.vmap(jax.grad(jax.grad(jax.nn.softplus))))(xs)
    expected = np.exp(-np.logaddexp(-np.asarray(xs), 0.))
    expected *= np.exp(-np.logaddexp(np.asarray(xs), 0.))
    np.testing.assert_allclose(curvatures, expected, rtol=3e-5, atol=3e-8)


if __name__ == "__main__":
  absltest.main()
