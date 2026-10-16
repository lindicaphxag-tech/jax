from absl.testing import absltest

import numpy as np

from jax import random
from jax._src import test_util as jtu


class ChoiceZeroProbabilityTest(jtu.JaxTestCase):

  def test_zero_probability_entries_are_never_sampled(self):
    n = 100_000
    zero_tail = 10_000
    probs = np.ones(n, dtype=np.float32)
    probs[-zero_tail:] = 0

    samples = np.asarray(random.choice(
        random.key(0), n, shape=(100_000,), replace=True, p=probs))

    self.assertEqual(np.intersect1d(samples, np.arange(n - zero_tail, n)).size, 0)

  def test_zero_probability_support_is_preserved_under_jit(self):
    probs = np.array([0.0, 0.25, 0.0, 0.75, 0.0], dtype=np.float32)

    def sample(key):
      return random.choice(key, 5, shape=(4096,), replace=True, p=probs)

    samples = np.asarray(jtu.jit(sample)(random.key(123)))
    self.assertEqual(np.intersect1d(samples, np.array([0, 2, 4])).size, 0)


if __name__ == "__main__":
  absltest.main()
