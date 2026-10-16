from absl.testing import absltest

import numpy as np

import jax.numpy as jnp
from jax import random
from jax import scipy as jsp
from jax._src import test_util as jtu


class ChoiceZeroProbabilityTest(jtu.JaxTestCase):

  def test_zero_probability_entries_are_never_sampled(self):
    # Reduced from the reported million-element failure mode: a long softmax
    # distribution followed by an exact zero-probability tail can acquire
    # artificial cumulative mass at the zero entries under parallel cumsum.
    n = 1_000_000
    zero_tail = 10_000
    key = random.key(100)
    logits = random.normal(key, (n,), dtype=np.float32)
    logits = logits.at[-zero_tail:].set(-jnp.inf)
    probs = jnp.exp(logits - jsp.special.logsumexp(logits))

    samples = np.asarray(
        random.choice(
            random.key(101), n, shape=(100_000,), replace=True, p=probs
        )
    )

    self.assertEqual(np.intersect1d(samples, np.arange(n - zero_tail, n)).size, 0)

  def test_zero_probability_support_is_preserved_under_jit(self):
    probs = np.array([0.0, 0.25, 0.0, 0.75, 0.0], dtype=np.float32)

    def sample(key):
      return random.choice(key, 5, shape=(4096,), replace=True, p=probs)

    samples = np.asarray(jtu.jit(sample)(random.key(123)))
    self.assertEqual(np.intersect1d(samples, np.array([0, 2, 4])).size, 0)


if __name__ == "__main__":
  absltest.main()
