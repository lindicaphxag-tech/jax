# Upstream issue draft — loss of integer support in high-rate exact Poisson

**Suggested title:** `jax.random.poisson exact method loses integer residue classes at large rates despite moment-correct log-PMF`

**Related but not duplicate:** [JAX #41202](https://github.com/jax-ml/jax/issues/41202) and [PR #41228](https://github.com/jax-ml/jax/pull/41228) address **float32 log-likelihood cancellation / incorrect variance**. This is a separate **candidate-count representation** problem.

The public `jax.random.poisson(..., method='exact')` implementation casts `lam` to `float32`. The transformed-rejection proposal computes a float32 `floor(... + lam)` before converting to an integer. For mean `lambda=1e8`, the representable float32 values around the mean differ by **8**, so integer counts outside the `8*Z` lattice cannot normally be produced.

Minimal CPU reproduction, tested with **JAX 0.9.0.1**, without special hardware:

```python
import jax
import jax.numpy as jnp
import numpy as np
x = np.asarray(jax.random.poisson(jax.random.key(2026), 1e8,
                                   shape=(12000,), dtype=jnp.int32))
print(np.unique(x % 8))  # [0]
with jax.enable_x64(True):
    x64 = np.asarray(jax.random.poisson(
        jax.random.key(2026), jnp.asarray(1e8, jnp.float64),
        shape=(12000,), dtype=jnp.int64))
print(np.unique(x64 % 8))  # [0] as well
```

More rates `2e7, 4e7, 7e7, 1e8, 1.5e8` produced lattices of spacing `2,4,8,8,16`. Independent NumPy controls populated all residues. Replacing **only** the log-PMF acceptance expression with #41228's candidate restored sample variance/rate to approximately `1.01` in each case, but **did not fix the integer lattice**.

See:
- [Fork public-API reproducer](https://github.com/lindicaphxag-tech/jax/blob/research/poisson-integer-support-20261026/research/poisson_integer_support_20261026.py)
- [Derivation and boundary](https://github.com/lindicaphxag-tech/jax/blob/research/poisson-integer-support-20261026/research/POISSON_INTEGER_SUPPORT_BOUNDARY_20261026.md)
- [Fork-only integer-centered experimental production diff](https://github.com/lindicaphxag-tech/jax/compare/280b8edff3ecf6397d5906f71385fe50bb3e9b75...research/poisson-centered-integer-proposal-20261026)

The experimental integer-centered BTRS proposal restores tested residues and approximately correct moments; **it is not yet a proven exact sampler**, because its acceptance likelihood still rounds large integer counts to float32. It is not an upstream PR and should not be merged without further mathematical and native backend verification.

**Proposal:** keep the ongoing #41228 review focused, and treat full integer support and verified rejection-law correctness as a separate algorithmic improvement. A valid implementation needs integer offset handling and a provably appropriate acceptance predicate; output `int64` alone does not fix the forced float32 internal proposal.

**Submission status:** publishing a new issue to `jax-ml/jax` was attempted on 2026-10-26 but the connected GitHub integration returned **403**. This is only a draft preserved on the author's fork, **not an official upstream issue**.
