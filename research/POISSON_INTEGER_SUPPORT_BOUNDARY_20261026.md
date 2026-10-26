# Independent Poisson correctness boundary: float32 output lattice

**Related but distinct:** [JAX #41202](https://github.com/jax-ml/jax/issues/41202) / [PR #41228](https://github.com/jax-ml/jax/pull/41228) fix *acceptance log-likelihood cancellation and biased moments*. This note concerns **discrete count support** even after those moments are restored. It is not part of the pending official PR and is not an assertion of upstream acceptance.

## Actual production mechanism

In current candidate source `jax/_src/random/core.py`, the `poisson(..., method='exact')` public wrapper converts every input rate to `np.float32` before dispatch. The BTRS proposal computes

```python
k = lax.floor((2*a/u_shifted+b)*u + lam + 0.43)
return k.astype(dtype)
```

Since `k` is a float32 value before integer conversion, adjacent integer counts are not representable for high means. For a positive float32 value in the binade `[2**e, 2**(e+1))`, its spacing is `2**(e-23)`. Near `lambda=1e8`, this spacing is 8.

Therefore repairing the log-PMF computation in PR #41228 can fix moment distortion **without** recovering Poisson's complete integer support.

## Reproduction, no physical GPU

[Public-native JAX diagnostic](../research/poisson_integer_support_20261026.py). Locally run against the actual released JAX **0.9.0.1 CPU** public API, with **12,000 samples per mean** and NumPy Poisson as independent control:

| rate | float32 ULP | JAX multiple-of-ULP | NumPy multiple-of-ULP | JAX variance/rate (unpatched release) |
| ---: | ---: | ---: | ---: | ---: |
| 2e7 | 2 | 100% | 50.41% | 1.4467 |
| 4e7 | 4 | 100% | 25.57% | 1.6434 |
| 7e7 | 8 | 100% | 12.76% | 1.2462 |
| 1e8 | 8 | 100% | 12.73% | 1.4318 |
| 1.5e8 | 16 | 100% | 6.48% | 1.2349 |

A **separate source-transplant counterfactual**, [independent CPU source-matched acceptance A/B](https://github.com/lindicaphxag-tech/jax/blob/research/poisson-hotpath-benchmark-20261026/benchmarks/poisson_source_matched_cpu_ab_20261026.py), held the released BTRS proposal, RNG, and integer conversion unchanged but substituted the full current candidate log-PMF arithmetic. With the same 12,000-draw workload, all samples still landed on the float32 lattice at each tested mean, **while corrected variance/rate was 1.0097, 1.0102, 1.0098, 1.0098, 1.0102** respectively. That counterfactual is **not a native exact-head CI run**; the production `float32` cast and proposal are directly visible on official PR source.

## Mathematical distinction: moments vs distribution

If a proposed distribution Q has support only on multiples of `m`, the total variation distance satisfies

`TV(Q, Poi(lambda)) >= P(Poi(lambda) not divisible by m)`.

By the roots-of-unity filter,

`P(Poi(lambda) divisible by m) <= 1/m + (m-1)/m * exp(-lambda * (1-cos(2*pi/m)))`.

For `m=8` and `lambda=1e8`, the right side is indistinguishable from `1/8` in ordinary precision, implying a **TV lower bound of nearly 7/8 IF Q is restricted to multiples of 8**. The qualifier matters: the observed sample was entirely on the lattice, but the source code alone does not prove that every possible extremely rare output is restricted to that binade. This is a rigorous conditional lower bound plus a deterministic floating-point mechanism, **not** a certified unconditional TV bound for the full JAX distribution.

## Correct scope and future acceptance gates

1. Keep #41228's finite-log-PMF/moment repair separately reviewable. Do **not** silently claim it makes high-rate Poisson sampling distributionally exact.
2. Define a target for **integer support correctness** under the default `jax_enable_x64=False`; simply changing the output to `int64` cannot restore lost integer residues after an internal float32 proposal.
3. Evaluate a discrete-reconstruction or higher-precision integer proposal algorithm, including correctness of the acceptance ratio, unbiased integer residues, cost and batching/vmap/JIT; do not propose unproved uniform post-hoc noise.
4. Add native source-head CPU tests and, if approved, GPU checks, with a deterministic residue-class red/green diagnostic and independent distributional validation.

Research result only. No algorithmic fix or additional official merged contribution is claimed.


## x64 enabled does not by itself restore integer support (local JAX 0.9.0.1 CPU)

The independent control also executed:

```python
with jax.enable_x64(True):
    samples = jax.random.poisson(
        jax.random.key(2026),
        jnp.asarray(1e8, dtype=jnp.float64),
        shape=(12000,), dtype=jnp.int64)
    assert np.array_equal(np.unique(np.asarray(samples) % 8), [0])
```

The resulting `int64` samples at `lambda=1e8` were **all divisible by 8**, and at `lambda=1.5e8` all were divisible by 16, with 12,000 outputs each. This follows from the **forced float32 internal rate cast in the exact public wrapper**; wider input/output precision is lost before the BTRS proposal. The relevant bug fix is about the **internal integer proposal and probability law**, not an output dtype preference.

One public issue creation attempt at `jax-ml/jax` failed with **403 insufficient GitHub integration permission**. Therefore this branch and any draft issue description are **fork research, not an upstream issue or PR**, and no acceptance is claimed.
