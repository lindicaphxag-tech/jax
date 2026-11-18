# Paired CPU Poisson rejection experiment — 2026-11-18

**Review target:** [jax-ml/jax #41228](https://github.com/jax-ml/jax/pull/41228), recorded PR head `9428e21c09e6a19cadbf9ed02d6ec78b165f6246`.

## Important scope boundary

This experiment used **installed JAX 0.9.0.1 on a local CPU**.
The candidate `_poisson_log_pmf` is a manual transcription of the PR-head
numerical helper, wired into a rejection sampler adapted to this runtime's
private API. The **matched baseline** uses the *same* sampler loop, RNG
interface, candidate retention rule and shape, replacing only the
acceptance-score formula with the original cancellation-prone
`-lam + k * log(lam) - lgamma(k + 1)`.

**It was not run in an exact checkout of the PR head**. It must not be
represented as approved upstream CI, GPU/TPU evidence or a decisive
performance guarantee.

Parameters: batch 32,768; 15 paired seeds, 491,520 samples per
variant/rate; float32; maximum 100 iterations; compilation/warmup
excluded; ABBA per-seed timings; independent seed bootstrap (5,000 resamples)
of median new/old runtime ratio.

| Rate λ | Old pooled variance | Candidate pooled variance | Ideal | Median new/old paired time | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| 30 | 29.94657 | 29.94650 | 30 | 0.9763 | 0.9380–0.9980 |
| 10,000 | 9,993.72791 | 9,993.86961 | 10,000 | 0.96995 | 0.9526–0.9861 |
| 100,000,000 | 145,608,894.87 | 99,930,188.89 | 100,000,000 | 1.01171 | 0.9637–1.1975 |

Both variants produced zero `-1` nonacceptance outputs in this test.
The λ=1e8 variance discrepancy is large and reproducible, while the
timing data **do not establish a reliable speedup or slowdown** at that rate.
Smaller-rate apparent gains should not be generalized beyond this CPU setup.

Additional independent numerical tests against SciPy and 100-digit
mpmath passed 15/15. For extreme f32 far-tail log-probabilities of order
-5e23, accuracy must use a scale-aware relative tolerance, not a fixed
0.1 absolute error bound. The first attempt at an inappropriate
absolute bound failed; the corrected numerically meaningful assertion
passed. This is a **test-tolerance correction**, not a production bug fix.

## Acceptance plan

The existing branch also contains an **exact-head GitHub workflow**
(`.github/workflows/poisson-41228-benchmark.yml`) and benchmark source
(`research/poisson_41228_benchmark.py`) intended to run on the actual
PR checkout, but the required successful Actions-run receipt is still
absent. The head-checkout implementation has different internal import
and sampling APIs from installed 0.9.0.1; results from one cannot be
silently attributed to the other. The reviewer still has
`CHANGES_REQUESTED`; separate source-exact CI plus simplified tests are
needed before claiming this gate resolved.
