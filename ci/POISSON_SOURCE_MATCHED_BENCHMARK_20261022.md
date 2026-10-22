# JAX Poisson #41228 — source-exact performance review

**Date:** 2026-10-22. **Scope:** CPU verification evidence only;
not an upstream algorithm patch, no claims of measured GPU speedups.

## Baseline validity

- The stable variant is the actual checkout's JIT-compiled
  `jax._src.random.core._poisson_rejection` method.
- The reference is generated from the **same source AST**, changing only the
  exact two pre-PR source details:
  1. restore the original **outer-loop hoist**
     `log_lam = lax.log(lam)`;
  2. restore `t = -lam + k * log_lam - lax_special.lgamma(k + 1)`
     instead of `t = _poisson_log_pmf(k, lam)`.
- Both use the same seed, input rates, dtype, vector shapes, stop bound and
  actual Hoermann transformed-rejection loop. We excluded JIT compilation,
  called `block_until_ready()`, and alternated the execution order in 15
  paired rounds per condition.
- This is more faithful than comparing only isolated `logpmf` expressions.
  It remains a microbenchmark of the native sampling function, not a whole
  ML training workload.
- Source code: [benchmark script](../ci/poisson_native_sampler_review_benchmark.py).
  Recorded CI run: [GitHub Actions 39808521472](https://github.com/lindicaphxag-tech/jax/actions/runs/39808521472).

## Observed median timings (patch / original; lower is better)

| lambda | samples | Ubuntu | Windows |
|---:|---:|---:|---:|
| 10 | 1,024 | 0.526 | 0.626 |
| 10 | 8,192 | 0.589 | 0.607 |
| 1,000 | 1,024 | 0.556 | 0.626 |
| 1,000 | 8,192 | 0.591 | 0.594 |
| 100,000,000 | 1,024 | 0.838 | 0.936 |
| 100,000,000 | 8,192 | 0.964 | 0.935 |

Ubuntu and Windows jobs completed successfully. The macOS job was
**still queued when this report was prepared**; do not portray the entire
matrix as successful until it completes.

Each runner installed checked-out JAX 0.12.0.dev20261022 with CPU jaxlib
0.11.2. JAX's public SciPy Poisson oracle spot-check at lambda 1e8 passed:
maximum absolute difference of the selected float32 log-PMF positions was
`7.152557373046875e-7`.

## Interpretation and acceptance criteria

- These runs **do not support** a hot-loop performance regression on either
  tested CPU operating system. Some cases appear faster; treat the magnitudes
  as environment-dependent observations, not platform-wide speedup promises.
- Gamma-function removal plausibly explains why a more elaborate but
  elementary-arithmetic formula can outperform the earlier implementation,
  but the runs alone do not establish a causal attribution to a single
  subexpression.
- The performance review addresses [upstream PR #41228](https://github.com/jax-ml/jax/pull/41228)'s
  maintainer concern about the transformed-rejection loop. Remaining
  maintainer feedback includes public API integration and test consolidation.
- The correction reuses established saddlepoint/Stirling numerics.
  It is a substantive cross-module correctness contribution, but **not a
  newly invented Poisson distribution or originality proof**.
- A final upstream acceptance decision must depend on the current reviewer's
  requirements, the target repository's native CI, and GPU/platform
  performance if requested.

## Connection and status

This evidence was first prepared for a reply to upstream PR #41228.
The connected GitHub integration returned HTTP 403 when attempting to post
that comment in the external repository, so **no upstream comment was
published by this action**. The fork benchmark PR and its immutable Actions
logs remain available for a contributor with write permission to reference.


## Independent seed holdout: actual sampler distribution correctness

**New evidence (2026-10-22):** the source-matched comparison now runs the
actual old and new compiled Poisson rejection samplers on **disjoint random
seed groups**, with identical keys in each matched pair.

- Discovery/pilot: seeds `16000..16007`, 8 × 8192 = 65,536 samples per
  implementation, at rates 1e6 and 1e8.
- Disjoint fixed-seed replication: seeds `36000..36011`, 12 × 8192 =
  **98,304 samples per implementation and rate**, at rates 1e6, 1e7, 1e8.
- Same sampler, dtype `int32`, JAX CPU backend, no Python reimplementation of
  sampling or selection. Values are converted to float64 for sample-moment
  summaries only. We report per-seed variance-ratio standard errors, not
  an unsupported end-to-end training metric.

| Poisson lambda | Old variance / lambda | Stable variance / lambda | Old normal-KS | Stable normal-KS |
|---:|---:|---:|---:|---:|
| 1e6 | 1.04287 | 1.00694 | 0.01018 | 0.00296 |
| 1e7 | 0.93491 | 1.00670 | 0.01580 | 0.00274 |
| 1e8 | **1.75308** | **1.00682** | 0.02810 | 0.00285 |

At lambda = 1e8, the per-seed standard error of the variance-to-lambda
ratio was **0.00692** for original, **0.00422** for stable (12 independent
seed blocks). The old sampler's +75.3% excess sample variance is far larger
than these within-run block errors. The pilot (different seeds) also measured
1.72197 vs 1.00101. The statistical effect is consistent across seeds.

The normal-KS column uses the Gaussian approximation to standardized Poisson
at large lambda, *not* an exact Poisson goodness-of-fit p-value.
Two OS runners using identical seeds produced identical sample values;
they are reproducibility checks **not independent statistical replications**.

The finite-precision mechanism: the old acceptance log-density subtracts
quantities of order lambda × log(lambda) before obtaining a log-PMF of order
log(lambda). At lambda = 1e8, float32 rounding can overwhelm the density
differences that matter to transformed rejection. Replacing this acceptance
calculation with the stabilized shared deviance/Stirling core corrects those
large-parameter sample moments in the observed tests.

The evidence establishes **correctness of a production algorithm's output
distribution** rather than just the scalar `logpmf` routine. It does not
establish a new sampling family or universal distributional convergence theorem.

[Holdout CI and raw machine logs](https://github.com/lindicaphxag-tech/jax/actions/runs/39811318899)

## Performance caution after repeated runners

Performance measurements across different runs are not monotone: Ubuntu
lambda=1e8, 8192-sample patched/original median ratio was around 0.96 in an
earlier 15-pair run, but **1.43 in a later run**. Windows runs remained near
0.93 for that case. Thus the data do **not** support claiming a robust global
speedup or absence of every slowdown. CI runner CPU contention and execution
environment must be controlled before making stronger runtime guarantees.
The causal correctness result is stronger than the performance conclusion.

## Native repository tests

On the Linux CI runner, **31 selected `PoissonLogPmf` public tests and
1 `PoissonLargeLambdaVariance` sampler test passed** using the real checkout
and native JAX test suite, independent of the source-extracted benchmark.
See the same [holdout workflow](https://github.com/lindicaphxag-tech/jax/actions/runs/39811318899).

**Upstream contribution status remains unchanged:** [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228)
is open with maintainer changes requested. This validation report is only in
the fork because the current connector returned HTTP 403 when attempting a
direct issue comment on the external repo.
