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
