# Exact-head acceptance audit for jax-ml/jax #41228
Date: 2026-10-23
Upstream PR: https://github.com/jax-ml/jax/pull/41228
Pinned upstream PR head: `80e1a31993374837801f6fe841d8af5fcd7b164f`
This branch starts **exactly** at that commit and adds only independent validation files; it does not change upstream implementation or tests.

## Primary objective
Address the maintainer's explicitly requested *native rejection-loop performance* assessment using an identical source-matched comparator. Do not infer end-to-end training gains from timings.

## Executable validation
The `ci/poisson_native_sampler_review_benchmark.py` script:
1. Uses the checked-out `jax._src.random.core._poisson_rejection` itself as the patched implementation.
2. AST-transforms that same function to recover the original hoisted `log_lam` and the legacy acceptance log-PMF for a matched comparator.
3. JIT-warms both variants, alternates paired timing order with `block_until_ready()`, and reports CPU measurements at rates 10, 1000, and 1e8 for shape sizes 1024 and 8192.
4. Tests disjoint seed blocks at rates 1e6, 1e7, and 1e8 with sample variances and standardized normal-approximation KS diagnostics.
5. Checks the public log-PMF against a SciPy oracle and runs native Poisson public API and high-rate sampler tests.

The workflow also executes `ci/poisson_central_deviance_remainder_certificate.py`. The 8-term central Taylor truncation bound is mathematical truncation evidence only, **not** a complete floating-point rounding bound.

## Existing evidence (older source head; do not relabel as exact-head results)
Previous fork CI on October 22:
- https://github.com/lindicaphxag-tech/jax/actions/runs/39811318899
- https://github.com/lindicaphxag-tech/jax/actions/runs/39816810530
These reported held-out variance/lambda 1.75308 (legacy) vs 1.00682 (stable) at lambda=1e8, but performance ratios varied across runners; there is no reliable blanket speedup assertion. The current branch is intended to revalidate against the precise PR head above.

## Review/merge gates
- Record **this branch's actual workflow run ID and outcome** before claiming current-head reproducibility.
- Verify JAX native tests, x64-sensitive cases, lint, and exact upstream status on the currently reviewed head.
- Reviewer jakevdp also requested test consolidation and parameterization; retain unique boundary/autodiff cases, avoid redundant cross-suite tests.
- Keep benchmark files **outside** the upstream PR. Merged count does not change until upstream actually merges.

## Limitations
CPU only; timing noise, runner contention and backend/compiler variation limit performance generalization. The original reference is a source-level reconstruction, not a separate tagged release. External reviewer approval is not yet recorded.
