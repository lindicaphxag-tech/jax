# Poisson BTRS hot path — verified numerical/performance boundaries (2026-10-26)

Upstream: [JAX #41228](https://github.com/jax-ml/jax/pull/41228). These are **independent fork experiments**, not upstream acceptance or latest-main integration.

## Numerical signal and observed tradeoff

On locally installed **JAX 0.9.0.1 CPU**, a source-preserving BTRS experiment swapped the historical rejection acceptance log-PMF with the mathematical formula currently proposed in #41228. The recorded one-seed experiment at `lambda=1e8`, 20,000 draws, had sample variance / lambda **1.469** (historical) versus **0.993** (candidate); repeated fixed-seed validation with 5 seeds / 100,000 draws had **1.44675** versus **0.99784**. Compiled CPU latency was noisy across runs, and the candidate exhibited overhead on some runs. **Do not claim a stable speedup or slowdown ratio.** Current JAX main, GPU, multi-shape inputs and full PR tests were not exercised.

Relevant diagnostic scripts:
- [Source-matched BTRS three-arm benchmark](../benchmarks/poisson_btrs_multiseed_diagnostic_20261026.py): designed for the exact PR checkout; it has **not yet been run there**.
- [Float32 roundoff-envelope probe](../benchmarks/poisson_roundoff_envelope_20261026.py): run on local JAX 0.9.0.1, compares historical arithmetic against SciPy float64 at float32-representable inputs.

## Critical falsification: naive fixed-rate gating is unsafe

A previous fork-only *experimental* selective arm used the historical fast cancellation-prone expression when **all rates < 1e6**. **This heuristic is not recommended.** Direct float32 log-PMF errors near the acceptance-critical region `k=lambda+{-3,-1,0,1,3}*sqrt(lambda)` show significant error **below** 1e6:

| lambda | Maximum absolute error of historical log-PMF |
| ---: | ---: |
| 1e3 | 0.00038 |
| 1e4 | 0.01048 |
| 3e4 | 0.05158 |
| 1e5 | 0.07566 |
| 2e5 | 0.27870 |
| 5e5 | 0.47963 |
| **7.5e5** | **1.31715** |
| 1e6 | 0.82669 |
| 1e7 | 24.97799 |
| 1e8 | 270.62958 |

This is the historical JAX computation against SciPy double-precision reference at the same float32-representable k, so it measures **roundoff in the acceptance log likelihood**, not simply stochastic sample variance. The acceptance threshold compares log quantities, so log-PMF error of order one can materially alter rejection decisions even when five-seed variance appears near one. Float64 SciPy here is only a reference in the tested rate range, not a universal arbitrary-precision oracle.

The experiment is **falsified as a general accuracy-preserving optimization** despite encouraging throughput timings. Its purpose on this fork is to retain the negative control and prevent premature upstream use; it is not a proposed production change.

## Next acceptance gates

1. Test current #41228 **exact-head** source and native JAX tests, including batched mixed rates and both float32/float64.
2. Profile the stable computation and consider algebraic simplification or a **provable error-based** selective strategy, not a heuristic threshold.
3. Preserve public log-PMF domain semantics, JIT/autodiff behavior, and statistically controlled sampling tests.
4. Request maintainer re-review only after benchmark logs are grounded in the exact changed revision; no external PR comment was posted because the available GitHub integration returned 403.

No new upstream merged commit is claimed.
