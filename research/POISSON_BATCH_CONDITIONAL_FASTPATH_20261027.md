# JAX Poisson rejection: guarded central-region hot path (fork research, 2026-10-27)

Status: an independent fork-only prototype based on official JAX PR #41228 head 280b8edf. This is not an accepted upstream PR, merge, or full latest-main CI pass.

## Mechanism and equivalence scope

The proposed private helper `_poisson_rejection_log_pmf` uses an all-elements JAX runtime guard: safe_k > 10, lambda > 0 and abs((safe_k-lambda)/safe_k) <= 0.01. Only when **every** proposal in a rejection iteration passes does it compute the selected eighth-degree central-deviance polynomial and asymptotic Stirling correction. Otherwise it invokes the unmodified full `_poisson_log_pmf` for the entire batch.

Inside this domain, lambda/safe_k >= 0.99, so the original log-PMF never selects its far-tail expression. Also safe_k > 10 means its asymptotic Stirling term is selected. Consequently the central branch evaluates the same selected mathematical formula, apart from possible floating-point fusion/rounding. All public jax.scipy.stats.poisson.logpmf calls remain unchanged.

This is **not** a proof of bitwise identity on every device/key. A batch-wide reduction inside the rejection loop may be expensive on XLA/GPU or for mixed and heavy-tailed batches; all device-performance claims remain provisional.

## Reproduced local JAX 0.9.0.1 CPU validation

Reference = the current #41228 full stable acceptance log-PMF transplanted into the released native BTRS method. Experimental = same BTRS source and PRNG, substituting only this private batch-conditional helper. Warmed JIT, 24 interleaved reversed-order timing pairs per condition; timings exclude compilation.

| λ | Draws | Experimental/full paired median ratio |
|---:|---:|---:|
| 20 | 8192 | 0.810 |
| 1000 | 8192 | 0.842 |
| 3e6 | 8192 | 0.819 |
| 1e8 | 8192 | 0.835 |
| 1e8 | 32768 | 0.845 |

Source-matched checks: 172,032 fixed-seed samples across six scalar rates all produced the same integer results; four mixed-rate configurations across eight seeds also matched exactly. A broad float32/float64 log-PMF edge grid including zero/near-zero rates and extreme counts was identical, aside from central floating-point differences up to approximately 9.5e-7. These are finite-sample checks, not formal complete distribution correctness proofs.

## Reproducible artifacts and adoption gates

- Experimental production source: [jax/_src/random/core.py](../jax/_src/random/core.py)
- Exact-source comparison and 24-pair benchmark: [ci/poisson_btrs_conditional_guard_20261027.py](../ci/poisson_btrs_conditional_guard_20261027.py)
- Official pending PR remains untouched: https://github.com/jax-ml/jax/pull/41228

The new fork code and exact-checkout regression script are pushed, but have NOT been demonstrated passing latest-main native GitHub Actions. Complete current-head CPU tests, mixed-rate and extreme-tail sampling, XLA GPU performance, JIT/vmap and appropriate dtype/algorithm semantics before promotion. A separate high-rate integer-lattice defect remains unresolved and is not addressed by this performance optimization.

Prior negative control: dropping the far-tail formula entirely broke public log-PMF finite extreme values; this helper retains the full fallback.


## 2026-10-27 extra mixed-rate stress test (run locally, not exact PR head)

A second **independent local JAX 0.9.0.1 CPU run** used the actual released BTRS method body with full versus guarded PR log-PMF source substitutions, 4096 draws, 12 interleaved JIT-warmed CPU timing pairs per case. Exact same-key integer output equality was observed in all five cases:

| Workload | Guarded/full median paired time | Full median ms | Guarded median ms |
|---|---:|---:|---:|
| Uniform λ=1e3 | 0.93793 | 1.504 | 1.409 |
| Uniform λ=1e8 | 0.86085 | 1.166 | 0.993 |
| 4-value mix [300, 1e3, 3e6, 1e8] | 0.86184 | 1.148 | 0.979 |
| 4095×1e8 plus 1×10 | 0.90457 | 1.176 | 1.050 |
| Repeated [10, 11, 20, 1e3, 1e8] | 0.89057 | 1.815 | 1.619 |

[Recorded JSON statistics](POISSON_GUARDED_MIXED_CPU_RESULTS_20261027.json) and the [extended exact-checkout benchmark gate](../ci/poisson_btrs_conditional_guard_20261027.py) are committed. The JSON is a record of the **local released-runtime source-matched run**, not the output of the freshly committed exact-checkout script. Relative speedups are exploratory and should be repeated on the true target JAX PR head and GPU.

An attempt to edit the official JAX #41228 PR description to share the performance tradeoff and this new candidate returned **403 Resource not accessible by integration**. It is inaccurate to claim the description or maintainer thread was updated; these findings remain in the author's fork unless shared through a permitted interface.
