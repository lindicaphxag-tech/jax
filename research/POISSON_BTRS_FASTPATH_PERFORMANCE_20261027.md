# JAX #41228: BTRS central fast-path performance study

Date: 2026-10-27. Research branch: `research/poisson-fastpath-benchmark-20261027`, based on PR head `757e072f3af323b09a3e6e0870185283d21fa034`.

## Why this experiment exists

The open review thread asks whether the more involved stable Poisson log-PMF is too expensive inside the rejection-sampling loop. The production patch adds a bounded sampler-only fast path: only when **every** proposal in the batch has `lambda >= 1e6`, `k > 10`, and `abs((k-lambda)/k) <= 0.01` does it evaluate the central series/Stirling expression directly. Any batch with a lower rate, small count, or one out-of-domain proposal falls back to the complete stable helper. The public `jax.scipy.stats.poisson.logpmf` path remains unchanged.

## Preliminary three-way CPU timing

Environment: installed JAX/jaxlib 0.9.0.1, CPU. The benchmark extracts the installed BTRS method and substitutes only the acceptance-PMF expression. It is a **source-matched experiment, not execution of the exact current PR checkout**, not official upstream CI, and not GPU evidence. 24 warmed, interleaved timing measurements per cell; compilation excluded.

| lambda | n | Old expression (ms) | Full stable helper (ms) | Guarded fast path (ms) | Fast/full ratio |
|---:|---:|---:|---:|---:|---:|
| 1e6 | 8,192 | 1.688 | 1.989 | 1.683 | 0.846 |
| 1e6 | 32,768 | 4.631 | 4.755 | 4.199 | 0.883 |
| 3e6 | 8,192 | 1.956 | 2.101 | 1.796 | 0.855 |
| 3e6 | 32,768 | 5.116 | 4.827 | 4.157 | 0.861 |
| 1e8 | 8,192 | 1.582 | 2.124 | 1.827 | 0.860 |
| 1e8 | 32,768 | 4.105 | 4.672 | 4.108 | 0.879 |

The guarded path was about 12–15% faster than the full stable helper in the larger central-batch regimes. At `lambda=1e8, n=8192`, it remained ~15% slower than the old expression; at `n=32768` it was approximately tied with the old expression. Timing is host-dependent and no universal speedup is claimed.

At `lambda=1e8`, the recorded old-expression samples had variance/`lambda` around 1.44; stable and guarded paths were around 1.01. These are fixed-key batch diagnostics, not a replacement for the statistical regression test.

## Correctness checks

- **172,032 held-out same-key CPU samples** matched exactly between the full stable helper and the guarded sampler variant, across six rates and seven seeds.
- On a 49-point central-domain grid, the standalone central expression differed from the full helper by at most `9.54e-7`; the regression test uses absolute tolerance `5e-5`.
- For mixed-tail batches, the compiled guarded helper chose the full stable fallback and matched its output exactly. In a synthetic 32,768-element helper benchmark, one far-tail item added ~2.4% median time and 5% tail items added ~14.9%; this is a stress test of the fallback predicate, not a sampler-distribution result.

## Reproducibility and status

- [Exact-source benchmark script](../ci/poisson_sampler_fastpath_benchmark_20261027.py) imports the actual current checkout's `_poisson_log_pmf`, `_poisson_log_pmf_for_rejection`, and `_poisson_rejection`; it reconstructs the old and full-stable variants by replacing only the call site. Run it from this branch to get a new local JSON file.
- [Raw source-matched measurements](POISSON_BTRS_FASTPATH_CPU_RESULTS_20261027.json).
- The official PR head has the fast path and two focused tests, but the exact-source benchmark script has **not yet been confirmed running on that head**. Current observed official checks are CLA and `check-changes`; full native test CI is not confirmed.
- The connected GitHub integration returns 403 for writing a reply to the maintainer's inline review. The production code and tests are on the PR branch, but the benchmark explanation has not been successfully posted into that review thread.

## Merge recommendation

Keep this optimization only if maintainers consider the hot-path tradeoff important. It is deliberately sampler-local, gated to batches that satisfy the existing central-domain conditions, and retains the full stable fallback for everything else. If reviewers prefer minimum diff size, revert the guarded helper and retain the numerical correctness fix with the honest measured overhead.
