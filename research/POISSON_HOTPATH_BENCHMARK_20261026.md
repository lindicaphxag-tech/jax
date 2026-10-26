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


## 2026-10-26 local source-matched reproducibility audit (new evidence)

A fresh Linux CPU run on installed **JAX 0.9.0.1** extracted the actual released production `_poisson_rejection` method, replaced **only** its acceptance log-PMF with the exact mathematical operations from upstream candidate head `3894c33`, and ran 5 fixed random seeds x 6,000 int32 samples/rate. This explicitly accommodates the legacy module path `jax._src.random`; recent JAX main uses `jax._src.random.core`.

| lambda | baseline variance/lambda | stable variance/lambda | baseline median ms | candidate median ms |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 1.00353 | 1.00353 | 2.562 | 3.335 |
| 100 | 1.01176 | 1.01176 | 1.559 | 1.799 |
| 1e3 | 1.00600 | 1.00582 | 1.664 | 1.816 |
| 1e5 | 1.00889 | 0.99678 | 1.741 | 1.827 |
| 1e6 | 1.03426 | 0.99744 | 1.711 | 1.975 |
| 1e8 | 1.43034 | 0.99763 | 1.510 | 1.797 |

Additional local `jit`, `vmap`, `grad`, and second derivative checks on float32 vectors found expected finite results and gradients; notably `k=0, mu=0` yielded log-PMF=0 and first derivative -1. Large finite float32 center rates up to `1e38` produced log-PMF within `1e-5` of the asymptotic `-0.5*log(2*pi*mu)` reference on CPU.

**Authoritative newly added fork artifacts:**
- [Source-matched legacy/current-layout A/B](../benchmarks/poisson_source_matched_cpu_ab_20261026.py)
- [JIT/vmap/grad/Hessian and extreme-rate contract](../benchmarks/poisson_ad_contract_cpu_20261026.py)
- [Fork-only GitHub Actions workflow](../.github/workflows/poisson-cpu-contract-20261026.yml)

**Validation boundary:** the local executed harness implements the exact candidate arithmetic on an older JAX sampler body, not the newest JAX PR installation. The linked fork scripts were published after the local run; do not claim those exact checked-in files have passed GitHub Actions until a completed run is visible. At the most recent check, no Actions workflow run had appeared for the branch. Benchmark timings are device/workload-specific.

**Merge decision:** retain the stable formula but do not use the naive `lambda<1e6` fast-path threshold. The next upstream acceptance gate is a reproducible exact-head benchmark with all appropriate dtypes, mixed-rate shapes, and a maintainer-approved performance tradeoff; do not open duplicate PRs.


## Mixed-rate regression: transition and far-tail variance, fixed seeds (2026-10-26)

**Now in official upstream PR test** [jax-ml/jax#41228 at `280b8edf`](https://github.com/jax-ml/jax/commit/280b8edff3ecf6397d5906f71385fe50bb3e9b75). This changes the existing 20,000-draw Poisson variance regression from a single rate to two contiguous blocks of 10,000 draws: `[3e6]*10000 + [1e8]*10000`. JAX's vectorized sampler handles both in the same call; no additional test method or draws were added.

Locally installed **JAX 0.9.0.1 CPU** with actual source-matched BTRS (the only change is candidate #41228 acceptance log-PMF):

| Key seed | λ=3e6 old | λ=3e6 stable | λ=1e8 old | λ=1e8 stable |
|---:|---:|---:|---:|---:|
| 0 | 0.90502 | 1.00090 | 1.49778 | 1.00515 |
| 1 | 0.88729 | 0.99478 | 1.42163 | 0.99891 |
| 7 | 0.90231 | 0.99685 | 1.49184 | 1.00139 |
| 37 | 0.92976 | 0.98585 | 1.44336 | 0.98345 |
| 53 | 0.90077 | 1.00066 | 1.49748 | 0.98174 |

Numbers are blockwise **sample variance divided by λ**, with target 1.00. A separate local call to the unmodified released **public** `jax.random.poisson(jax.random.key(0), rate_vector, dtype=np.int32)` reproduced the exact old seed-0 values **0.9050187216 / 1.4977815172**. The candidate is not a separately installed latest-JAX source tree in this local control; it reuses the actual released rejection sampler with the precise PR formula substitution.

The two parameter regions produce **opposite signed variance errors** in the historical float32 sampler at equal sample budget. This justifies testing both instead of inferring all failure modes from one large mean. The five seeds are corroborative, not a confidence interval or a claim of all-dtype/platform validity. Full official JAX CI and maintainer approval remain pending.


## Independent 24-pair hot-loop performance audit (2026-10-26 evening)

**Scope:** locally installed JAX **0.9.0.1 CPU** only. Took the released production \`_poisson_rejection\` method body verbatim and replaced **only** the acceptance log-PMF with the operations at official PR #41228; identical PRNG key, compiled/JIT warmed before timing, \`block_until_ready\` on each run. Used 24 interleaved/reversed-order paired repetitions **per regime**, and the *same sampled key* for both implementations. Times below exclude JIT compilation. Do **not** present this as latest-JAX exact-head or GPU performance.

| mean | sample shape | original CPU median (ms) | candidate CPU median (ms) | median paired candidate/original |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 1,024 | 0.401 | 0.462 | 1.145 |
| 20 | 8,192 | 2.071 | 2.375 | 1.158 |
| 20 | 32,768 | 6.526 | 7.404 | 1.118 |
| 1,000 | 1,024 | 0.267 | 0.278 | 1.037 |
| 1,000 | 8,192 | 2.368 | 2.564 | 1.114 |
| 1,000 | 32,768 | 5.027 | 5.711 | 1.153 |
| 3e6 | 1,024 | 0.253 | 0.276 | 1.064 |
| 3e6 | 8,192 | 1.906 | 2.202 | 1.168 |
| 3e6 | 32,768 | 5.014 | 5.623 | 1.107 |
| 1e8 | 1,024 | 0.260 | 0.295 | 1.088 |
| 1e8 | 8,192 | 1.530 | 2.196 | 1.431 |
| 1e8 | 32,768 | 3.313 | 5.651 | **1.681** |

For the last two high-rate conditions, a nonparametric 2,000-resample bootstrap of the **24 paired ratios**, descriptive only given possible serial timing drift, gave percentile intervals **[1.378, 1.507]** at n=8192 and **[1.630, 1.720]** at n=32768. On the small n=1024 workloads several intervals included 1, and the dispatch, CPU load and compilation setup may dominate. This is a *performance regression signal* that must be acknowledged in review, not proof of a compiler- or hardware-invariant slowdown.

### Negative control: removing the separate far-tail log-PMF

A CPU source-transplant simplification replaced the far-tail ratio/log fallback with direct \`k*log1p(-(k-lambda)/k)+(k-lambda)\` outside the central series, intending to eliminate several hot-path logs. **It is NOT an acceptable public log-PMF replacement:** \`lambda=1e-8, k=1\` changes a finite log-PMF (about \`-18.42\`) to \`-inf\` due float32 rounding; for \`lambda=20, k=1e6\` it shifts the log probability by more than a thousand log units. Sampling the ordinary regimes happened to produce identical outputs for matched keys, but did not prove general correctness. 16-pair timings did **not** show stable improvement: at \`lambda=1e8, n=8192\`, stable and simplified medians were 2.035 and 2.040 ms respectively; at n=32768, 5.279 vs 5.129 ms. Therefore **do not promote this fast-path** into upstream production code.

### Reviewer decision implications

1. The unmodified complex stable formula solves an observed accuracy issue but has **measurable CPU sampling throughput cost**. Do not claim it is fast without exact-head platform controls.
2. Keep a faithful reference branch for the existing #41228 correctness fix. Compare simpler forms on a **separate fork** and reject any version that loses public support/tail semantics or sampling distribution consistency.
3. Repeat source-native performance tests on the exact current PR head with the JAX CI-supported CPU, then profile fusion and device-specific costs. The fork already has [a source-matched exact-head benchmark](../ci/poisson_native_sampler_review_benchmark.py) on its dedicated CI review branch; that script is **not** part of this benchmark branch. A prior exact-head native Actions run is not interchangeable with this new pairwise timing experiment.
