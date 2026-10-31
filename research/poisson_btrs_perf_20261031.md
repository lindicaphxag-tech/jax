# Source-extracted Poisson BTRS CPU benchmark — 2026-10-31

## Why this exists
JAX PR [#41228](https://github.com/jax-ml/jax/pull/41228) has a specific maintainer review question: the numerically stable log-PMF is evaluated inside a repeated rejection-sampling loop, so correctness alone is not sufficient. This branch leaves the **official PR unchanged**. `poisson_btrs_perf_20261031.py` imports the proposed helper from the PR head and compares it against the old cancellation-prone log-PMF in the same source-extracted BTRS loop.

## Independent local pilot (not upstream CI)
Environment: Linux x86-64, Python 3.13.5, JAX/jaxlib 0.9.0.1, CPU backend, NumPy 2.3.5, SciPy 1.17.0. Pilot used a manually transcribed PR helper with a compatibility-only change to its small-k clamp (`jnp.clip` replacing upstream's `lax.clamp(lax._const(...))`); public JAX PRNG helpers replace JAX internal `_split`. Same fixed key for two alternatives, 8,192 samples each, 15 warm JIT runs; ratios are **environment-specific**, not a project performance promise.

| dtype | rate | old median (ms) | proposed median (ms) | ratio new/old | variance old | variance new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 100 | 1.762 | 2.019 | 1.15 | 103 | 103 |
| float32 | 1e6 | 1.756 | 1.978 | 1.13 | 1,053,372 | 1,029,869 |
| float32 | 1e8 | 1.605 | 2.117 | 1.32 | 147,955,609 | 103,055,498 |
| float64 | 100 | 4.886 | 3.899 | 0.80 | 102 | 102 |
| float64 | 1e6 | 3.735 | 3.991 | 1.07 | 1,016,965 | 1,016,965 |
| float64 | 1e8 | 3.730 | 3.621 | 0.97 | 101,655,249 | 101,655,249 |

At float32 rate 1e8, the old source-extracted sampler shows ~47.96% variance inflation, while the proposed helper gives ~3.06% relative variance difference in this one fixed-seed finite sample. The new helper incurs ~32% median loop latency overhead in this pilot. The number of samples is too small to establish a general distributional guarantee; use multiple independent keys, confidence intervals, and current-PR native CI before claiming either robust quality or throughput.

The separate proposed log-PMF check against arbitrary precision at `lambda=1e12` returned float32 max absolute difference approximately 1.26e-6 and float64 essentially machine-level agreement for the five tested cases. These accuracy checks also use a local source-extracted implementation.

## Mechanism and invariant
**Required invariant:** rejection acceptance must be based on an accurately computed log-PMF rather than cancellation-prone large terms. **Cost constraint:** improved accuracy must be evaluated in the actual repeated rejection path, not only in a vectorized single-function benchmark. A correctness fix may trade off runtime; neither the variance repair nor speed can be inferred from a `logpmf` microbenchmark alone.

## Review / CI boundary
The maintainer's performance-review thread remains open; this independent research evidence does **not** substitute for a benchmark on the exact upstream PR checkout, latest JAX/jaxlib, CPU/GPU devices, full CI, or review approval. Run `python research/poisson_btrs_perf_20261031.py` after installing/checking out this research branch and compare the JSON. Do not merge a performance optimization based solely on this pilot.

## Explain in 30 seconds
“The old Poisson rejection acceptance computes several O(lambda log lambda) terms that cancel badly in float32. In a fixed-seed BTRS comparison at lambda 1e8, the old path's sample variance inflated around 48%, and the stable log-PMF brought the measured variance much closer to its target, but cost around 32% CPU latency in that pilot. I kept the two claims separate and built a reproducible exact-head benchmark rather than asserting the new math is faster or already accepted.”
