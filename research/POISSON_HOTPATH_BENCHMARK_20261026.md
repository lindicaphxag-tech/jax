# Poisson rejection hot-path: independent CPU benchmark (2026-10-26)

Official PR: [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228). Reproduction script on this branch: [benchmarks/poisson_rejection_logpmf_hotpath_20261026.py](../benchmarks/poisson_rejection_logpmf_hotpath_20261026.py).

## What was tested

An independent local **JAX 0.9.0.1 CPU** test used its actual BTRS sampler method body, changing *only* the log-PMF expression to the candidate stable formula transcribed from #41228. Each float32 rate used 20,000 samples, seeded PRNG keys, seven synchronized post-compilation timings (reported median).

| lambda | original ms | stable ms | slowdown | original var/lambda | stable var/lambda |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 4.626 | 10.173 | 2.20x | 1.000 | 1.000 |
| 100 | 7.185 | 8.074 | 1.12x | 0.991 | 0.991 |
| 1000 | 3.210 | 3.793 | 1.18x | 0.996 | 0.996 |
| 100000 | 3.456 | 3.810 | 1.10x | 0.993 | 0.993 |
| 100000000 | 3.683 | 7.169 | 1.95x | 1.469 | 0.993 |

The variance ratio is from the final deterministic draw, **not** an uncertainty estimate across many seeds. Candidate improves large-parameter variance fidelity in this specific CPU case but incurs measurable throughput regressions. 

## Scope and next gate

These data are **not** latest JAX main, not a full exact-head PR benchmark, not a GPU measurement, and do not establish cross-platform timing ratios. The separate benchmark script derives a baseline from the current candidate checkout's actual BTRS source by restoring only the original acceptance log-PMF; it has not yet produced an official PR-head CI result.

Before approving: execute the exact-head benchmark on a supported version with multiple seeds, include compilation cost, and assess whether a narrower sampler-specific stable path can reduce overhead without duplicating conflicting public log-PMF semantics. Do not claim a performance win without measurements.
