# PR #41228: full rejection-sampling CPU evidence, including 12 independent keys

**External upstream PR:** https://github.com/jax-ml/jax/pull/41228  
**Audit date:** 2026-11-21 (Asia/Singapore)  
**Branch base:** `735b5b1ab8cae36f6a269efa31a75f8107bea67c` (the latest checked upstream PR head at the time of branch creation).  
**Status:** read-only numerical/performance research evidence. This is **not** an additional upstream PR, is **not** a passing upstream test, and does **not** change the production source.

## Why the reviewer requested this

The human maintainer's October 20 review noted that the stable Poisson log-PMF is evaluated repeatedly inside the rejection sampling loop, making performance important. The PR includes an opt-in `benchmarks/random_benchmark.py` microbenchmark, but the code must still be measured under representative CPU/GPU configurations and compared against the old implementation. The current human review state includes `CHANGES_REQUESTED`.

## Exactly what was executed

A reproducible local Linux CPU experiment used:

- Python 3.13, `jax==0.9.0.1`, JAX's **installed historical** `jax._src.random._poisson_rejection` rejection sampler, and `jax.random.key(seed)`.
- The same installed sampler with its old direct `-lam + k*log(lam) - lgamma(k+1)` expression replaced by a **standalone transcription** of the stable `_poisson_log_pmf` logic visible in PR #41228. A file-level `inspect.getsource` assertion anchored the replacement point.
- JIT compilation and initial warm call excluded from timed steady-state samples; `block_until_ready` included in each timed call. Same rate, shape, dtype, `max_iters=400` and RNG keys for both arms.
- Float32 Poisson rates `1e4` and `1e8`; 12 independent integer seeds, `0..11`; each seed generates 32,768 output samples **per implementation** at each rate. The theoretical Poisson variance is equal to its rate.
- A complementary 9-repeat, same-key JIT warm-call test at 8,192 and 32,768 samples was also executed. The original JSON captures all per-case first-call and timed values.

**Crucial scope caveat:** This is an instrumented **old JAX 0.9.0.1** compiled rejection loop with a transcribed new formula, NOT an exact checkout of PR #41228, NOT the public `jax.random.poisson` wrapper and NOT GPU, multi-device, or project CI evidence. The code was actually run, but its timing cannot be assumed equal to the latest JAX branch.

## Measured multi-seed statistics (12 × 32,768 output samples per arm)

| Rate | Historical direct: mean seed variance | Patch-equivalent stable: mean seed variance | Theoretical variance | Stable/old variance ratio |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | **10,016.30** | **10,004.59** | 10,000 | 0.99883 |
| 100,000,000 | **146,496,409.90** | **100,140,079.57** | 100,000,000 | 0.68357 |

At rate 1e8 the old float32 log-PMF produces a substantial variance inflation (~1.465× nominal); the stable formula removes it within the sampling uncertainty of this probe. At 1e4, both match the expected variance.

The 12-key test reported **median kernel wall-time** (steady state, different keys) of 8.214 versus 8.293 ms at rate 1e4, and 7.925 versus 8.445 ms at rate 1e8 (old versus stable). Thus the observed stable/old ratios were about 1.010× and 1.066× in that **particular** test order and CPU environment. In the separate **same-key, 9-repeat warm-call** trial at rate 1e8, stable/old was about 1.68× with 8,192 outputs and 1.27× with 32,768 outputs; this discrepancy reflects sensitivity to CPU/caching and measurement protocol. **Do not select one latency number as a final regression bound.** A controlled randomized timing order with robust host/GPU repeat statistics is still required.

The new formula has a real numerical-correctness benefit; whether the performance is merge-acceptable requires **exact-source** benchmarking and maintainer judgment.

## Evidence artifact with executed sources

A conversation-delivered ZIP was created, passed Python ZIP CRC validation and SHA-256 audited. Its exact name and SHA-256:

```text
JAX_41228_完整泊松采样_12种子实测20261121.zip
SHA-256: 27a3599e7a81fa5b534f7a5119bbaf0329b997be4a3585dab9f30e49470fe651
```

Contains:
- `jax_poisson_full_rejection_probe.py` executed source (SHA-256 `2a427a66950d0759bc89b2a8816f1f6ee46ee6ebd3fee88024b0101385bc5768`);
- `jax_poisson_multiseed_probe.py` executed source (SHA-256 `36f921906e4d8cf9f4e7d4db020491b1c0eb3b0b4bbbaf655098fcfd7df3422f`);
- `JAX_PR_41228_full_sampler_comparison_20261121.json` (SHA-256 `313e3812f53a2accfef9d39770ed041e80750eed5232733c76a675c004ea48f0`);
- `JAX_PR_41228_full_sampler_12_seeds_20261121.json` (SHA-256 `0bc106523dc680c5c2164d67d68e51742243773fc3c92a8960b89256cc75c805`);
- an explanatory README and file checksum manifest.

**The ZIP exists as a user-facing conversation attachment, not a binary GitHub upload.** This Markdown file records its integrity, numerical evidence and limitations and is based on the current PR head.

## Concrete final reviewer acceptance gates

1. **Exact PR checkout:** install current `jax`/jaxlib from that source, run the project's benchmark entrypoints `poisson_rejection_standard_rate_float32` and `poisson_rejection_large_rate_float32` with warm-up and completed-device timing.
2. **Control:** compare to historical `jax` from identical environment, dtype, shape and device; avoid including compilation in runtime and randomize measurement ordering.
3. **Correctness:** keep `jax.scipy.stats.poisson.logpmf` parity, large-parameter variance and the source-linked probability test as distinct regression gates, not a dozen redundant cases.
4. **Review:** confirm remaining comments and branch protection after rerunning targeted upstream tests. A previously requested change is not automatically approved merely because a branch becomes `mergeable=true`.

No maintainer review message was posted or claimed. The existing upstream PR remains open; do not infer author acceptance or external merge from this evidence.
