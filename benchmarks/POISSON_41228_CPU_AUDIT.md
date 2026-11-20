# JAX #41228 — measured CPU log-PMF kernel performance (2026-11-20)

## Scope and reproducibility

The Python 3 CPU timing reported here was actually run in a container with JAX **0.9.0.1**, SciPy **1.17.0**, Linux x86_64. An isolated local benchmark transcribed the structure of the proposed stable `_poisson_log_pmf` from [upstream PR #41228](https://github.com/jax-ml/jax/pull/41228) and compared it with the prior direct formula `-lam + k*log(lam) - lgamma(k+1)`.

**This benchmark was NOT run from a clean checkout of the PR source.** The stand-alone local transcription substitutes an inactive small-count tail helper, and it benchmarks the log-PMF kernel only; it does NOT establish end-to-end `jax.random.poisson` throughput, device portability or a correct CI gate.

To remove that limitation, this **separate research branch** (based on PR head `ec88f105d1f2c8637c08dbaa7bb1072f719b561a`) also ships `benchmarks/poisson_41228_cpu_perf.py`, which imports the exact PR source `jax._src.random.core._poisson_log_pmf`. Run it **from a checkout of this branch with the local checkout installed** to obtain an exact-version result; do not cite the table below as that exact-run receipt.

## Executed stand-alone microbenchmark

Numbers are median JIT **warm invocation** times (milliseconds), five repetitions. Cold compilation is excluded from these medians. Values are reference observations, not promises of cross-machine speed.

| λ | Vector size | Candidate region | Original (ms) | Stable (ms) | Stable/original |
|---:|---:|---|---:|---:|---:|
| 20 | 1 | central | 0.00585 | 0.00589 | 1.01× |
| 20 | 4,096 | central | 0.06457 | 0.15930 | 2.47× |
| 1e5 | 4,096 | central | 0.04311 | 0.05203 | 1.21× |
| 1e8 | 4,096 | central | 0.04427 | 0.06279 | 1.42× |
| 1e8 | 65,536 | central | 0.30622 | 0.32594 | 1.06× |
| 1e8 | 4,096 | tail | 0.06055 | 0.08939 | 1.48× |

The stable calculation produced finite numbers for all of these generated candidates. The direct formula differed from it by **up to 267.45 log-PMF units** for λ=1e8 central cases (JAX float32), demonstrating severe cancellation in the original computation. This internal delta is **not itself a high-precision oracle**: validate the stable function against an independently evaluated reference at large rates.

On λ≤1e5, comparison against SciPy's float64 logpmf for these test vectors gave maximum absolute stable error ≤8.2×10⁻⁷; this is a limited numerical smoke test, not a complete domain or gradient validation.

The timing overhead is real in this microbenchmark: the stable method was approximately 6–147% slower for the larger vector cases tested; however, the endpoint `random.poisson` cost also depends on uniform sampling, acceptance rate, CPU/GPU compilation, and control-flow loops. The maintained PR already includes relevant full function logic and numerical regression coverage.

## What the upstream reviewer actually requested

The JAX review by @jakevdp raised a performance concern for the new log-PMF inside a repeated rejection loop, and asked for a smaller, clearer set of parameterized and public-API tests. The PR currently has a shared `jax.scipy.stats.poisson.logpmf` helper and several vectorized/parameterized tests. Resolve the remaining **human CHANGES_REQUESTED** state against their specific latest comments rather than assuming successful CLA checks imply approval.

### Exact-source follow-up (not executed in this audit)

1. Use a clean checkout of `research/poisson-41228-cpu-benchmark-20261120` and `pip install -e .` with compatible dependencies in a fresh environment.
2. Run `python benchmarks/poisson_41228_cpu_perf.py --repeats 9 --out poisson_41228_cpu_result.json`.
3. Compare both the warm log-PMF cost and **full `jax.random.poisson` runtime** at representative rates and shapes; report compilation separately.
4. Validate the output against high-precision reference values at λ=1e8 and model-relevant extremes; run numerical tests across float32/float64 and CPU/GPU when available.
5. Keep this performance harness opt-in until maintainer review; **do not submit an extra PR or claim merged**.

The official upstream PR is open. This private fork research branch is not attached as a new upstream PR and is not evidence of an accepted human review.
