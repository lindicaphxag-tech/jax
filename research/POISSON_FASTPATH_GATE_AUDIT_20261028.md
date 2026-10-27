# JAX #41228 batch-wide fast-path audit — 2026-10-28

This is a fork-side audit of the batch-level `lax.cond(jnp.all(central), ...)` optimization that was briefly present on the official PR head. It is **not** an official JAX CI result.

## Method

- Environment: installed JAX 0.9.0.1, CPU.
- Extracted the installed `_poisson_rejection` body and changed only the acceptance-PMF call.
- Used the full stable log-PMF and a source-matched transcription of the current PR's central formula and gate.
- Estimated all-central gate activation from 80 independent proposal batches per rate and shape.
- Benchmarked 24 paired warmed calls with alternating order and matching PRNG keys. Host CPU timings are diagnostic and noisy, not release-grade performance numbers.

Reproducer: [`ci/poisson_fastpath_gate_audit_20261028.py`](../ci/poisson_fastpath_gate_audit_20261028.py)  
Raw measurements: [`research/POISSON_FASTPATH_GATE_AUDIT_20261028.json`](./POISSON_FASTPATH_GATE_AUDIT_20261028.json)

## Finding 1: the global gate rarely activates for larger batches

| λ | batch size | all-central hits / 80 |
|---:|---:|---:|
| 1e6 | 256 | 2 |
| 1e6 | 1,024 | 0 |
| 1e6 | 8,192 | 0 |
| 1e6 | 32,768 | 0 |
| 1e8 | 256 | 58 |
| 1e8 | 1,024 | 23 |
| 1e8 | 8,192 | 0 |
| 1e8 | 32,768 | 0 |
| 1e9 | 256 | 70 |
| 1e9 | 1,024 | 53 |
| 1e9 | 8,192 | 2 |
| 1e9 | 32,768 | 0 |

The condition requires **every proposal** in the batch to satisfy the central predicate. One tail proposal sends the entire batch through the general formula, so the fast path becomes improbable at large batch sizes.

## Finding 2: timing is mixed, not a robust general win

For 12 rate/shape pairs, gated/full median latency ratios ranged from 0.708 to 1.197. At `λ=1e8, n=32768`, the ratio was 1.016 (about 1.6% slower); at `λ=1e7, n=32768`, it was 1.197 (about 19.7% slower). Several smaller/high-rate cases improved, but the current evidence does not support a general speedup claim. Paired same-key samples agreed on all tested seeds; this is a deterministic counterfactual check, not a proof of distributional identity over all seeds.

## Decision applied to the official PR

The unvalidated global fast path and its two helper-specific tests were removed from [JAX #41228](https://github.com/jax-ml/jax/pull/41228). The PR again calls the stable log-PMF directly and retains the large-lambda variance regression test. This deliberately favors a narrower correctness patch over an optimization that did not show consistent wins.

A future optimization should be evaluated separately, with exact-head CPU and accelerator benchmarks, gate-hit-rate instrumentation, realistic batch sizes, and no claim of speedup unless the representative workload matrix supports it. A chunked conditional prototype was also tested locally, but results were mixed; it is **not** recommended for upstream integration yet.

## Current status

- Official PR remains open; no merge or new maintainer acceptance is claimed.
- This audit and raw data live on the author's research branch.
- The latest official source head has not been run through full JAX native CI in this environment.

## Baseline versus stable-PMF red/green context

A separate source-matched run on the installed JAX 0.9.0.1 CPU sampler compared the original acceptance expression with the stable helper over five independent 6,000-sample batches per rate (this is also **not** the exact official checkout):

| λ | original variance / λ | stable variance / λ | original median ms | stable median ms |
|---:|---:|---:|---:|---:|
| 1e6 | 1.03426 | 0.99744 | 4.576 | 4.061 |
| 1e8 | 1.43034 | 0.99763 | 2.288 | 2.672 |

This is the actual trade-off: the stable formula removes the severe high-rate variance distortion (the issue's original failure mode) but costs about 16.8% in this particular `lambda=1e8` CPU timing. It is not universally slower either (the `lambda=1e6` run was faster); this is why the PR should claim the correctness fix and disclose the performance limitation rather than attach an unvalidated batch optimization.
