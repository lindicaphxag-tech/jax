# JAX Poisson #41228 — independent CPU acceptance evidence

**Source revision**: [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228), SHA `9428e21c09e6a19cadbf9ed02d6ec78b165f6246`.

This **separate fork branch** records benchmark evidence without expanding the official pull request's production/test diff. It must not be interpreted as upstream approval or a completed exact-head CI until the action run actually finishes successfully.

## Preliminary independent CPU finding (2026-11-18)

The initial experiment transcribed the proposed log-PMF into installed JAX 0.9.0.1, not the official PR checkout. Nine seeded batches of 65,536 samples each, float32:

| λ | Old sample variance | New sample variance | Target | New/old paired latency median (95% seed bootstrap) |
|---:|---:|---:|---:|---:|
| 30 | 30.073 | 30.072 | 30 | 0.946 (0.939–1.046) |
| 10,000 | 10,031.83 | 10,029.34 | 10,000 | 0.980 (0.891–1.014) |
| 100,000,000 | 145,830,285 | 100,314,980 | 100,000,000 | 1.060 (0.935–1.236) |

At λ=1e8, the old numerical formula inflated sample variance by +45.83%, whereas the proposal yielded +0.315% error in this CPU experiment. This addresses the original numerical issue, **not** a performance-speedup claim: every latency uncertainty interval overlaps 1, and separate single-seed measurements yielded different ordering.

## Exact-head runnable benchmark

```bash
python -m pip install -e .
python research/poisson_41228_benchmark.py --n 65536 --seeds 9
```

The script uses the **actual PR-head** `jax._src.random.core._poisson_rejection` for the new implementation and a source-equivalent pre-PR sampler for the old implementation. It warms both, performs ABBA timing, computes pooled variance across multiple RNG seeds, and writes `poisson_41228_exact_head_results.json`.

- The default run is **CPU float32 only**. It does not replace upstream tests or establish CUDA/TPU performance.
- Before relying on results, verify that the branch differs from the PR SHA only by files under `research/` and the independent workflow.
- The benchmark caps rejection iterations at 100 for efficient test runs. Both variants use the same cap.
- Hosted runners are shared and may suffer scheduling noise. The paired ratio/bootstrap is descriptive, not a conclusive performance guarantee.

## Pending acceptance

Run the exact-head workflow; inspect the resulting raw JSON artifact and the job logs. Only then share exact-head measurements with the upstream reviewer. Keep the official JAX PR focused on test consolidation and the numeric fix.
