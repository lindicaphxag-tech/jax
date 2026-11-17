# Independent source-transcribed Poisson CPU recheck (2026-11-18)

Upstream: [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228), proposed source head `9428e21c09e6a19cadbf9ed02d6ec78b165f6246`.

**Methodological boundary:** the proposed `_poisson_log_pmf` and `_poisson_rejection` paths were manually transcribed into an installed JAX **0.9.0.1** runtime to make a local CPU comparison with its stock routines. The rejection path was adapted to the installed API with `counts=None`. This was **not a checkout of the upstream PR**, did not execute JAX's full test suite, and does not replace the separate exact-head workflow in this branch. The result is *candidate plausibility evidence*, not official acceptance/performance evidence.

## 201-point deterministic central-region log-PMF grid versus SciPy

Points: integer-rounded `k = mu + t sqrt(mu)`, 201 evenly spaced `t` in [-5,5]. Error is maximum absolute difference in returned log probability.

| Dtype | μ | Stock JAX 0.9.0.1 | Proposed isolated function |
|---|---:|---:|---:|
| f32 | 1e7 | 38.2407979 | 1.7583e-6 |
| f32 | 1e8 | 394.6801419 | 1.9073e-6 |
| f32 | 1e9 | 3606.5622826 | 5.7220e-6 |
| f64 | 1e7 | 5.9605e-8 | 3.9882e-8 |
| f64 | 1e8 | 4.7684e-7 | 4.0275e-7 |
| f64 | 1e9 | 3.8147e-6 | 4.4049e-6 |

All tested values were finite. Float64 at 1e9 does **not** show a universal improvement.

## Paired 32,768-sample rejection-loop check, 9 seeds, float32 CPU

Independent fixed-seed samples with a 100-iteration cap, `counts=None`; compiled and warmed before ABBA timing. The variance estimator pools the 9 seed batches (294,912 samples). Runtime is wall-clock median of the timed invocation samples, excluding JIT compilation.

| μ | Stock variance | Candidate variance | Ideal variance | Candidate/stock median elapsed |
|---|---:|---:|---:|---:|
| 30 | 30.2224 | 29.9867 | 30 | 1.0175 |
| 1e4 | 10030.05 | 9992.68 | 10000 | 1.1852 |
| 1e8 | 145483039.70 | 99914438.72 | 100000000 | 1.0706 |

**Interpretation:** the source-transcribed correction materially reduces the tested f32 large-rate variance defect. It does not establish an across-the-board speedup: the candidate took longer in each of these three local comparisons, with the middle rate especially noisy. No GPU/TPU data, full-PR source build, confidence interval for performance, or maintainer approval is asserted.

## Outstanding upstream gates

- Reviewer @jakevdp requested tests condensed to public `scipy.stats.poisson.logpmf` calls using `_CheckAgainstNumpy`/`_CompileAndCheck`, and benchmark of the rejection sampler rather than the standalone PMF only.
- Keep this performance note **outside** the submitted PR's production diff; report a source-exact reproducible benchmark when the workflow actually runs.
- Current upstream status at the time of this note: PR open, `CHANGES_REQUESTED`; no approval was obtained by this experiment.

The isolated scripts and JSON outputs are included in the independent local evidence bundle. Their source is manually matched and must be rechecked against any future PR revision.
