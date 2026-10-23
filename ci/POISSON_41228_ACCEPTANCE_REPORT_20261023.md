# JAX #41228: paired performance and numerical correctness acceptance note

**Date:** 2026-10-23  
**Official upstream:** https://github.com/jax-ml/jax/pull/41228 (open at check time)  
**Exact fork-only CI:** https://github.com/lindicaphxag-tech/jax/actions/runs/39948824118 (**Success**)  
**Benchmark implementation:** [ci/poisson_41228_paired_perf.py](../ci/poisson_41228_paired_perf.py)

## Reviewer concern addressed

The JAX reviewer specifically requested a benchmark for the added transformed-rejection `log_pmf` hot-path work. This study compares the *unchanged pre-fix acceptance expression* (`-lam + k*log(lam) - lgamma(k + 1)`) against the **actual** current PR implementation `jax._src.random.core._poisson_rejection`. The test uses the same RNG, schedule, shape and dtype for paired runs. No production code is changed on the official PR.

## Native CI result (one GitHub Actions CPU worker)

JAX `0.12.0.dev20261023+2aa5428`, CPU. JIT is warmed up, timings are synchronized with `block_until_ready`, order alternates AB/BA, and each case uses 10 paired blocks. Ratio less than 1 means the PR version was faster in **this** environment.

| lambda | batch | median new/old time | old variance error | new variance error |
|---:|---:|---:|---:|---:|
| 1e4 | 8,192 | 0.5820 | 1.180% | 1.212% |
| 1e4 | 32,768 | 0.4353 | 0.145% | 0.084% |
| 1e8 | 8,192 | 0.9110 | 67.655% | 1.060% |
| 1e8 | 32,768 | 0.6886 | 72.682% | 0.364% |

The full individual pair times and variance outputs are printed in the [Actions job logs](https://github.com/lindicaphxag-tech/jax/actions/runs/39948824118). The job passed install, source benchmark, and checkout without errors.

## Interpretation limits (important)

- These numbers are **not** a universal speedup claim; GPU, TPU, alternate CPU architectures, compiler cache state and other shapes are not covered.
- This is a newly compiled, paired reference reconstruction of the old rejection loop versus the live corrected internal sampler. It is not a fully isolated upstream release A/B; an exact historical Git checkout comparison would strengthen attribution.
- Earlier local JAX 0.9.0.1 CPU tests of the transcribed candidate saw overhead (roughly 1.1–1.34x), while this **source-matched** CI showed speedups. This environment difference must be disclosed rather than hidden.
- The one-seed sample variance is **not** a p-value; statistical acceptance requires independent seeds and larger runs. The regression tests already separately cover public API and sampler correctness.
- Performance measurement is noisy and short (~10 pairs/case). Avoid performance guarantees or changing the hot path purely from these four CPU conditions.

## Highest-value review closure

1. Ask which benchmark shapes, dtypes and platforms the maintainer considers representative; extend the **same** paired protocol if needed.
2. Confirm the now-shared Poisson log-PMF, public-API parameterization, and trimmed sampler variance test meet JAX test-maintenance expectations.
3. Request a fresh human review of the current 4-file, single-feature PR without introducing separate algorithm ideas.
4. Check official PR's exact-head required checks. Fork CI success does not imply official CI green or reviewer approval.

**Contribution claim:** A reproducible, source-matched benchmark answering a concrete review objection while honestly preserving its external validity limits. Not a new original numerical algorithm or a merged upstream change.
