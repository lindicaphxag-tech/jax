# JAX #41228 current-head acceptance evidence — 2026-10-23

**Target official PR:** https://github.com/jax-ml/jax/pull/41228  
**Exact upstream head tested:** `0e91f36697e1239fa5f054314d3b01878842cc44`  
**Fork-only validation PR:** https://github.com/lindicaphxag-tech/jax/pull/18  
**Successful 3-OS GitHub Actions:** https://github.com/lindicaphxag-tech/jax/actions/runs/39954368575

This is validation of the **revised 10th commit**, which consolidated the public Poisson numerical tests by editing `tests/scipy_stats_test.py` (+39/−59 compared with prior head `80e1a319`). It is not a new proposed upstream implementation and it is not proof of a new probability distribution.

## Native acceptance checks
- Ubuntu: **95 passed, 2 skipped** in `tests/scipy_stats_test.py tests/random_lax_test.py -k 'Poisson'` (all non-Poisson tests deselected).
- Ubuntu with `JAX_ENABLE_X64=true`: **28 passed** in `tests/scipy_stats_test.py -k 'PoissonLogPmf'`.
- Ubuntu, Windows, macOS: all matrix jobs success; source syntax and Ruff checks passed on actual PR source.
- CPU matched-source sampler benchmark: production patched `_poisson_rejection` vs restored legacy expression/hoisted `log_lam` in an AST copy of exactly the same function.
- 12 held-out random seed blocks × 8,192 samples each; at lambda=1e8, empirical variance/lambda old **1.753081056** vs patched **1.006821165**, theoretical 1.0.
- Selected public log-PMF vs SciPy float64 reference max absolute difference **7.152557373046875e-7**.

## Matched-source 8,192-sample CPU runtime ratio (patched / original; 15 interleaved timing pairs)

| lambda | Ubuntu | Windows | macOS |
| ---: | ---: | ---: | ---: |
| 10 | 0.5972 | 0.5934 | 0.8101 |
| 1000 | 0.6041 | 0.5876 | 0.8951 |
| 1e8 | 0.9729 | 0.9162 | **1.3332** |

In this specific macOS run at large lambda, the corrected sampler was slower (~33.3% by median ratio). Runtime ratios from shared runners have varied between runs, and no universal CPU speedup, GPU neutrality, or performance-regression-free claim is warranted. Sampling correctness is supported more strongly than performance.

## Merge gates
1. Upstream `jakevdp` review still includes an unresolved benchmark concern and no newer approval; the earlier `CHANGES_REQUESTED` review remains in the record. Do not equate mergeable=true with ready-to-merge.
2. The consolidated tests and source-matched performance evidence are ready to share. Posting a comment to the official JAX repository via the connected GitHub integration previously returned HTTP 403; do not claim maintainer has received this artifact.
3. Avoid merging the unrelated fork-only experimental batched conditional branch into official #41228 without explicit reviewer interest. The exploratory control-flow speedups were shown by a 3-arm causal study to arise even when the batch predicate is false; mechanism not fully attributed. Details in fork Draft #17.
4. Future progress must be recorded as actual reviewer acceptance / CI, and finally merged state; this report does not increment external merged count.

### Distinguish scope
These tests establish current-head reproducibility of the claimed numerical fix under the tested JAX CPU environments and selected seeds/parameters, not full all-platform correctness, a formal finite precision bound, or whole-workload speedup.


## 2026-10-24 reviewer handoff — current official head `3c2f800a`

**Fresh GitHub verification:** official [JAX #41228](https://github.com/jax-ml/jax/pull/41228) remains Open and GitHub reports `mergeable=true`, but jakevdp's most recent review remains `CHANGES_REQUESTED` with unresolved performance concern. The **current source blob** `jax/_src/random/core.py @ 97ad6ce9862336279b323264cb34c5406e4fdb21` is **identical** in the benchmarked upstream head `0e91f366` and latest official head `3c2f800a`; later revision simplified public numerical tests, not the performance path.

**Current test revision:** [newer current-head native Ubuntu/macOS/Windows receipt #39973265481](https://github.com/lindicaphxag-tech/jax/actions/runs/39973265481): 88 scoped Poisson tests passed on EACH OS; Ubuntu additionally passed 24 tests with x64 enabled. Test refactoring implements maintainer-suggested `jtu.sample_product`, vector input, and `_CheckAgainstNumpy` / `_CompileAndCheck` where SciPy supplies an accurate oracle. At float32-maximum scale SciPy itself exhibits catastrophic cancellation in its returned large-λ log-PMF, so a separately justified asymptotic reference is required rather than pretending an inaccurate SciPy value is ground truth.

**Upstream communications limitation:** both (a) replying to the maintainer's numeric inline performance comment `#discussion_r4329673957` and (b) posting a normal top-level PR comment through this integration returned **403 Resource not accessible by integration** on October 24. **Neither comment was published.** Do not silently count them as a reviewer follow-up. Review request attempts in a separate HF repository also returned 403, suggesting restricted upstream write scope, but root cause of permissions is not yet verified.

### Concise official reviewer reply, prepared but NOT SENT

> Following up on the benchmark concern: I benchmarked the current production `_poisson_rejection` source against a source-identical pre-fix version, restoring only the original log-PMF expression. Both paths used identical keys, warmed CPU JITs and 15 alternated repetitions per configuration. At λ=10 / 1e3 / 1e8 (8192 draws), patched/original median ratios were Ubuntu 0.60 / 0.60 / 0.97; Windows 0.59 / 0.59 / 0.92; macOS 0.81 / 0.90 / **1.33**. The macOS large-λ regression is real for that run; I am **not** claiming universal speedup. The 12-seed, 98,304-draw λ=1e8 holdout measured original variance/λ **1.7531**, patched **1.0068** (theory 1). The latest test simplifications passed [88 Poisson tests on each of 3 OSs + 24 x64](https://github.com/lindicaphxag-tech/jax/actions/runs/39973265481). The benchmarked sampler source file is byte-identical to official `3c2f800a`. [Benchmark source](https://github.com/lindicaphxag-tech/jax/blob/ci/poisson-41228-0e91-review-gate-20261023/ci/poisson_native_sampler_review_benchmark.py) · [logs](https://github.com/lindicaphxag-tech/jax/actions/runs/39954368575). Could you advise whether this measured correctness/performance tradeoff is acceptable, or whether a particular device/benchmark is required to approve?

**Do not** rewrite or rebase the official PR just to generate new statuses while this performance review remains open. The real three-way merge audit [#39985192364](https://github.com/lindicaphxag-tech/jax/actions/runs/39985192364) already found no source conflict; no speculative production speed path has been introduced.
