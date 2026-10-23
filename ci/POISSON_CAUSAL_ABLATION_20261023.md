# JAX #41228 / batch admission: three-arm causal interpretation
2026-10-23. **Independent fork research only; not a new upstream contribution or merged PR.**

## Source-matched experiment
Production function: real JAX `_poisson_rejection`, pinned to the reviewed stable #41228 baseline plus the fork-only `_poisson_rejection_log_pmf` certificate experiment. We used three arms compiled in the same JAX CPU backend:

1. **Stable**: reconstruct the current official PR's stable helper call from the production function AST.
2. **Gated**: original experimental batch-certified `lax_control_flow.cond`, executing the central polynomial only if **all** candidates meet the predicate.
3. **Forced fallback**: reconstruct the gated helper from source but statically replace its predicate with `False`, retaining the rest of the helper/control-flow structure.

The experiment uses identical JAX PRNG keys/dtypes/shapes and excludes compilation from paired runtime timings. Four interleaved blocks × 13 repetitions, four workload families, n=8192, three OS runners. Each compiled sampler is compared with the baseline on four additional seeds. All compared output integer arrays matched in the tested workloads.

## Machine results
[Three-OS GitHub Actions success #39953179437](https://github.com/lindicaphxag-tech/jax/actions/runs/39953179437)

Each entry is `median(patched/stable)` across four blocks; **<1** means faster. This is not a speedup guarantee across backends.

| Input rates (8192 samples) | Gated Ubuntu | Forced Ubuntu | Gated Windows | Forced Windows | Gated macOS | Forced macOS |
|---|---:|---:|---:|---:|---:|---:|
| all 1e8 | 0.824 | 1.001 | 0.890 | 1.000 | 0.725 | 0.921 |
| one 10 among 1e8 | 0.825 | 1.001 | 0.911 | 1.018 | 0.862 | 1.002 |
| repeated [10, 1000, 1e6, 1e8] | 0.821 | 0.989 | 0.917 | 1.033 | 0.905 | 1.014 |
| all 1000 | 0.830 | 0.993 | 0.927 | 0.998 | 0.885 | 0.963 |

## Causal falsification and consequences
- In the **one-low** and **all-1000** workloads the batch certificate is always false. The gated sampler nevertheless runs faster in the measured CPU code. Therefore the statement **'the entire observed speed gain comes from skipping tail arithmetic on admitted batches' is false for these workloads**.
- The **forced-fallback** arm generally runs similarly to the baseline, even though it preserves a conditional-control-flow AST. This establishes that a runtime-dependent branch/compiler-program difference is relevant, but does **not prove** which XLA fusion, scheduling, vectorization, or code-generation transformation causes the gain.
- The gate's semantic value on truly central batches remains mathematically meaningful, but latency gains in other workloads must be described as compiler-dependent observations rather than a demonstrated reduction in executed arithmetic.
- Source-lowered StableHLO character counts on this pinned shape: baseline **37,071**, gated **42,809**, forced-fallback **41,599**; the gated graph is larger textually. Character count is not GPU memory, runtime instruction count, or device executable size.
- Initial cold compile measurements on one condition were noisy and sequentially ordered; subsequent conditions hit JIT cache (near-zero recompilation). **No valid comparative compilation-latency or GPU performance claim** has yet been made.
- Identical keys across operating systems are consistency checks, **not independent Monte Carlo seeds**.

## Upstream-go decision
**Do not add the batched gate to official JAX #41228 yet.** It would increase the reviewed implementation by 37 lines / one deleted line for a CPU performance phenomenon whose compiler cause is not characterized. Particularly, claims about `all` admission being responsible for speedups are currently too strong. A maintainer-facing plan should keep the #41228 numerical-correctness fix independently reviewable and attach exact-head statistical/runtime evidence separately.

If proceeding, require a smaller maintainable fast-path design or a maintainer request, plus GPU tests, repeatable compile-time/control-flow attribution and no public `poisson.logpmf` regression. New upstream reviewers must not be told that draft/fork PRs are already merged.
