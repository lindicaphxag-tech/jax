# Experimental batched central-domain admission for Poisson rejection

**Status:** Fork-only research candidate. **Not submitted to official JAX #41228.** Does not constitute a novel probability distribution or a formal proof of GPU speedup.

## Root cause and design
The current upstream JAX #41228 already fixes catastrophic float32 cancellation in the Hoermann transformed rejection acceptance log-PMF via deviance / Stirling correction. However, an elementwise `where` does not by itself guarantee that expensive inactive branches are skipped in XLA CPU code generation.

The experimental `_poisson_rejection_log_pmf` checks a *scalar batch predicate*:

```
central = all((lam >= 1e6) & (k > 10) & (abs((k-lam)/max(k,1)) <= 0.01))
```

When true, `lax_control_flow.cond` lazily executes only the existing central-deviance polynomial and the k>10 Stirling correction. When false, it calls the unchanged stable general-purpose helper. Public `jax.scipy.stats.poisson.logpmf` remains unchanged. This is *domain-certified computational specialization*, not a new sampling distribution.

### Equivalence argument within the predicate
Let `t=(k-lam)/k`. On every admitted element, k>10 and |t|<=0.01:
- `safe_k=max(k,1)=k` and the central `small_delta` selector is true;
- `lam/k=1-t >= 0.99>0.5`, so the far-tail selector is false;
- `k!=0`, `lam>0`, and k>10 make the zero-mass, invalid-rate and small-k Stirling branches inactive;
- therefore the returned expression is the **same central polynomial and large-k Stirling expression** as the established stable helper (modulo compiler-level floating-point reassociation).
- When the batch predicate is false the existing helper is called, so no new acceptance formula is introduced.

The degree-8 polynomial is *already an approximation* of the central deviance; the elementary Taylor truncation remainder satisfies `|R8| <= k|t|^9/(9(1-|t|))`. This does NOT bound all float32 rounding, prove universal bitwise equivalence under other compilers/backends, or prove exact sampling for every parameter.

## Experimental controls
- Production `_poisson_rejection` in the exact checkout is the *intervention*.
- An AST transformation of exactly the same source reverts only the replacement score call to the already-reviewed `_poisson_log_pmf` to form the stable *reference*.
- All comparisons use identical keys, arrays, dtypes and stopping bounds; JIT compile is excluded from execution timing.
- Fixed and mixed rates, multiple keys, selected explicit central/fallback inputs and 15 interleaved pairs are checked.
- Independently, 5 timing blocks × 21 interleaved runs per configuration tested repeatability on CPU on three operating systems.
- GPU timing, JIT compilation latency and generalized workloads have not been established.

## Initial three-platform study

**Actions #39948744184:** https://github.com/lindicaphxag-tech/jax/actions/runs/39948744184 (Ubuntu, Windows, macOS: success).
Patched/baseline median execution-time ratios for 8192 samples and 15 pairs:

| lambda | Ubuntu | Windows | macOS |
|---:|---:|---:|---:|
| 10 | 0.933 | 0.825 | 0.733 |
| 1e3 | 0.944 | 0.845 | 0.845 |
| 1e6 | 0.951 | 0.854 | 0.748 |
| 1e8 | 0.937 | 0.851 | 0.690 |

All 8 scalar-rate shape configurations × 6 fixed seeds had zero sample-array mismatches on each OS. Native Linux Poisson tests: 30 log-PMF cases passed, 1 x64-only skipped, and 1 high-lambda variance test passed.

## Independent repeatability and heterogeneous input validation

**Actions #39949171147:** https://github.com/lindicaphxag-tech/jax/actions/runs/39949171147 (all three OS jobs: success). For 8192 samples, median within-run patched/baseline ratio over 5 blocks, 21 interleaved pairs per block:

| lambda | Ubuntu | Windows | macOS |
|---:|---:|---:|---:|
| 10 | 0.863 | 0.852 | 0.756 |
| 1e3 | 0.868 | 0.862 | 0.963 |
| 1e6 | 0.876 | 0.862 | 0.745 |
| 1e8 | 0.871 | 0.871 | 0.760 |

All 6 additional mixed-rate seeds had zero output mismatches (CPU). Selected score inputs covering central, fallback, negative proposals and mixed rates agreed within float32 tolerance; a mixed-rate score had a small compiler-level difference of 1.5259e-5. Individual macOS timing blocks varied, including a >1 ratio at 1e8; no universal speedup claim follows.

**Stronger branch-coverage audit:** a subsequent test revision replaces a mislabeled “one far” example with a genuinely far candidate `k=1e8+3e6` and asserts that the central case is admitted while the three distinct fallback cases are rejected. Its exact-head CI status must be checked before claiming success for that revision.

## Merge gates and remaining questions
1. Check the new explicit branch-coverage CI and x64-enabled boundary tests.
2. Benchmark compiled program size/compile latency and at least one GPU where available.
3. Understand why even lambda 10 and 1000 (where the new predicate is deterministically false) sometimes appear faster. The actual improvement may involve XLA code-generation/fusion changes rather than mathematically fewer operations alone.
4. Re-run under newer compilers, heterogeneous rates and seeds not used to design the candidate.
5. Keep the existing upstream correction separate; do not enlarge its currently reviewed scope without maintainer feedback.

**Contribution state:** Fork Draft PR #16, upstream #41228 still Open. Validated performance improvement in measured CPU cases is an engineering result; algorithmic originality and upstream acceptance are unproven.
