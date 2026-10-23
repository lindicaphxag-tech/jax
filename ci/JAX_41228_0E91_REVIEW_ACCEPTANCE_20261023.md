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
