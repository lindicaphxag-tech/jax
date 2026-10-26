# Integer-centered BTRS Poisson: mechanism, CPU controls and unresolved proof

**2026-10-26. Fork-only candidate, based on official JAX Poisson PR #41228 head `280b8edf`.** The official PR was **not** modified by this research branch. This is a proof-of-concept, **not** a validated exact Poisson sampler or a new external merged PR.

## Invariant and local candidate

The exact real-arithmetic identity is

```
floor(lambda + proposal + 0.43)
= floor(lambda) + floor(proposal + (lambda - floor(lambda)) + 0.43).
```

JAX's previous `float32` evaluation of the *left* side loses integer residues when `lambda >= 2**24`, because adding the large mean rounds the entire candidate to a sparse float32 lattice before an integer conversion. The branch's research implementation evaluates the *right* side, rounds the small proposal first, then sums the integer base and offset in the **output integer dtype**. For means below `2**24`, the implementation deliberately preserves the old float-proposal path and converts to integers only after the legacy floor.

**Important numerical limitation:** the branch now retains the exact integer-centered `local_offset` as an optional `exact_delta` input to `_poisson_log_pmf` for the central deviance, instead of recomputing `float32(k) - lambda` after `k` has been rounded to a sparse float lattice. The function still uses a float32-rounded `k` in the normalizer and far-tail factors, and the full BTRS law, exceptional inputs and range/overflow remain **unproved**. Restoring integer support and central log-likelihood precision are necessary but not sufficient to certify a statistically exact Poisson distribution.

## Native released JAX source-transplant pilots

CPU JAX 0.9.0.1; only `_poisson_rejection` method body modified in an in-memory code copy; candidate log-PMF operations identical to the pending #41228 version. The installed library was **not modified**.

3 fixed seeds × 6,000 samples each (18,000 per mean), retaining identical keys for baseline and candidate:

| mean | float32 ULP | Stable log-PMF only: available integer residues | Centered integer proposal: available residues | Centered variance / mean |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 1 | 1/1 | 1/1 | 0.9901 |
| 1e3 | 1 | 1/1 | 1/1 | 1.0082 |
| 1e6 | 1 | 1/1 | 1/1 | 1.0028 |
| 2e7 | 2 | 1/2 | 2/2 | 1.0029 |
| 4e7 | 4 | 1/4 | 4/4 | 1.0030 |
| 7e7 | 8 | 1/8 | 8/8 | 1.0030 |
| 1e8 | 8 | 1/8 | 8/8 | 1.0032 |
| 1.5e8 | 16 | 1/16 | 16/16 | 1.0035 |

**Legacy same-key preservation:** With a guarded `lambda < 2**24` path, exact per-sample equality against the unmodified stable-log-PMF sampler was observed for 15,000 draws each at `lambda=10,10.25,20,20.5,100,1000,1e5,1e6,1e7`. This is an empirical compatibility check on one CPU runtime, not a cross-device bitwise-guarantee.

## Disjoint holdout of 100,000 samples per mean

10 different seed values `211,223,227,229,233,239,241,251,257,263`, each with 10,000 draws (independent from the 3-seed pilot). For the integer-centered candidate:

| mean | recovered residue classes | variance/mean | Pearson chi-square uniform-residue p |
| ---: | ---: | ---: | ---: |
| 2e7 | 2/2 | 0.99223 | 0.879 |
| 4e7 | 4/4 | 0.99234 | 0.713 |
| 7e7 | 8/8 | 0.99233 | 0.197 |
| 1e8 | 8/8 | 0.99231 | 0.241 |
| 1.5e8 | 16/16 | 0.99221 | 0.148 |

Earlier, disjoint 6-seed × 8,000-draw controls had `p=0.0399` for residue uniformity at `lambda=1e8`. The larger independent holdout returned `p=0.241`; **do not erase the earlier negative-looking result or claim formal uniformity from a nonsignificant p-value**. These checks only provide finite-sample diagnostics for residues and moments, and no multiple-parameter goodness-of-fit correction was attempted.

Native CPU interleaved latency readings fluctuate across rates and seeds; a stable speedup or bounded slowdown has **not** been established. This work is also not a GPU/XLA compiler regression pass or official JAX current-head test.

## Acceptance gates before any official PR

1. Validate exact-head source and full JAX native tests on CPU, then GPU and x64/int64 combinations.
2. Eliminate acceptance-likelihood rounding on centered integer candidates, or prove a compatible error/correction bound; no unproved dithering.
3. Prove proposal/rejection correctness, including negative and overflow candidate handling and rare-tail support.
4. Compare full distribution laws under precommitted metrics/seed splits; test JIT, vmap, broadcasting and mixed-rate shapes.
5. Obtain maintainer guidance on whether this warrants a separate high-rate exact-sampling PR rather than expanding #41228. Current official issue creation is blocked by GitHub integration 403.

### Authorship / status

- **Official JAX:** [#41228](https://github.com/jax-ml/jax/pull/41228) is Open; this branch is **not** its head.
- **Fork research branch:** `research/poisson-centered-integer-proposal-20261026`.
- **Production experiment:** [jax/_src/random/core.py](../jax/_src/random/core.py) with guarded integer-centered proposal.
- **Regression test:** [tests/random_lax_test.py](../tests/random_lax_test.py), requires all float32 integer residues to appear at selected large means.
- **Related upstream issue draft:** [JAX_POISSON_INTEGER_SUPPORT_UPSTREAM_ISSUE_DRAFT.md](JAX_POISSON_INTEGER_SUPPORT_UPSTREAM_ISSUE_DRAFT.md).


## Additional integer-offset log-PMF precision gate (independent CPU experiment)

On local JAX 0.9.0.1 CPU, compare 2,000 independently generated **integer** count values per mean within `mean ± 5*sqrt(mean)` against `scipy.stats.poisson.logpmf` in float64. The historical `float32(k)` calculation loses the low residue bits; the optional `exact_delta` parameter preserves the signed integer offset before evaluating central deviance:

| mean | Old, float32-rounded-k log-PMF max abs error | Centered exact-delta max abs error |
| ---: | ---: | ---: |
| 2e7 | 0.001117 | 0.00000334 |
| 4e7 | 0.001582 | 0.00000310 |
| 7e7 | 0.002369 | 0.00000358 |
| 1e8 | 0.001992 | 0.00000286 |
| 1.5e8 | 0.003265 | 0.00000334 |

The test was a **source-equivalent local formula transposition**, not a full native-head JAX test. A new in-repository regression `testPoissonIntegerOffsetLogPmfAccuracy` compares representative integer counts against the SciPy oracle with `atol=2e-4`.

A separate 5-seed × 10,000-draw CPU sampling check of the integer-centered proposal **with** this improved acceptance deviance still occupied all float32 spacing classes, with per-rate variance/mean 0.9976–0.9977 at means from 2e7 to 1.5e8. For `lambda=1e8`, the 8-way residue chi-square p-value was about 0.337. These are not proof of an exact distribution or of a particular performance bound.

The optional `exact_delta` path is used **only within this research branch's BTRS sampler**; public SciPy-shaped `poisson.logpmf(k,mu)` continues to call the unchanged two-argument formulation. This division limits public API risk, though cross-device/compiler and complete numerical acceptance correctness still require independent testing.


## 2026-10-26 falsification and mitigation: signed int32 overflow

The earlier *unguarded* integer-centered prototype is **unsafe** near the positive int32 maximum. On a source-matched **JAX 0.9.0.1 CPU** test with 4,096 draws at `lambda = 2**31 - 8192 = 2,147,475,456`, it returned both large positive and **negative** counts (`min=-2,147,483,624`), and the observed sample variance/rate was approximately **2.0788e9**. The released source-matched stable-log-PMF comparator also has a serious pre-existing distributional defect in this range (variance/rate ~0.4299) but did **not** exhibit this prototype's catastrophic signed-wrap variance. This invalidates any claim that the prototype is correct for all int32 rates.

**New conservative guard in fork source:** use integer-centering only when `2**24 <= lambda < 2**30` **and** `abs(local_offset) <= 2**29`. Compute `safe_base` and `safe_offset` using this predicate **before integer casts/addition**. Within this selected region `0 <= base < 2**30` and `|offset| <= 2**29`, so their sum lies between `-2**29` and `(3/2)*2**30`, strictly inside the signed int32 range. Outside, preserve the existing stable-log-PMF sampler's historical float-candidate path; this is a **scope guard, not a fix for the original extreme-rate pathology**.

**Independent local guarded-vs-stable CPU control**, 4,096 matched-key samples per rate, reproducing the same conditional calculation:
- `lambda=1e3,1e6,1e7,2**24-8`: bit-for-bit equal arrays (low-rate legacy behavior preserved in tested configuration).
- `lambda=2**24,2e7,1e8,2**28`: centered proposal exercises all sampled float32 integer residue classes, variance/rate ~1.0122 for these fixed keys (not a distributional proof).
- `lambda=2**30`: unchanged source-matched sampler samples in the tested key, variance/rate ~1.0120 but only 2/128 residue classes.
- `lambda=2**31-8192`: guarded prototype returns **zero negative counts** in 4,096 tested draws, variance/rate ~0.4207, compared to ~0.4299 for existing source-matched stable sampler. Arrays need **not** remain identical in this near-int32-overflow region due altered fused numerical expressions; *both* have inadequate distribution fidelity.

A new fork-only native regression `testPoissonCenteredInt32RangeGuard` checks the negative-wrap witness. The exact source-transplant reproducer is also available for download from the accompanying research chat; this branch has **not** passed official latest-main native CI. For rates near and above the signed-int32 representability boundary, a correct solution needs explicit distribution/API range semantics; fallback is only damage containment.

**Research integrity update:** all earlier holdout results in the verified 2e7–1.5e8 range remain relevant, but the safe domain of the implementation has been narrowed. Do not describe it as a universally exact integer-valued sampler.
