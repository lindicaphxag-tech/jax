# JAX #41228: batch-wide central fast-path audit (2026-10-28)

## Question

Does the current official-PR strategy

`lax.cond(jnp.all(central), central_formula, full_stable_formula)`

consistently reduce BTRS acceptance-PMF cost for realistic vector batch sizes?

## Method and limits

- Reproduced the current PR's central predicate and two formulas in a source-matched probe that instruments the installed JAX `_poisson_rejection` method.
- Environment: JAX 0.9.0.1, CPU only. This is **not** a checkout of the exact upstream PR source, not an official JAX CI run, and not evidence for TPU/GPU.
- Gate activation: 80 independent uniform proposal batches per `lambda × batch size`, using the same BTRS proposal expression.
- Timings: 24 paired calls with alternating execution order after warm-up; same PRNG key per stable/gated comparison. The host CPU runtime is noisy; interpret medians as diagnostic, not a release benchmark.
- Exact paired samples agreed in all tested seeds and shapes (minimum agreement 1.0), but this does not prove distributional equivalence for all seeds.

## Key result: global gate activation collapses with batch size

At `lambda=1e8`, all-central batches occurred in 58/80 trials at n=256, 23/80 at n=1024, and 0/80 at n=8192 or 32768. At `lambda=1e9`, rates were 70/80, 53/80, 2/80 and 0/80, respectively. For `lambda=1e6`, there were no all-central batches in 80 trials at n≥1024.

The gate is therefore a batch-level optimization whose fast branch becomes improbable for larger batches: one far-tail proposal makes the entire batch execute the full stable helper.

## Paired CPU timing

| lambda | n | full stable median (ms) | global-gated median (ms) | gated/full |
|---:|---:|---:|---:|---:|
| 1e6 | 1,024 | 0.396 | 0.357 | 0.900 |
| 1e6 | 8,192 | 2.620 | 2.912 | 1.111 |
| 1e6 | 32,768 | 6.303 | 6.565 | 1.042 |
| 1e7 | 1,024 | 0.361 | 0.371 | 1.026 |
| 1e7 | 8,192 | 2.872 | 2.743 | 0.955 |
| 1e7 | 32,768 | 6.473 | 7.745 | 1.197 |
| 1e8 | 1,024 | 0.472 | 0.545 | 1.156 |
| 1e8 | 8,192 | 3.235 | 3.063 | 0.947 |
| 1e8 | 32,768 | 6.960 | 7.070 | 1.016 |
| 1e9 | 1,024 | 0.407 | 0.288 | 0.708 |
| 1e9 | 8,192 | 2.668 | 2.931 | 1.098 |
| 1e9 | 32,768 | 6.192 | 6.623 | 1.070 |

The gate wins in some small/high-rate cases but regresses on several common large batches. This is not a robust general hot-path improvement.

## Recommendation

Do **not** claim the global all-central gate as a generally faster solution. Keep the correctness fix focused on the stable PMF; treat any specialized BTRS acceleration as a separate follow-up that must demonstrate exact-head CPU and accelerator performance across realistic shapes and report the fraction of batches that take the fast branch. A chunked gate was explored separately, but its CPU results were also mixed and do not yet justify upstream integration.

The current official PR's performance question remains open. This report is fork-side evidence only; no JAX maintainer has been notified through this report.

