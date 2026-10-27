# High-precision Poisson acceptance oracle for JAX #41228 (2026-10-27)

**Fork-only numerical verification**, branched from the precise external upstream PR [jax-ml/jax#41228](https://github.com/jax-ml/jax/pull/41228), head `280b8edff3ecf6397d5906f71385fe50bb3e9b75`. This does **not** update the official PR, start official CI or remove the requested performance review.

## Scientific reason for an independent reference

For huge mean lambda and integer count k approximately lambda, SciPy's standard double-precision expression `-lambda + k*log(lambda) - gammaln(k+1)` can suffer catastrophic cancellation. In this limit, using ordinary float64 `scipy.stats.poisson.logpmf` as an **absolute** oracle is misleading; a correct stable implementation can differ from SciPy in the least significant decimal places of the log probability without being wrong.

Compute the mathematical log-PMF using 100 decimal-digit `mpmath` on **exactly the float32/float64-representable integer rate and count** that JAX receives. The distinction matters because float32 rounds 1e12 to 999999995904 and has a large count lattice spacing near that rate. There are 9 rates (`10`, `100`, `1e3`, `1e5`, `1e7`, `1e8`, `1e9`, `1e10`, `1e12`), 7 count offsets measured in standard deviations, and two dtypes = **126 test cases**.

## Measured CPU results (JAX 0.9.0.1 source-matched transposition, NOT exact PR checkout)

| Dtype | Cases | Maximum absolute error vs 100-digit truth, stable JAX formula | Maximum absolute error vs 100-digit truth, SciPy 64-bit |
|---|---:|---:|---:|
| float32 | 63 | 0.0001160714 | 0.0039803391 |
| float64 | 63 | 9.40803e-12 | 0.0037784920 |

At exact represented rate/count `999999995904` (float32), 100-digit truth for log-PMF is `-14.73444908912103`; the stable JAX result is `-14.73444938659668` (absolute error `2.975e-7`), while the ordinary SciPy result is `-14.73046875` (absolute error `0.0039803`). For a distinct float64 tail case at rate `1e12`, k `999993000000`, stable JAX returns `-39.23450275802353` and the 100-digit oracle agrees to output precision, while SciPy returns `-39.23828125` (error `0.00377849`).

**This is not a blanket accuracy superiority statement.** Across the grid, float32 JAX's maximum error is `0.000116` at `lambda=1e5, k=93675`, where the precision budget of an f32 implementation differs from f64. Specific SciPy large-rate errors reflect cancellation in the public method on the tested SciPy version. No claim is made for all inputs, all JAX devices or Poisson sampler distribution support.

## Reproducible exact-head validation

[Executable high-precision test](../ci/poisson_high_precision_oracle_20261027.py) imports **actual** `jax._src.random.core._poisson_log_pmf` and the **actual public** `jax.scipy.stats.poisson.logpmf`, checks their equality on 126 inputs, and compares each against 100-digit truth. The script must be executed on an editable fork checkout based on the official PR head to supply native current-head evidence. It is staged but **has not been confirmed to run on the exact fork head or pass GitHub Actions**. Python dependency: `mpmath` in addition to JAX and SciPy.

An independently executed local JAX 0.9.0.1 source-matched probe produced the 126-case results above; [local raw measurements are available](https://github.com/lindicaphxag-tech/jax/tree/research/poisson-highprecision-oracle-20261027) only if subsequently uploaded. Avoid presenting those runtime figures as results of this newly staged exact-checkout script.

## Merge strategy

- Keep [official JAX #41228](https://github.com/jax-ml/jax/pull/41228) focused on numerical correctness; do not tack on a batch-conditional performance branch whose adversarial tail results show regressions.
- Use this independent high-precision oracle to support reproducible discussion **after** it is run on the exact PR head.
- The reviewer has an outstanding hot-path **performance** concern; high-precision correctness evidence does not remove the need for representative, exact-head BTRS CPU/GPU measurements.
- Do not report that maintainer saw these findings: connected GitHub comment write to this official PR previously returned 403.
