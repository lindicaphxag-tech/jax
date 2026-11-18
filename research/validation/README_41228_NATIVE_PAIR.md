# JAX #41228 — reproducible exact-commit source benchmark

**Scope:** fork-only CPU evidence for [upstream JAX #41228](https://github.com/jax-ml/jax/pull/41228). This evidence branch is **not** the submitted PR branch and does not modify its implementation.

## Source identities (pinned)

- Baseline: `d3f12669d7db55596d413eba9cfcd214ae5a3ea4` (upstream PR base at audit)
- Candidate: `9428e21c09e6a19cadbf9ed02d6ec78b165f6246` (submitted PR HEAD at audit)
- Evidence branch: `evidence/41228-native-rejection-benchmark-20261119`
- Source benchmark: `benchmark_poisson_41228_native.py`
- Machine-readable comparator: `compare_poisson_41228_native.py`
- Public, explicit-source workflow: `.github/workflows/poisson_41228_exact_commit_pair.yml`

**Design:** Run the same harness in separate clean source checkouts, and require the checked-out Git SHA *and* `jax.__file__` rooted in the checkout. The harness times the actual public `jax.random.poisson` JIT sampling path (which selects the rejection path for these rates) as well as `jax.scipy.stats.poisson.logpmf`, records first-call/compilation latency separately, then 40 warmed calls plus sample moment smoke checks. It reports float32 and float64; rates 3e6 and 1e8; batch sizes 256 and 2048. No arbitrary runtime threshold or pass/fail performance claim is baked into comparison: numerical errors, ratios and independent sample statistics are reported.

## Reproduce locally from two separate worktrees

```bash
# In a fresh fork clone with the evidence branch available:
git worktree add /tmp/jax-41228-base d3f12669d7db55596d413eba9cfcd214ae5a3ea4
git worktree add /tmp/jax-41228-candidate 9428e21c09e6a19cadbf9ed02d6ec78b165f6246
# Use separate environments for each checkout, each installing its respective JAX:
cd /tmp/jax-41228-base
python -m pip install -e ".[ci]" scipy
python /path/to/evidence/research/validation/benchmark_poisson_41228_native.py \
  --require-sha d3f12669d7db55596d413eba9cfcd214ae5a3ea4 --out baseline.json
cd /tmp/jax-41228-candidate
python -m pip install -e ".[ci]" scipy
python /path/to/evidence/research/validation/benchmark_poisson_41228_native.py \
  --require-sha 9428e21c09e6a19cadbf9ed02d6ec78b165f6246 --out candidate.json
python /path/to/evidence/research/validation/compare_poisson_41228_native.py \
  /tmp/jax-41228-base/baseline.json candidate.json --out comparison.json
```

**Important:** The two installations must be run in separate Python environments or separate fresh hosts so that editable installs and imported JAX modules cannot collide. The GitHub Actions workflow uses separate jobs for this purpose. For the most defensible *paired* latency estimate, both versions should be run under controlled resource allocation on the **same host**, alternating order at the whole-run level. Different GitHub-hosted runners are useful as a coarse CI screen, **not a controlled performance comparison**.

## Verification status

At creation, GitHub fork Actions returned **zero runs** on this evidence branch; no success or timing result can be claimed from the new workflow. Existing earlier fork benchmark `research/poisson_41228_benchmark.py` on `ci/poisson-41228-native-review-gate-20261118` already implements a more closely *paired* ABBA comparison by emulating the original acceptance path inside the candidate checkout. Its source and script exist, but that branch also has no observed Actions run. These approaches answer different questions: **historical algorithm delta on one environment** versus **exact base/candidate checkout behavior**.

## Remaining maintainer gates

- Upstream reviewer `jakevdp` formally requested test consolidation and benchmarking of the rejection loop. `CHANGES_REQUESTED` remains, with no inferred approval.
- Need executed source-pinned benchmark results, ideally full-suite native tests and GPU/TPU context if requested.
- Need assess whether production complexity should be reduced rather than merely optimizing the benchmark.
- Do not reopen TRL audit-only bugs or send repeat PR comments as substitutes for real incidents.
