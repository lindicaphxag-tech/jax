"""Independent *exact PR source* CPU benchmark for JAX Poisson review #41228.

The legacy comparator is built from the identical production
_poisson_rejection AST with ONLY the acceptance log-PMF expression restored
to the pre-patch formula. The new sampler is the actual production method.
Both use identical requests/keys, a single CPU runner and JIT warmups.

Run only on a fork validation branch. This is not a production algorithm change.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import os
import statistics
import textwrap
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from jax._src import lax
from jax._src.random import core

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)


def source_matched_legacy_rejection():
  source = textwrap.dedent(inspect.getsource(core._poisson_rejection))
  tree = ast.parse(source)
  func = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
  func = copy.deepcopy(func)
  func.name = "_benchmark_legacy_poisson_rejection"
  func.decorator_list = []

  class RevertLogPmf(ast.NodeTransformer):
    count = 0

    def visit_Assign(self, node):
      if (
          len(node.targets) == 1
          and isinstance(node.targets[0], ast.Name)
          and node.targets[0].id == "t"
          and isinstance(node.value, ast.Call)
          and isinstance(node.value.func, ast.Name)
          and node.value.func.id == "_poisson_log_pmf"
      ):
        self.count += 1
        original = ast.parse("t = -lam + k * log_lam - lax_special.lgamma(k + 1)").body[0]
        return ast.copy_location(original, node)
      return self.generic_visit(node)

  transform = RevertLogPmf()
  func = transform.visit(func)
  # Restore the original per-function, pre-loop log(lambda) hoist as well.
  func.body.insert(1, ast.parse("log_lam = lax.log(lam)").body[0])
  assert transform.count == 1, "Production source no longer has the expected one-line log-PMF substitution"
  module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
  globals_ns = vars(core).copy()
  exec(compile(module, "<native-poisson-source-minus-stable-logpmf>", "exec"), globals_ns)
  return jax.jit(globals_ns[func.name], static_argnums=(2, 3, 4))


def paired_medians(base, patch, key, lam, n, repeats=15):
  # Interleave and reverse order to reduce JIT warmup, thermal and drift bias.
  shape = (n,)
  dtype = np.dtype("int32")
  max_iters = int(np.iinfo(np.int32).max)
  fns = (base, patch)
  for fn in fns:
    fn(key, lam, shape, dtype, max_iters).block_until_ready()
  series = [[], []]
  for i in range(repeats):
    order = (0, 1) if i % 2 == 0 else (1, 0)
    for idx in order:
      t0 = time.perf_counter()
      fns[idx](key, lam, shape, dtype, max_iters).block_until_ready()
      series[idx].append(time.perf_counter() - t0)
  return statistics.median(series[0]), statistics.median(series[1]), {
      "min_ms": [round(min(x) * 1000, 5) for x in series],
      "max_ms": [round(max(x) * 1000, 5) for x in series],
      "repeats": repeats,
  }



def empirical_sampler_comparison(base, patch):
  """Predeclared paired-seed distribution diagnostics, not tuned to the result.

  Both native sampling methods receive the same keys and inputs. The Poisson
  standardized law is near normal for these very large rates; we also report
  empirical moments against exact Poisson mean and variance. The standard
  normal KS diagnostic is a quality check, not an exact Poisson GOF p-value.
  """
  result=[]
  n=8192
  shape=(n,)
  dtype=np.dtype("int32")
  max_iters=int(np.iinfo(np.int32).max)
  for mean in (1e6, 1e8):
    lam=jnp.full(shape,mean,dtype=jnp.float32)
    arrays=[[],[]]
    for seed in range(8):
      key=jax.random.key(16000+seed)
      arrays[0].append(np.asarray(base(key,lam,shape,dtype,max_iters),dtype=np.float64))
      arrays[1].append(np.asarray(patch(key,lam,shape,dtype,max_iters),dtype=np.float64))
    old,cur=(np.concatenate(samples) for samples in arrays)
    measured=[]
    for tag,x in (("original",old),("stable",cur)):
      standardized=(x-mean)/np.sqrt(mean)
      ks=float(scipy.stats.kstest(standardized,scipy.stats.norm.cdf).statistic)
      moments={
          "method":tag,
          "count":len(x),
          "mean_error_standard_errors":float((x.mean()-mean)/np.sqrt(mean/len(x))),
          "variance_to_poisson":float(x.var(ddof=1)/mean),
          "normal_ks":ks,
          "incomplete_fraction":float(np.mean(x<0)),
      }
      measured.append(moments)
      print("DISTRIBUTION",json.dumps({"lambda":mean,**moments},sort_keys=True),flush=True)
    assert measured[1]["incomplete_fraction"]==0,measured[1]
    assert 0.8<measured[1]["variance_to_poisson"]<1.2,measured[1]
    row={
        "lambda":mean,
        "same_key_changed_fraction":float(np.mean(old!=cur)),
        "original":measured[0],
        "stable":measured[1]
    }
    result.append(row)
  return result


def main():
  baseline = source_matched_legacy_rejection()
  new = core._poisson_rejection
  key = jax.random.key(777)
  print("native_jax_version", jax.__version__, "jaxlib", jax.lib.__version__)
  print("device", str(jax.devices()[0]))
  result = []
  for mean in (10.0, 1000.0, 1e8):
    for n in (1024, 8192):
      rates = jnp.full((n,), mean, dtype=jnp.float32)
      old_s, new_s, spread = paired_medians(baseline, new, key, rates, n)
      row = {
          "lambda": mean,
          "n": n,
          "baseline_ms": round(old_s * 1000, 5),
          "patch_ms": round(new_s * 1000, 5),
          "slowdown": round(new_s / old_s, 4),
          "span": spread,
      }
      result.append(row)
      print("TIMING", json.dumps(row, sort_keys=True), flush=True)
  k = jnp.asarray(1e8, dtype=jnp.float32) + jnp.asarray(
      [-10000.0, -1000.0, 0.0, 1000.0, 10000.0], dtype=jnp.float32)
  actual = np.asarray(core._poisson_log_pmf(k, jnp.asarray(1e8, dtype=jnp.float32)))
  oracle = scipy.stats.poisson.logpmf(np.asarray(k, dtype=np.float64), 1e8)
  max_err = float(np.max(np.abs(actual - oracle)))
  assert np.all(np.isfinite(actual)), actual
  assert max_err < 0.02, (actual, oracle, max_err)
  print("REFERENCE", json.dumps({"abs_logpmf_error_max": max_err}))
  distribution=empirical_sampler_comparison(baseline,new)
  # A numerical performance ratio is a measurement, not an acceptance criterion
  # or a claim of statistical bias correction.
  with open("poisson_sampler_benchmark.json", "w", encoding="utf-8") as out:
    json.dump(
        {"jax": jax.__version__, "jaxlib": jax.lib.__version__,
         "device": str(jax.devices()[0]), "results": result,
         "distribution":distribution,
         "max_logpmf_abs_error": max_err,
         "method": "native_source_ast_single_expression_revert"},
        out, indent=2)
  print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
  main()
