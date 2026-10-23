"""Causal control for JAX #41228 batch-certified Poisson acceptance.

Comparisons:
  stable: source-matched PR #41228 helper called directly
  gated: experimental lazy batch central certificate
  forced_fallback: same gated helper and cond structure with predicate false

The third arm tests whether speedups arise even when the central formula is
never executed; CPU HLO text length/first compile time are diagnostics, not
validated compiler optimization attribution.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import statistics
import textwrap
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.random import core
from poisson_batched_central_ab import old_stable_sampler

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)


def forced_fallback_sampler():
  source = textwrap.dedent(inspect.getsource(core._poisson_rejection_log_pmf))
  tree = ast.parse(source)
  f = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)))
  f.name = "_review_forced_fallback_score"

  class ReplaceCertificate(ast.NodeTransformer):
    replaced = 0
    def visit_Assign(self, node):
      if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id == "central":
          self.replaced += 1
          node.value = ast.parse("jnp.asarray(False)", mode="eval").body
      return self.generic_visit(node)

  patch = ReplaceCertificate()
  f = patch.visit(f)
  assert patch.replaced == 1
  ns = vars(core).copy()
  module = ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[]))
  exec(compile(module, "<forced-fallback-score>", "exec"), ns)

  source = textwrap.dedent(inspect.getsource(core._poisson_rejection))
  tree = ast.parse(source)
  outer = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)))
  outer.decorator_list = []
  outer.name = "_review_forced_fallback_sampler"

  class ReplaceHelper(ast.NodeTransformer):
    replaced = 0
    def visit_Call(self, node):
      node = self.generic_visit(node)
      if isinstance(node.func, ast.Name) and node.func.id == "_poisson_rejection_log_pmf":
        self.replaced += 1
        node.func.id = "_review_forced_fallback_score"
      return node

  rewrite = ReplaceHelper()
  outer = rewrite.visit(outer)
  assert rewrite.replaced == 1
  mod2 = ast.fix_missing_locations(ast.Module(body=[outer], type_ignores=[]))
  exec(compile(mod2, "<forced-fallback-sampler>", "exec"), ns)
  return jax.jit(ns[outer.name], static_argnums=(2, 3, 4))


def cases(n):
  large = np.full(n, 1e8, np.float32)
  one_low = large.copy()
  one_low[0] = 10.0
  stripe = np.resize(np.array([10., 1000., 1e6, 1e8], np.float32), n)
  return {
      "homogeneous_large": large,
      "one_low_rate_disables_gate": one_low,
      "heterogeneous_stripes": stripe,
      "homogeneous_small": np.full(n, 1000.0, np.float32),
  }


def compile_diagnostic(fn, key, rates, n):
  args = (key, rates, (n,), np.dtype("int32"), int(np.iinfo(np.int32).max))
  t0 = time.perf_counter()
  lowered = fn.lower(*args)
  stablehlo_text = lowered.as_text()
  t1 = time.perf_counter()
  executable = lowered.compile()
  t2 = time.perf_counter()
  # static_argnums=(2, 3, 4) are embedded in the compiled executable.
  executable(key, rates).block_until_ready()
  return {"lower_ms": (t1-t0)*1000, "compile_ms": (t2-t1)*1000,
          "stablehlo_chars":len(stablehlo_text)}


def measure(funcs, rates, n, *, blocks=4, repeats=13):
  shape = (n,)
  dtype = np.dtype("int32")
  bound = int(np.iinfo(np.int32).max)
  perblock = []
  for block in range(blocks):
    key = jax.random.key(291 + 991 * block)
    args = (key, rates, shape, dtype, bound)
    for fn in funcs:
      fn(*args).block_until_ready()
    measurements = [[], [], []]
    for rep in range(repeats):
      order = ((0,1,2), (1,2,0), (2,0,1))[(rep+block)%3]
      for index in order:
        start = time.perf_counter()
        funcs[index](*args).block_until_ready()
        measurements[index].append((time.perf_counter()-start)*1000)
    medians=[statistics.median(x) for x in measurements]
    perblock.append({"block":block, "stable_ms":medians[0],
          "gated_ms":medians[1], "fallback_ms":medians[2],
          "gated_over_stable":medians[1]/medians[0],
          "fallback_over_stable":medians[2]/medians[0]})
  return {"median_gated_over_stable":statistics.median(x["gated_over_stable"] for x in perblock),
      "median_fallback_over_stable":statistics.median(x["fallback_over_stable"] for x in perblock),
      "blocks":perblock}


def main():
  funcs=(old_stable_sampler(),core._poisson_rejection,forced_fallback_sampler())
  n=8192
  print("BACKEND",jax.__version__,jax.lib.__version__,jax.devices(),flush=True)
  results=[]
  for title, raw in cases(n).items():
    rates=jnp.asarray(raw)
    key=jax.random.key(17)
    diagnostics=[compile_diagnostic(fn,key,rates,n) for fn in funcs]
    mismatches=[0,0]
    for seed in (31,73,101,997):
      key=jax.random.key(seed)
      args=(key,rates,(n,),np.dtype("int32"),int(np.iinfo(np.int32).max))
      baseline=np.asarray(funcs[0](*args))
      for i in (1,2):
        mismatches[i-1]+=int(np.count_nonzero(baseline != np.asarray(funcs[i](*args))))
    timing=measure(funcs,rates,n)
    result={"case":title,"mismatches_gated_fallback":mismatches,
            "lower_compile_hlo":[{"arm":name,**d} for name,d in zip(
                ("stable","gated","forced_fallback"),diagnostics)],
            **timing}
    results.append(result)
    print("CAUSAL_RESULT",json.dumps(result,sort_keys=True),flush=True)
    assert mismatches==[0,0],result
  with open("poisson_admission_causal_ablation.json","w",encoding="utf-8") as f:
    json.dump({"jax":jax.__version__,"results":results},f,indent=2)
  print("CAUSAL_ABLATION_COMPLETE",flush=True)


if __name__=="__main__":
  main()
