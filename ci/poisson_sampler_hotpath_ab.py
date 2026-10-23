"""A/B test the sampler-only finite-rate log-PMF specialization.

Execute on the actual JAX checkout; reconstruct the pre-specialization code
from exactly the same production _poisson_rejection AST. No sampling proxy.
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

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)


def stable_before_specialization():
  tree = ast.parse(textwrap.dedent(inspect.getsource(core._poisson_rejection)))
  fun = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
  fun = copy.deepcopy(fun)
  fun.name = "_benchmark_stable_before_sampler_specialization"
  fun.decorator_list = []

  class RestoreDefaultHelper(ast.NodeTransformer):
    changes = 0

    def visit_Call(self, node):
      node = self.generic_visit(node)
      if isinstance(node.func, ast.Name) and node.func.id == "_poisson_log_pmf":
        assert len(node.keywords) == 1
        assert node.keywords[0].arg == "_sampler_fast_path"
        assert isinstance(node.keywords[0].value, ast.Constant)
        assert node.keywords[0].value.value is True
        node.keywords = []
        self.changes += 1
      return node

  patch = RestoreDefaultHelper()
  fun = patch.visit(fun)
  assert patch.changes == 1
  module = ast.fix_missing_locations(ast.Module(body=[fun], type_ignores=[]))
  namespace = vars(core).copy()
  exec(compile(module, "<source-matched-pre-fast-path>", "exec"), namespace)
  return jax.jit(namespace[fun.name], static_argnums=(2, 3, 4))


def paired_timing(old, new, key, rates, n, repeats=15):
  shape = (n,)
  dtype = np.dtype("int32")
  iterations = int(np.iinfo(np.int32).max)
  args = (key, rates, shape, dtype, iterations)
  old(*args).block_until_ready()
  new(*args).block_until_ready()
  times = [[], []]
  funcs = (old, new)
  for i in range(repeats):
    order = (0, 1) if i % 2 == 0 else (1, 0)
    for idx in order:
      start = time.perf_counter()
      funcs[idx](*args).block_until_ready()
      times[idx].append(1000 * (time.perf_counter() - start))
  return {
      "old_median_ms": statistics.median(times[0]),
      "new_median_ms": statistics.median(times[1]),
      "ratio": statistics.median(times[1]) / statistics.median(times[0]),
      "min_ms": [min(a) for a in times],
      "max_ms": [max(a) for a in times],
  }


def main():
  old = stable_before_specialization()
  new = core._poisson_rejection
  key = jax.random.key(777)
  print("VERSIONS", jax.__version__, jax.lib.__version__, jax.devices())
  results = []
  for lam in (10., 1000., 1e8):
    for n in (1024, 8192):
      rates = jnp.full((n,), lam, dtype=jnp.float32)
      times = paired_timing(old, new, key, rates, n)
      for seed in (42, 273, 777):
        k = jax.random.key(seed)
        old_samples = np.asarray(old(k, rates, (n,), np.dtype("int32"), int(np.iinfo(np.int32).max)))
        new_samples = np.asarray(new(k, rates, (n,), np.dtype("int32"), int(np.iinfo(np.int32).max)))
        mismatches = int(np.count_nonzero(old_samples != new_samples))
        print("EQUIVALENCE", json.dumps({"lambda": lam, "n": n, "seed": seed, "mismatches": mismatches}))
        assert mismatches == 0, (lam, n, seed, mismatches)
      record = {"lambda": lam, "n": n, **times}
      results.append(record)
      print("AB_TIMING", json.dumps(record, sort_keys=True), flush=True)
  print("SUMMARY", json.dumps(results, sort_keys=True), flush=True)
  print("SAMPLER FAST PATH EQUIVALENCE PASSED", flush=True)


if __name__ == "__main__":
  main()
