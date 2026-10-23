"""Source-matched JAX Poisson sampler lazy-center A/B.

Control reconstructs exactly the same current _poisson_rejection AST with
its one log-PMF call reverted to the already-reviewed stable helper.
The intervention changes only the per-iteration numerical score evaluator.
Do not equate a matched-key comparison with a general proof for every seed.
"""
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


def old_stable_sampler():
  tree = ast.parse(textwrap.dedent(inspect.getsource(core._poisson_rejection)))
  f = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)))
  f.decorator_list = []
  f.name = "_reviewed_stable_rejection"
  class RevertScore(ast.NodeTransformer):
    matches = 0
    def visit_Call(self, node):
      node = self.generic_visit(node)
      if isinstance(node.func, ast.Name) and node.func.id == "_poisson_rejection_log_pmf":
        self.matches += 1
        node.func.id = "_poisson_log_pmf"
      return node
  action = RevertScore()
  f = action.visit(f)
  assert action.matches == 1, "unexpected production source"
  code = ast.fix_missing_locations(ast.Module(body=[f], type_ignores=[]))
  context = vars(core).copy()
  exec(compile(code, "<reviewed-exact-stable-sampler>", "exec"), context)
  return jax.jit(context[f.name], static_argnums=(2, 3, 4))


def compare(old, new, lam, n, seeds=(42, 273, 777, 36000, 36001, 36002)):
  shape, dtype = (n,), np.dtype("int32")
  iters = int(np.iinfo(dtype).max)
  x = jnp.full(shape, lam, dtype=jnp.float32)
  disagreements = 0
  for seed in seeds:
    key = jax.random.key(seed)
    a = np.asarray(old(key, x, shape, dtype, iters))
    b = np.asarray(new(key, x, shape, dtype, iters))
    disagreements += np.count_nonzero(a != b)
  print("EQUIVALENCE", json.dumps({"lambda":lam, "n":n,"seeds":len(seeds),
                                   "mismatches":int(disagreements)}), flush=True)
  assert disagreements == 0
  return x


def timed_pair(old, new, lam, n, repeats=15):
  shape, dtype = (n,), np.dtype("int32")
  iters = int(np.iinfo(dtype).max)
  x = jnp.full(shape, lam, dtype=jnp.float32)
  key = jax.random.key(31337)
  args = (key, x, shape, dtype, iters)
  for f in (old, new):
    f(*args).block_until_ready()
  samples = [[], []]
  for i in range(repeats):
    order = (0,1) if i%2==0 else (1,0)
    for index in order:
      t0 = time.perf_counter()
      (old,new)[index](*args).block_until_ready()
      samples[index].append((time.perf_counter()-t0)*1000)
  old_ms = statistics.median(samples[0])
  new_ms = statistics.median(samples[1])
  print("TIMING", json.dumps({"lambda":lam,"n":n,"old_ms":old_ms,
        "new_ms":new_ms,"ratio":new_ms/old_ms,
        "old_min":min(samples[0]),"new_min":min(samples[1])}),flush=True)


def main():
  old = old_stable_sampler()
  new = core._poisson_rejection
  print("ENV", jax.__version__,jax.lib.__version__,jax.devices())
  for lam in (10.,1000.,1e6,1e8):
    for n in (1024,8192):
      compare(old,new,lam,n)
      timed_pair(old,new,lam,n)
  print("ALL TESTED SAMPLER ARRAYS MATCH",flush=True)


if __name__ == "__main__":
  main()
