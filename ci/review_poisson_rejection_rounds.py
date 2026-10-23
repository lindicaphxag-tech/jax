"""Inspect exact BTRS while-loop iteration counts old vs fixed PMF.

The instrumented function is mechanically identical to JAX's checked-out
_poisson_rejection, except it returns the loop iteration counter too. The
old acceptance expression is restored exactly as in the paired benchmark.
This separates changing rejection dynamics from log-PMF computational cost.
"""
import inspect
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax._src.random import core


def traced_sampler(name, *, old):
  source = inspect.getsource(core._poisson_rejection)
  a = "def _poisson_rejection("
  assert source.count(a) == 1
  source = source.replace(a, f"def {name}(")
  body = (
      "  k = lax_control_flow.while_loop(cond_fn, body_fn, (0, k_init, accepted, key))[1]\n"
      "  return k.astype(dtype)"
  )
  assert source.count(body) == 1, "Source changed: review BTRS instrumentation"
  source = source.replace(
      body,
      "  final_carry = lax_control_flow.while_loop(cond_fn, body_fn, (0, k_init, accepted, key))\n"
      "  return final_carry[1].astype(dtype), final_carry[0]",
  )
  if old:
    assert source.count("t = _poisson_log_pmf(k, lam)") == 1
    assert source.count("  b = 0.931 + 2.53 * lax.sqrt(lam)") == 1
    source = source.replace(
        "  b = 0.931 + 2.53 * lax.sqrt(lam)",
        "  log_lam = lax.log(lam)\n  b = 0.931 + 2.53 * lax.sqrt(lam)",
    ).replace(
        "t = _poisson_log_pmf(k, lam)",
        "t = -lam + k * log_lam - lax_special.lgamma(k + 1)",
    )
  ns = vars(core).copy()
  exec(compile(source, "<rejection-iteration-audit>", "exec"), ns)
  return ns[name]


def main():
  print("JAX",jax.__version__,"source",core.__file__,"devices",jax.devices())
  assert "/site-packages/" not in core.__file__
  old = traced_sampler("_poisson_rejection_old_with_count", old=True)
  new = traced_sampler("_poisson_rejection_new_with_count", old=False)
  for count, lam in [(1024,1e4),(1024,1e8),(8192,1e8),(32768,1e8)]:
    rates = jnp.full((count,), lam, dtype=jnp.float32)
    key = random.key(20261023)
    original, old_rounds = old(key, rates, (count,), jnp.int32, 100)
    fixed, new_rounds = new(key, rates, (count,), jnp.int32, 100)
    original.block_until_ready()
    fixed.block_until_ready()
    old_np = np.asarray(original, dtype=np.float64)
    new_np = np.asarray(fixed, dtype=np.float64)
    print("BTRS_ROUNDS",
          "size",count,"lambda",lam,
          "old",int(old_rounds),"new",int(new_rounds),
          "old_std",round(float(old_np.std()),2),
          "fixed_std",round(float(new_np.std()),2),
          "target_std",round(float(np.sqrt(lam)),2))
    assert (new_np >= 0).all()
    if lam == 1e8 and count >= 8192:
      assert abs(new_np.std() / np.sqrt(lam) - 1.0) < 0.12
  print("PASS: real JAX rejection sampler loop rounds instrumented without changing algorithm")


if __name__ == "__main__":
  main()
