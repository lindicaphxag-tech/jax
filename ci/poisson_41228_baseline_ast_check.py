"""Audit that the old benchmark matches the *real* pinned JAX base AST.

This is a source-equivalence check of the Python function body only.
It does NOT establish equal compiled code across different JAX checkouts.
"""
import ast
import hashlib
from pathlib import Path
import urllib.request


BASE_SHA = "a64c1df95008b7402a3cc2e83e036c2431a25999"
BLOB_SHA = "a693bb580ee30c9e768bf256fbbd2baf82a17e04"
SOURCE = ("https://raw.githubusercontent.com/jax-ml/jax/"
          + BASE_SHA + "/jax/_src/random/core.py")


class Normalize(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        # The reference function uses explicit `m.` for the same
        # names that the original implementation accessed in module scope.
        if (isinstance(node.value, ast.Name) and node.value.id == "m"
                and node.attr in ("_split", "uniform", "lax_special",
                                  "lax_control_flow")):
            return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
        return node


def method(tree, name):
    return next(
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def normalize(fn):
    fn.name = "_poisson_rejection"
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    if fn.args.vararg:
        fn.args.vararg.annotation = None
    if fn.args.kwarg:
        fn.args.kwarg.annotation = None
    return Normalize().visit(fn)


def run():
    with urllib.request.urlopen(SOURCE, timeout=30) as res:
        base = res.read()
    blob = hashlib.sha1(
        b"blob " + str(len(base)).encode() + b"\\0" + base
    ).hexdigest()
    if blob != BLOB_SHA:
        raise AssertionError(f"pinned base blob mismatch: {blob}")
    benchmark = Path(__file__).with_name("poisson_41228_paired_perf.py")
    before = normalize(method(ast.parse(base.decode()), "_poisson_rejection"))
    after = normalize(method(ast.parse(benchmark.read_text()), "old_rejection"))
    a = ast.dump(before, include_attributes=False)
    b = ast.dump(after, include_attributes=False)
    if a != b:
        # Avoid printing huge ASTs in CI; every mismatch must be investigated.
        raise AssertionError("reference baseline differs from official pre-fix Python AST")
    print("PINNED_SOURCE_AST_EQUIVALENCE_OK",
          BASE_SHA, BLOB_SHA, flush=True)


if __name__ == "__main__":
    run()
