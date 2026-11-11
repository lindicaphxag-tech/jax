"""Reproducible high-precision certificate for the Poisson central deviance series.

For k >= 0, |t|<1, the degree-8 polynomial for
 k * [log(1-t) + t] has absolute *truncation* remainder at most
 k*|t|^9 / (9*(1-|t|)).
This does not bound floating-point rounding, Stirling tail, or full-sampler error.
"""
from decimal import Decimal, localcontext

with localcontext() as ctx:
    ctx.prec = 70
    ratios = ("-0.01", "-0.001", "0", "0.001", "0.01")
    ks = (1, 10_000, 100_000_000, 2_147_483_647)
    maximum_bound = Decimal(0)
    for k in ks:
        for raw in ratios:
            t = Decimal(raw)
            K = Decimal(k)
            exact = K * ((Decimal(1) - t).ln() + t)
            poly = -K * sum(t**n / Decimal(n) for n in range(2, 9))
            bound = K * abs(t)**9 / (Decimal(9) * (Decimal(1) - abs(t)))
            observed = abs(exact - poly)
            assert observed <= bound + Decimal("1e-60"), (k, t, observed, bound)
            maximum_bound = max(maximum_bound, bound)
            print(f"PASS k={k} t={t} residual={observed:.4E} bound={bound:.4E}")
    assert maximum_bound < Decimal("2.42e-10")
    print(f"PASS exact-series truncation certificate max bound={maximum_bound:.6E}")
    print("SCOPE: excludes float32 rounding, Gamma/Stirling remainder and rejection logic")
