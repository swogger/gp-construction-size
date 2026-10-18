"""Review point m2 (post hoc, existing data): are the proven shortest constructions (found on the grid in (2, 3))
identities beyond that domain? Each calculator-function construction (data/family/minimal_<op>.json) is evaluated on
wider real domains and compared with the function (relative tolerance 1e-8, where both are finite).
-> paper/results/domain_check.md"""
import json
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.family import CALC, CALC2, FAMILY
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = {"(0.5, 6)": (0.5, 6.0), "(0.05, 20)": (0.05, 20.0)}


def check(g, expr, f, lo, hi, rng):
    x = rng.uniform(lo, hi, 400).astype(np.complex128)
    xs = x if len(g.variables) == 1 else {"x": x, "y": rng.uniform(lo, hi, 400).astype(np.complex128)}
    want = f(xs) if len(g.variables) == 1 else f(xs["x"], xs["y"])
    got = G.evaluate(g, G.parse_expr(expr, g), xs)
    wf = np.isfinite(want)                      # points where the function itself is defined (finite)
    both = wf & np.isfinite(got)
    wrong = both & ~np.isclose(got, want, rtol=1e-8, atol=1e-10)
    undefined = wf & ~np.isfinite(got)
    if wrong.any():
        return "wrong", float(wrong[wf].mean())
    if undefined.mean() > 0.05 * max(wf.mean(), 1e-9):
        return "undefined", float(undefined[wf].mean())
    return "holds", 0.0


def main():
    rng = np.random.default_rng(0)
    L = ["# Review m2 (post hoc): do the shortest constructions hold beyond (2, 3)?", "",
         "Each proven shortest construction of a calculator function is evaluated at 400 random points in wider real "
         "domains (x, and y for two-variable functions). Classified at points where the function itself is finite: *wrong* = the construction gives a different "
         "value somewhere (relative 1e-8), *undefined* = it overflows or is NaN at > 5% of them, otherwise *holds* (no counterexample among the 400 points).", "",
         "| Operator | constructions | " + " | ".join(f"{k}: holds / undefined / wrong" for k in DOMAINS) + " | wrong on (0.05, 20) |",
         "|---|---|" + "---|" * len(DOMAINS) + "---|"]
    tot = {k: None for k in DOMAINS}; n_all = 0; V = {}
    for op in FAMILY:
        rows, fails = [], []
        for gname, fns in ((op, CALC), ("F2-" + op[2:], CALC2)):
            p = ROOT / f"data/family/minimal_{gname}.json"
            if not p.exists():
                continue
            g, tab = G.get(gname), json.loads(p.read_text())["table"]
            for fn, hit in tab.items():
                if fn not in fns or not hit or not hit.get("verified"):
                    continue
                res = {k: check(g, hit["expr"], fns[fn], *v, rng) for k, v in DOMAINS.items()}
                rows.append({k: v[0] for k, v in res.items()})
                if res["(0.05, 20)"][0] == "wrong":
                    fails.append(f"{fn} ({res['(0.05, 20)'][1]:.0%})")
        n = len(rows); n_all += n
        V[op] = dict(n=n, wrong_mid=sum(r["(0.5, 6)"] == "wrong" for r in rows))
        cnt = lambda k, c: sum(r[k] == c for r in rows)
        for k in DOMAINS:
            tot[k] = [a + b for a, b in zip(tot.get(k) or [0, 0, 0], [cnt(k, "holds"), cnt(k, "undefined"), cnt(k, "wrong")])]
        L.append(f"| {op} | {n} | " + " | ".join(f"{cnt(k, 'holds')} / {cnt(k, 'undefined')} / {cnt(k, 'wrong')}" for k in DOMAINS)
                 + f" | {', '.join(fails) or '–'} |")
    L += ["", "All operators (holds / undefined / wrong): " + "; ".join(f"{k}: {t[0]} / {t[1]} / {t[2]} of {n_all}" for k, t in tot.items())
          + ". Percentages: share of points (where the function is finite) with a wrong value.", ""]
    out = ROOT / "paper/results/domain_check.md"
    out.write_text("\n".join(L) + "\n")
    values.record("domain_check", {op[2:]: v for op, v in V.items()})
    print("\n".join(L))


if __name__ == "__main__":
    main()
