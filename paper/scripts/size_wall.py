"""Post hoc (not preregistered): like-for-like budget scaling. K50 per budget on the SAME targets at every budget
(paired target bootstrap), outcomes with the corrected scoring (data/rescored), on the fixed engine (fixed.py).
Set A: all seven operators, sizes 9-17, the 16x targets (first 20 per size): E0, 4x, 16x.
Set B: EML, LDE0, PLM, sizes 11-17, the 64x targets (first 10 per size): E0, 4x, 16x, 64x.
-> paper/results/size_wall.md"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.family_full import _k50
from reach.family import FAMILY
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B, RNG = 1000, np.random.default_rng(20261017)   # same stream as exchange_rate.py, so the slope intervals agree


def outcomes():
    """{(op, target, budget_multiple): [y, ...]}"""
    d = defaultdict(list)
    for ph, exp, mult in (("full_gp", "conf", None), ("full_budget", "budget", None), ("full_budget64", "budget64", 64)):
        flips = set(json.loads((ROOT / f"data/rescored/{ph}.json").read_text()))
        for k, r in fixed.results(ph).items():
            if r.get("exp") != exp:
                continue
            m = mult or {200: 1, 800: 4, 3200: 16}[r["generations"]]
            d[(r["grammar"], r["target_id"], m)].append(float(r.get("exact_hash_match") == "true" or k in flips))
    return d


def table(d, ops, mults, ref):
    """ref: budget multiple whose target set defines the matched set."""
    tg = {g: sorted({t for (o, t, m) in d if o == g and m == ref}) for g in ops}
    K = {t: int(t.split("_K")[1][:2]) for g in ops for t in tg[g]}

    def k50(sel, g, m):
        ks, ys = [], []
        for t in sel[g]:
            ys += d[(g, t, m)]; ks += [K[t]] * len(d[(g, t, m)])
        return _k50(np.array(ks, float), np.array(ys))

    def pooled(sel, m):
        ks, ys = [], []
        for g in ops:
            for t in sel[g]:
                ys += d[(g, t, m)]; ks += [K[t]] * len(d[(g, t, m)])
        return _k50(np.array(ks, float), np.array(ys))

    pt = {(g, m): k50(tg, g, m) for g in ops for m in mults}
    pp = {m: pooled(tg, m) for m in mults}
    reps = []
    for _ in range(B):
        sel = {g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in ops}
        reps.append({m: pooled(sel, m) for m in mults})
    x = np.log2(np.array(mults, float))
    slope = lambda v: float(np.polyfit(x, v, 1)[0])
    sl = slope([pp[m] for m in mults])
    bs = [slope([r[m] for m in mults]) for r in reps if all(np.isfinite(r[m]) for m in mults)]
    return tg, pt, pp, sl, np.quantile(bs, [0.025, 0.975]), len(bs)


def main():
    d = outcomes()
    L = ["# Post hoc: like-for-like budget scaling (the size wall)", "",
         "Same targets at every budget; corrected scoring; crashed EDL runs replaced by their fixed-engine reruns; paired target bootstrap (B = 1000). "
         "Slope = increase of the pooled K50 per doubling of the budget (least squares on log2 budget).", ""]
    V = {}
    for name, ops, mults, ref, sizes in (("A: seven operators, sizes 9–17", FAMILY, [1, 4, 16], 16, "9–17"),
                                         ("B: EML, LDE0, PLM, sizes 11–17", ["F-EML", "F-LDE0", "F-PLM"], [1, 4, 16, 64], 64, "11–17")):
        tg, pt, pp, sl, ci, nb = table(d, ops, mults, ref)
        L += [f"## Set {name}", "", "| Operator | targets | " + " | ".join(f"K50 at {m}×" for m in mults) + " |",
              "|---|---|" + "---|" * len(mults)]
        L += [f"| {g} | {len(tg[g])} | " + " | ".join(f"{pt[(g, m)]:.2f}" for m in mults) + " |" for g in ops]
        L += [f"| **pooled** | {sum(len(v) for v in tg.values())} | " + " | ".join(f"**{pp[m]:.2f}**" for m in mults) + " |", "",
              f"Pooled increase per doubling: **{sl:+.2f} nodes** (95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], {nb} estimable replicates). "
              f"Total from E0 to {mults[-1]}×: {pp[mults[-1]] - pp[1]:+.2f} nodes.", ""]
        V[name[0]] = dict(pooled={str(m): pp[m] for m in mults}, slope=sl, slope_lo=ci[0], slope_hi=ci[1],
                          k50={f"{g[2:]}.{m}": pt[(g, m)] for g in ops for m in mults})
    out = ROOT / "paper/results/size_wall.md"
    out.write_text("\n".join(L) + "\n")
    values.record("size_wall", V)
    print("\n".join(L))


if __name__ == "__main__":
    main()
