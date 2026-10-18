"""Review point M3 (post hoc, existing data): does minimal size explain why the SAME function is easy under one
operator and hard under another? full_gp calculator runs (one and two variables; corrected scoring). Function
fixed effects: exact ~ K + C(function) uses only within-function, across-operator variation in K.
-> paper/results/within_function.md"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    flips = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    d = pd.DataFrame([dict(op=r["grammar"].replace("F2-", "F-"), fn=r["target_id"].split("_calc_")[1], K=float(r["K"]),
                           tid=r["target_id"], y=int(r.get("exact_hash_match") == "true" or k in flips))
                      for k, r in fixed.results("full_gp").items() if r.get("exp") in ("calc", "calc2")])
    multi = d.groupby("fn").op.nunique()
    m = d[d.fn.isin(multi[multi >= 3].index)]
    g = lambda f: smf.glm(f"y ~ {f}", m, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(m["tid"])[0]})
    fe, fe_op = g("K + C(fn)"), g("K + C(fn) + C(op)")
    null = smf.glm("y ~ C(fn)", m, family=sm.families.Binomial()).fit().deviance
    L = ["# Review M3 (post hoc): the same function under different operators", "",
         f"Calculator functions built by at least 3 operators: {m.fn.nunique()} functions, {m.tid.nunique()} "
         f"(operator, function) targets, {len(m)} runs (30 seeds each).", "",
         f"- Function fixed effects, `exact ~ K + C(function)`: K coefficient **{fe.params['K']:+.3f}** per node "
         f"(cluster-robust z = {fe.tvalues['K']:.1f}); deviance explained beyond the function alone: "
         f"{1 - fe.deviance / null:.2f}.",
         f"- Adding the operator, `+ C(operator)`: K coefficient {fe_op.params['K']:+.3f} (z = {fe_op.tvalues['K']:.1f}).", "",
         "| Function | operators | Spearman ρ(K, success) | per operator: K → observed success |", "|---|---|---|---|"]
    rhos = []
    for fn, x in m.groupby("fn"):
        t = x.groupby("op").agg(K=("K", "first"), s=("y", "mean")).sort_values("K")
        rho = spearmanr(t.K, t.s).correlation if t.K.nunique() > 1 and t.s.nunique() > 1 else float("nan")
        rhos.append(rho)
        L.append(f"| {fn} | {len(t)} | {rho:+.2f} | " + ", ".join(f"{o[2:]} {int(k)}→{s:.2f}" for o, (k, s) in t.iterrows()) + " |")
    r = np.array([v for v in rhos if not np.isnan(v)])
    L += ["", f"Median within-function ρ = {np.median(r):+.2f}; negative for {int((r < 0).sum())} of {len(r)} functions "
          f"with variation (ρ undefined where all operators share K or success).", ""]
    out = ROOT / "paper/results/within_function.md"
    out.write_text("\n".join(L) + "\n")
    rate = d.groupby(["op", "fn"]).agg(K=("K", "first"), s=("y", "mean"), n=("y", "size"))
    big = rate[rate.K >= 17]
    values.record("within_function", dict(
        functions=r.size, negative=int((r < 0).sum()), median_rho=float(np.median(r)), n_multi=int(m.fn.nunique()),
        slope=fe.params["K"], z=fe.tvalues["K"], slope_op=fe_op.params["K"], z_op=fe_op.tvalues["K"],
        rate={f"{o[2:]}.{f}": s for (o, f), s in rate.s.items()},
        n_excluded=int(sum(1 for v in rhos if np.isnan(v))),
        twox_successes=int(m[m.fn == "2x"].y.sum()), twox_runs=int((m.fn == "2x").sum()),
        max_rate_k17_excluding_PLI_xx=float(big.drop(("F-PLI", "x^x"), errors="ignore").s.max())))
    print("\n".join(L))


if __name__ == "__main__":
    main()
