"""Review point M2 (post hoc, existing data): how much does minimal size matter compared with the operator?
full_gp scout runs (corrected scoring; fixed engine). Logistic models; McFadden pseudo-R2; likelihood-ratio test of
the operator given size. Compute equivalents are in exchange_rate.py. -> paper/results/size_vs_operator.md"""
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main():
    flips = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    d = pd.DataFrame([dict(op=r["grammar"], K=float(r["K"]), tid=r["target_id"],
                           y=int(r.get("exact_hash_match") == "true" or k in flips))
                      for k, r in fixed.results("full_gp").items() if r.get("exp") == "conf"])
    fit = lambda f, data: smf.glm(f"y ~ {f}", data, family=sm.families.Binomial()).fit()
    L = ["# Review M2 (post hoc): minimal size against operator", "",
         f"`full_gp` scout runs: {len(d)} runs, {d.tid.nunique()} targets, 7 operators; corrected scoring.", "",
         "| Data | Model | McFadden R² | deviance |", "|---|---|---|---|"]
    lr, r2 = {}, {}
    for name, data in (("all seven", d), ("six (without EDL)", d[d.op != "F-EDL"])):
        null = fit("1", data).deviance
        dev = {}
        for f in ("K", "C(op)", "K + C(op)"):
            m = fit(f, data); dev[f] = m.deviance
            L.append(f"| {name} | {f} | {1 - m.deviance / null:.3f} | {m.deviance:.0f} |")
            r2[(name, f)] = 1 - m.deviance / null
        lr[name] = (dev["K"] - dev["K + C(op)"], data.op.nunique() - 1)
    # cluster-robust Wald test of the operator given size (runs nested in targets; D043)
    import numpy as np
    mw = smf.glm("y ~ K + C(op)", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.tid)[0]})
    names = [n for n in mw.params.index if n.startswith("C(op)")]
    R = np.zeros((len(names), len(mw.params))); [R.__setitem__((i, list(mw.params.index).index(n)), 1) for i, n in enumerate(names)]
    w = mw.wald_test(R, scalar=True)
    rates = d.groupby("K").y.mean()
    L += ["", f"Operator given size, cluster-robust Wald test: χ² = {float(w.statistic):.0f} on {len(names)} df, p = {float(w.pvalue):.1e}.", "",
          "Operator given size (likelihood ratio, K + C(op) against K; runs treated as independent): "
          + "; ".join(f"{n}: χ² = {c:.0f} on {k} df" for n, (c, k) in lr.items()) + ".", "",
          f"Exact recovery by size (all operators): " + ", ".join(f"K={int(k)}: {v:.2f}" for k, v in rates.items()) + ".", ""]
    values.record("size_vs_operator", dict(r2_size=r2[("all seven", "K")], r2_op=r2[("all seven", "C(op)")],
                                           r2_both=r2[("all seven", "K + C(op)")], lr_chi2=lr["all seven"][0], lr_df=lr["all seven"][1],
                                           wald_chi2=float(w.statistic), wald_df=len(names), wald_p=float(w.pvalue),
                                           size_over_op=r2[("all seven", "K")] / r2[("all seven", "C(op)")]))
    out = ROOT / "paper/results/size_vs_operator.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
