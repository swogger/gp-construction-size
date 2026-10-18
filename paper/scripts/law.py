"""Post hoc: one equation for recovery as a function of shortest size and budget.
logit P(recover) = a_g - beta * K + gamma * log2(E/E0), fitted on the like-for-like budget data (same targets at
E0, 4E0, 16E0; seven operators, sizes 9-17; corrected scoring), with operator intercepts a_g, a common size slope
beta and a common budget slope gamma; cluster-robust SEs by target. Then s = gamma / beta (nodes per doubling) and
one node = 2^(1/s) times the budget. Also the pooled fit without operator intercepts, and a fit with operator-specific
beta and gamma (per-operator rates; Wald test of equal gamma, cluster-robust). -> paper/results/law.md"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import size_wall  # noqa: E402
import values  # noqa: E402

from reach.family import FAMILY


def main():
    d = size_wall.outcomes()
    tg = {g: {t for (o, t, m) in d if o == g and m == 16} for g in FAMILY}
    rows = [dict(op=o, tid=t, K=int(t.split("_K")[1][:2]), l2=float(np.log2(m)), y=y)
            for (o, t, m), ys in d.items() if m in (1, 4, 16) and t in tg[o] for y in ys]
    df = pd.DataFrame(rows)
    L = ["# Post hoc: one equation for recovery against shortest size and budget", "",
         f"Like-for-like budget data: {len(df)} runs, {df.tid.nunique()} targets, budgets E0, 4E0, 16E0; corrected scoring.", "",
         "| Model | β (per node) | γ (per doubling) | s = γ/β (nodes per doubling) | one node = 2^(1/s) | McFadden R² |", "|---|---|---|---|---|---|"]
    null = smf.glm("y ~ 1", df, family=sm.families.Binomial()).fit().deviance
    V = dict(runs=len(df), targets=df.tid.nunique())
    for name, f in (("pooled", "y ~ K + l2"), ("operator intercepts", "y ~ C(op) + K + l2")):
        m = smf.glm(f, df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.tid)[0]})
        b, g = -m.params["K"], m.params["l2"]
        cov = m.cov_params().loc[["K", "l2"], ["K", "l2"]].values
        draws = np.random.default_rng(1).multivariate_normal([m.params["K"], g], cov, 20000)
        sd = draws[:, 1] / -draws[:, 0]
        lo, hi = np.quantile(sd, [0.025, 0.975])
        V["pooled" if name == "pooled" else "joint"] = dict(beta=b, beta_se=m.bse["K"], gamma=g, gamma_se=m.bse["l2"], s=g / b,
                                                            s_lo=lo, s_hi=hi, fold=2 ** (b / g), fold_lo=2 ** (1 / hi), fold_hi=2 ** (1 / lo))
        L.append(f"| {name} | {b:.3f} ± {m.bse['K']:.3f} | {g:.3f} ± {m.bse['l2']:.3f} | {g / b:.3f} [{lo:.3f}, {hi:.3f}] | "
                 f"{2 ** (b / g):.1f}× [{2 ** (1 / hi):.1f}, {2 ** (1 / lo):.1f}] | {1 - m.deviance / null:.3f} |")
    m = smf.glm("y ~ C(op) + C(op):K + C(op):l2 - 1", df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.tid)[0]})
    L += ["", "Per operator (operator-specific β and γ):", "", "| Operator | β | γ | s = γ/β | one node = 2^(1/s) |", "|---|---|---|---|---|"]
    for o in FAMILY:
        b, g = -m.params[f"C(op)[{o}]:K"], m.params[f"C(op)[{o}]:l2"]
        L.append(f"| {o[2:]} | {b:.3f} | {g:.3f} | {g / b:.2f} | {2 ** (b / g):.0f}× |")
        V.setdefault("op", {})[o[2:]] = dict(s=g / b, fold=2 ** (b / g))
    names = list(m.params.index); R = np.zeros((len(FAMILY) - 1, len(names)))
    for i, (a, c) in enumerate(zip(FAMILY, FAMILY[1:])):
        R[i, names.index(f"C(op)[{a}]:l2")], R[i, names.index(f"C(op)[{c}]:l2")] = 1, -1
    w = m.wald_test(R, scalar=True)
    L += ["", f"Equal γ across operators: Wald χ² = {float(w.statistic):.1f} on {len(FAMILY) - 1} df, p = {float(w.pvalue):.3f} (cluster-robust)."]
    (ROOT / "paper/results/law.md").write_text("\n".join(L) + "\n")
    ss = {o: x["s"] for o, x in V["op"].items()}
    V.update(wald_p=float(w.pvalue), op_s_min=min(ss.values()), op_s_max=max(ss.values()),
             op_fold_min=min(x["fold"] for x in V["op"].values()), op_fold_max=max(x["fold"] for x in V["op"].values()))
    ex = values.load()                       # needs exchange_rate.py and size_wall.py to have run first
    from scipy.stats import spearmanr
    reach = {o: ex[f"size_wall.A.k50.{o}.1"] for o in V["op"]}
    V["rho_s_reach"] = spearmanr([V["op"][o]["s"] for o in V["op"]], [reach[o] for o in V["op"]]).correlation
    V["rho_s_reach_p"] = spearmanr([V["op"][o]["s"] for o in V["op"]], [reach[o] for o in V["op"]]).pvalue
    L.append(f"Per-operator rate against reach at E0 (set A): Spearman ρ = {V['rho_s_reach']:+.2f} (p = {V['rho_s_reach_p']:.3f}, n = 7).")
    (ROOT / "paper/results/law.md").write_text("\n".join(L) + "\n")
    V["edl_fold_own_rate"] = 2 ** (ex["exchange_rate.edl.nodes"] / V["op"]["EDL"]["s"])
    values.record("law", V)
    print("\n".join(L))


if __name__ == "__main__":
    main()
