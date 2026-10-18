"""Post hoc: uncertainty of the registered H6 result and its attainable ceiling.
- 95% intervals for the held-out AUC and calibration slope: bootstrap of targets within each grammar (B = 1000).
- AUC with ties averaged: every run of a target shares one prediction, and the registered AUC function (no tie
  handling) varies slightly with run order.
- Ceiling: the AUC a perfectly calibrated predictor would reach on these targets, from outcomes simulated as
  Bernoulli(frozen prediction) for every run (300 simulations).
- Where the discrimination comes from: AUC within size classes (only pairs of runs on targets of the same shortest
  size), AUC and mean predicted against observed per grammar, and two baselines on the same runs: shortest size
  alone (ranking by -K) and an oracle that adds a grammar intercept to the frozen logit, fitted on the held-out
  runs themselves (in-sample, so an upper reference, not a predictor).
-> paper/results/h6_intervals.md"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import rankdata

from reach.analysis.k50 import logit
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B, SIMS, RNG = 1000, 300, np.random.default_rng(20261017)


def auc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def slope(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    return float(logit(lp, y)[1])


def main():
    pred = json.loads((ROOT / "data/heldout/predictions.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    for r in fixed.results("heldout_gp").values():
        if r.get("exp") in ("conf", "calc", "calc2"):
            by[r["grammar"].replace("S2-", "S-")][r["target_id"]].append(float(r.get("exact_hash_match") == "true"))
    tg = {g: sorted(t) for g, t in by.items()}

    def arrays(sel):
        y = np.array([v for g in sel for t in sel[g] for v in by[g][t]])
        p = np.array([pred[t]["p"] for g in sel for t in sel[g] for _ in by[g][t]])
        return y, p
    y, p = arrays(tg)
    reps = []
    for _ in range(B):
        yb, pb = arrays({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in tg})
        reps.append((auc(yb, pb), slope(yb, pb)))
    reps = np.array(reps)
    ceil = np.array([auc(ys, p) for ys in (RNG.random((SIMS, len(p))) < p).astype(float)])
    q = lambda v: np.quantile(v, [0.025, 0.975])
    L = ["# Post hoc: H6 intervals and ceiling", "",
         f"{len(y)} runs on {sum(len(t) for t in tg.values())} targets, four held-out grammars.", "",
         "| Quantity | value | 95% interval |", "|---|---|---|",
         f"| AUC (ties averaged) | {auc(y, p):.4f} | {q(reps[:, 0])[0]:.3f}–{q(reps[:, 0])[1]:.3f} (target bootstrap) |",
         f"| calibration slope | {slope(y, p):.3f} | {q(reps[:, 1])[0]:.3f}–{q(reps[:, 1])[1]:.3f} (target bootstrap) |",
         f"| AUC of a perfectly calibrated predictor (ceiling) | {ceil.mean():.3f} | {q(ceil)[0]:.3f}–{q(ceil)[1]:.3f} ({SIMS} simulations) |", ""]
    rows = pd.DataFrame([dict(g=g, t=t, K=pred[t]["K"], y=v, p=pred[t]["p"]) for g in tg for t in tg[g] for v in by[g][t]])
    rows["lp"] = np.log(rows.p / (1 - rows.p))

    def within_k(d):
        num = den = 0.0
        for _, s in d.groupby("K"):
            n1 = s.y.sum(); n0 = len(s) - n1
            if n1 and n0:
                num += auc(s.y.values, s.p.values) * n1 * n0; den += n1 * n0
        return num / den
    orc = smf.glm("y ~ C(g) - 1", rows, family=sm.families.Binomial(), offset=rows.lp).fit()
    L += ["Where the discrimination comes from (same runs):", "", "| Ranking | AUC |", "|---|---|",
          f"| shortest size alone (−K) | {auc(rows.y.values, -rows.K.values.astype(float)):.3f} |",
          f"| frozen predictor | {auc(rows.y.values, rows.p.values):.3f} |",
          f"| frozen predictor plus a grammar intercept fitted on these runs (in-sample oracle) | {auc(rows.y.values, orc.predict(rows, offset=rows.lp).values):.3f} |",
          f"| frozen predictor, within size classes only | {within_k(rows):.3f} |", "",
          "| Grammar | runs | AUC | mean predicted | observed |", "|---|---|---|---|---|"]
    V = dict(auc=auc(y, p), auc_lo=q(reps[:, 0])[0], auc_hi=q(reps[:, 0])[1], slope=slope(y, p), slope_lo=q(reps[:, 1])[0],
             slope_hi=q(reps[:, 1])[1], ceiling=ceil.mean(), size_only=auc(rows.y.values, -rows.K.values.astype(float)),
             within_k=within_k(rows), oracle=auc(rows.y.values, orc.predict(rows, offset=rows.lp).values),
             mean_pred=float(rows.p.mean()), mean_obs=float(rows.y.mean()), runs=len(rows), targets=int(rows.t.nunique()))
    for g, s in rows.groupby("g"):
        L.append(f"| {g} | {len(s)} | {auc(s.y.values, s.p.values):.3f} | {s.p.mean():.3f} | {s.y.mean():.3f} |")
        V[g] = dict(pred=float(s.p.mean()), obs=float(s.y.mean()))
    V["two_var_runs"] = int(sum(len(by[g][t]) for g in by for t in by[g] if pred[t].get("nvar")))
    V["two_var_targets"] = int(sum(1 for g in by for t in by[g] if pred[t].get("nvar")))
    V["max_grammar_miss"] = max(abs(x["obs"] - x["pred"]) for k, x in V.items() if isinstance(x, dict))
    values.record("h6_intervals", V)
    L.append("")
    (ROOT / "paper/results/h6_intervals.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
