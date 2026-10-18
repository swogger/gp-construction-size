"""Post hoc: the fixed penalty for undefined values and EDL's deficit.
The error (Section 3.4) is the MSE divided by the target's variance, with a fixed 1,000 per non-finite residual, so a
tree undefined at every point scores 1000/var: below a constant's error (about 1) when var > 1000.
- Share of random targets with var > 1000 per operator.
- Failed runs on those targets whose best tree ended undefined at every validation point (final error = 1000/var).
- deficit(): EDL - pooled six (K50) on a subset of targets with a 95% target bootstrap (B = 1000), used by
  sensitivity.py for the table of deficits under both penalties (supplement S8).
- Recovery against var > 1000, controlling for shortest size and operator (GP; cluster-robust by target).
-> paper/results/edl_artefact.md"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from reach.analysis.family_full import _k50
from reach.family import FAMILY
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B, SEED, LIMIT = 1000, 20261017, 1000.0
SIX = [g for g in FAMILY if g != "F-EDL"]
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def runs(phase, exp):
    p = ROOT / f"data/rescored/{phase}.json"
    fl = set(json.loads(p.read_text())) if p.exists() else set()
    return [dict(g=r["grammar"], t=r["target_id"], K=float(r["K"]), y=float(r.get("exact_hash_match") == "true" or k in fl),
                 e=r.get("final_nmse")) for k, r in fixed.results(phase).items() if r.get("exp") == exp]


def deficit(R, keep, seed=SEED):
    """EDL - pooled six K50 on the targets kept, with a 95% target bootstrap (own RNG, so every caller agrees)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(lambda: defaultdict(list))
    for r in R:
        if keep(r["t"]):
            by[r["g"]][r["t"]].append((r["K"], r["y"]))
    tg = {g: sorted(by[g]) for g in FAMILY}

    def d(sel):
        pts = lambda ops: np.array([p for g in ops for t in sel[g] for p in by[g][t]])
        e, s = pts(["F-EDL"]), pts(SIX)
        return _k50(e[:, 0], e[:, 1]) - _k50(s[:, 0], s[:, 1])
    reps = np.array([d({g: [tg[g][i] for i in rng.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY}) for _ in range(B)])
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return d(tg), lo, hi


def main():
    var = {t["target_id"]: t["var_target"] for g in FAMILY
           for t in json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())}
    high = lambda t: var[t] > LIMIT
    gp, nox, es = runs("full_gp", "conf"), runs("nox_gp", "confB"), runs("full_es", "es")

    def undefined(R, g):
        f = [r for r in R if r["g"] == g and not r["y"] and high(r["t"])]
        n = sum(1 for r in f if r["e"] not in (None, "") and np.isclose(float(r["e"]), LIMIT / var[r["t"]], rtol=1e-6))
        return n, len(f)
    L = ["# Post hoc: the fixed penalty for undefined values and EDL's deficit", "",
         "A tree undefined at every point scores 1000/var; for var > 1000 that is below the error of a constant.", "",
         "| Operator | random targets with var > 1000 | GP: failed runs on them ending undefined everywhere | GP without crossover: same |",
         "|---|---|---|---|"]
    V = {}
    for g in FAMILY:
        v = [high(t) for t in var if t.startswith(g + "_c_")]
        (a, b), (c, d) = undefined(gp, g), undefined(nox, g)
        V[g[2:]] = dict(share_high=float(np.mean(v)), undef_gp=a, fail_gp=b, undef_nox=c, fail_nox=d)
        L.append(f"| {NAMES[g]} | {100 * np.mean(v):.1f}% ({sum(v)} of {len(v)}) | {a} of {b} | {c} of {d} |")
    df = pd.DataFrame([dict(op=r["g"], tid=r["t"], K=r["K"], hi=float(high(r["t"])), y=r["y"]) for r in gp])
    m = smf.glm("y ~ C(op) + K + hi", df, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df.tid)[0]})
    L += ["", f"GP, all seven operators: var > 1000 changes the log-odds of recovery by {m.params['hi']:+.2f} "
          f"(cluster-robust SE {m.bse['hi']:.2f}) at equal shortest size and operator, against {m.params['K']:+.3f} per node.", ""]
    (ROOT / "paper/results/edl_artefact.md").write_text("\n".join(L) + "\n")
    others = [V[g[2:]]["share_high"] for g in FAMILY if g not in ("F-EDL", "F-PLM")]
    held = {}
    for h in sorted((ROOT / "data/targets").glob("S-*")):
        f = h / "targets_conf.json"
        if f.exists():
            vv = [t["var_target"] > LIMIT for t in json.loads(f.read_text())]
            held[h.name] = float(np.mean(vv))
    L += [f"Held-out grammars (not rerun with the scaled penalty): share of random targets with var > 1000 "
          + ", ".join(f"{k} {100 * x:.1f}%" for k, x in held.items()) + ".", ""]
    (ROOT / "paper/results/edl_artefact.md").write_text("\n".join(L) + "\n")
    V.update(heldout_high_min=min(held.values()), heldout_high_max=max(held.values()))
    V.update(share_high_others_max=max(others), n_ops_no_high=sum(V[g[2:]]["share_high"] == 0 for g in FAMILY),
             logodds_high=m.params["hi"], logodds_high_se=m.bse["hi"])
    values.record("edl_artefact", V)
    print("\n".join(L))


if __name__ == "__main__":
    main()
