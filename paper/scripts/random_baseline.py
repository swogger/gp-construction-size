"""Post hoc (D038): random sampling as a baseline, and phenotype frequency as a rival to shortest size.
Specified before the rand_sample runs. Trees are drawn exactly as GP's initial population (rand_sample, 10^6 per
operator). For each random target t: frequency f_t = hits / trees; a "run" of E0 = 2 x 10^4 random trees finds t with
probability 1 - (1 - f_t)^E0. Compared with GP at E0 (full_gp, corrected scoring, fixed engine) on the same targets:
- K50 of random sampling and of GP, per operator and pooled (binomial GLM on per-target probabilities);
- the share of GP's successes on targets that random sampling would find with probability >= 1/2;
- frequency against shortest size, and whether log frequency predicts GP's recovery beyond shortest size (logistic,
  cluster-robust by target; log10 of f_t + 1/trees, so that unseen targets enter at the detection limit).
-> paper/results/random_baseline.md, values"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from reach.family import FAMILY
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
E0 = 20_000
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def k50(df, col):
    if df[col].min() == df[col].max():
        return float("nan")
    m = sm.GLM(df[col].values, sm.add_constant(df.K.values.astype(float)), family=sm.families.Binomial()).fit()
    return float(-m.params[0] / m.params[1]) if m.params[1] < 0 else float("nan")


def analyse(hits, trees, gp):
    """hits: {grammar: {target: count}}; trees: {grammar: n}; gp: {target: [0/1 outcomes]}."""
    rows = []
    for g in FAMILY:
        for t in json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text()):
            tid = t["target_id"]
            if tid not in gp:
                continue
            f = hits[g].get(tid, 0) / trees[g]
            rows.append(dict(g=g, t=tid, K=t["K_construction"], f=f, logf=np.log10(f + 1 / trees[g]),
                             p_rand=1 - (1 - f) ** E0, p_gp=float(np.mean(gp[tid])), n_gp=len(gp[tid])))
    df = pd.DataFrame(rows)
    per = {g: (k50(df[df.g == g], "p_rand"), k50(df[df.g == g], "p_gp")) for g in FAMILY}
    pooled = (k50(df, "p_rand"), k50(df, "p_gp"))
    easy = df.p_rand >= 0.5
    share_easy = float((df.p_gp * df.n_gp)[easy].sum() / (df.p_gp * df.n_gp).sum())
    long = pd.DataFrame([dict(t=r.t, g=r.g, K=r.K, logf=r.logf, y=y) for r in df.itertuples() for y in gp[r.t]])
    m = smf.glm("y ~ C(g) + K + logf", long, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(long.t)[0]})
    seen = df[df.f > 0]
    rho = spearmanr(seen.K, seen.logf).correlation if len(seen) > 2 else float("nan")
    return df, per, pooled, share_easy, m, rho


def main():
    hits, trees = defaultdict(dict), defaultdict(int)
    for r in fixed.results("rand_sample").values():
        trees[r["grammar"]] += int(r["n"])
        for t, c in r["hits"].items():
            hits[r["grammar"]][t] = hits[r["grammar"]].get(t, 0) + c
    fl = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    gp = defaultdict(list)
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") == "conf":
            gp[r["target_id"]].append(float(r.get("exact_hash_match") == "true" or k in fl))
    df, per, pooled, share_easy, m, rho = analyse(hits, trees, gp)
    # added (D040): each search against random sampling on the same targets (pooled six), target bootstrap B = 1000
    def outcomes(ph, exp):
        p = ROOT / f"data/rescored/{ph}.json"
        f = set(json.loads(p.read_text())) if p.exists() else set()
        d = defaultdict(list)
        for k, r in fixed.results(ph).items():
            if r.get("exp") == exp:
                d[r["target_id"]].append(float(r.get("exact_hash_match") == "true" or k in f))
        return d
    rng, vs = np.random.default_rng(20261016), {}
    for name, ph, exp in (("gp", "full_gp", "conf"), ("nox", "nox_gp", "confB"), ("es", "full_es", "es")):
        dd = analyse(hits, trees, outcomes(ph, exp))[0]
        dd = dd[dd.g != "F-EDL"]
        diff = lambda x: k50(x, "p_gp") - k50(x, "p_rand")
        groups = {g: dd[dd.g == g] for g in dd.g.unique()}
        reps = [diff(pd.concat([x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31))) for x in groups.values()]))
                for _ in range(1000)]
        vs[name] = dict(k50=k50(dd, "p_gp"), rand=k50(dd, "p_rand"), d=diff(dd), lo=float(np.nanquantile(reps, 0.025)),
                        hi=float(np.nanquantile(reps, 0.975)), targets=len(dd))
    six = [g for g in FAMILY if g != "F-EDL"]
    rand_spread = max(per[g][0] for g in six) - min(per[g][0] for g in six)
    L = ["# Post hoc: random sampling as a baseline; phenotype frequency", "",
         f"Trees drawn as GP's initial population: {', '.join(f'{NAMES[g]} {trees[g]:,}' for g in FAMILY)}. "
         f"A random-sampling run draws E0 = {E0:,} trees.", "",
         "| Operator | targets seen at least once | K₅₀, random sampling at E0 | K₅₀, GP at E0 |", "|---|---|---|---|"]
    L += [f"| {NAMES[g]} | {int((df[df.g == g].f > 0).sum())} of {int((df.g == g).sum())} | {per[g][0]:.2f} | {per[g][1]:.2f} |" for g in FAMILY]
    L += [f"| pooled | {int((df.f > 0).sum())} of {len(df)} | {pooled[0]:.2f} | {pooled[1]:.2f} |", "",
          f"Share of GP's successes on targets that random sampling finds with probability ≥ 1/2 at E0: {share_easy:.0%}.", "",
          f"Frequency against shortest size (targets seen at least once): Spearman ρ = {rho:+.2f}.",
          f"GP recovery ~ operator + K + log10 frequency (cluster-robust): K {m.params['K']:+.3f} (z = {m.tvalues['K']:.1f}), "
          f"log10 frequency {m.params['logf']:+.3f} (z = {m.tvalues['logf']:.1f}). {int((df.f == 0).sum())} of {len(df)} targets "
          "were never seen and enter at the detection limit, so the frequency result rests mainly on small targets.", "",
          "Each search against random sampling on the same targets (six non-EDL operators; K₅₀; 95% target bootstrap):", "",
          "| Search | targets | K₅₀ of the search | K₅₀ of random sampling | difference |", "|---|---|---|---|---|"]
    L += [f"| {lab} | {vs[k]['targets']} | {vs[k]['k50']:.2f} | {vs[k]['rand']:.2f} | {vs[k]['d']:+.2f} [{vs[k]['lo']:+.2f}, {vs[k]['hi']:+.2f}] |"
          for k, lab in (("gp", "GP"), ("nox", "GP without crossover"), ("es", "(1+4)-ES (sizes 3–15)"))]
    L += ["", f"Spread of random sampling's K₅₀ among the six: {rand_spread:.2f} nodes.", ""]
    (ROOT / "paper/results/random_baseline.md").write_text("\n".join(L) + "\n")
    values.record("random_baseline", dict(k50_rand=pooled[0], k50_gp=pooled[1], gap=pooled[1] - pooled[0], share_easy=share_easy,
                                          rho=rho, coef_K=m.params["K"], z_K=m.tvalues["K"], coef_logf=m.params["logf"],
                                          z_logf=m.tvalues["logf"], unseen=float((df.f == 0).mean()), vs=vs,
                                          rand_spread=rand_spread, per={g[2:]: dict(rand=a, gp=b) for g, (a, b) in per.items()}))
    print("\n".join(L))


if __name__ == "__main__":
    main()
