"""Review point m6 (post hoc, existing data): does the single K50 summary hide differences in curve shape, and does
it depend on the logistic form? full_gp scout runs, corrected scoring. Per operator: logistic K50 and slope
(target bootstrap, B = 1000); nonparametric K50 (isotonic regression of the per-size success rate, interpolated
at 0.5). -> paper/results/curve_shapes.md"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.k50 import logit
from reach.family import FAMILY
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B, RNG = 1000, np.random.default_rng(20261016)


def iso_k50(K, y):
    ks = np.unique(K)
    r = np.array([y[K == k].mean() for k in ks]); w = np.array([(K == k).sum() for k in ks], float)
    vals, wts, idx = list(r), list(w), [[i] for i in range(len(r))]       # pool-adjacent-violators, decreasing
    i = 0
    while i < len(vals) - 1:
        if vals[i] < vals[i + 1]:
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            vals[i:i + 2], wts[i:i + 2], idx[i:i + 2] = [v], [wts[i] + wts[i + 1]], [idx[i] + idx[i + 1]]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty(len(r))
    for v, ii in zip(vals, idx):
        fit[ii] = v
    for j in range(len(ks) - 1):
        if fit[j] >= 0.5 > fit[j + 1]:
            return float(ks[j] + (fit[j] - 0.5) / (fit[j] - fit[j + 1]) * (ks[j + 1] - ks[j]))
    return float("nan")


def main():
    flips = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    by = defaultdict(lambda: defaultdict(list))
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") == "conf":
            by[r["grammar"]][r["target_id"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in flips)))
    L = ["# Review m6 (post hoc): curve shapes and a nonparametric K50", "",
         "| Operator | logistic K50 [95% CI] | logistic slope per node [95% CI] | isotonic K50 [95% CI] |", "|---|---|---|---|"]
    P = {}
    for g in FAMILY:
        ts = list(by[g].values())
        def est(sample):
            K = np.array([k for t in sample for k, _ in t]); y = np.array([v for t in sample for _, v in t])
            b = logit(K, y)
            return -b[0] / b[1], b[1], iso_k50(K, y)
        p = est(ts)
        P[g] = p
        reps = np.array([est([ts[i] for i in RNG.integers(0, len(ts), len(ts))]) for _ in range(B)])
        ci = lambda j: np.nanquantile(reps[:, j], [0.025, 0.975])
        L.append(f"| {g} | {p[0]:.2f} [{ci(0)[0]:.2f}, {ci(0)[1]:.2f}] | {p[1]:+.3f} [{ci(1)[0]:+.3f}, {ci(1)[1]:+.3f}] | "
                 f"{p[2]:.2f} [{ci(2)[0]:.2f}, {ci(2)[1]:.2f}] |")
    out = ROOT / "paper/results/curve_shapes.md"
    lg = {g: p[0] for g, p in P.items()}; iso = {g: p[2] for g, p in P.items()}
    # added (D039): the key effects under the isotonic estimate as well as the logistic one
    def pts(phase, exp, ops):
        fl = set(json.loads((ROOT / f"data/rescored/{phase}.json").read_text())) if (ROOT / f"data/rescored/{phase}.json").exists() else set()
        rs = [r for k, r in fixed.results(phase).items() if r.get("exp") == exp and r["grammar"] in ops]
        keys = [k for k, r in fixed.results(phase).items() if r.get("exp") == exp and r["grammar"] in ops]
        return (np.array([float(r["K"]) for r in rs]),
                np.array([float(r.get("exact_hash_match") == "true" or k in fl) for k, r in zip(keys, rs)]))
    six = [g for g in FAMILY if g != "F-EDL"]
    iso_eff = dict(crossover=iso_k50(*pts("full_gp", "conf", six)) - iso_k50(*pts("nox_gp", "confB", six)),
                   edl=iso_k50(*pts("full_gp", "conf", ["F-EDL"])) - iso_k50(*pts("full_gp", "conf", six)),
                   edl_scaled=iso_k50(*pts("pen_gp", "confP", ["F-EDL"])) - iso_k50(*pts("pen_gp", "confP", six)),
                   spread=max(iso[g] for g in six) - min(iso[g] for g in six))
    iso_eff["edl_share"] = 1 - iso_eff["edl_scaled"] / iso_eff["edl"]

    # added after the results (independent check, D048): H1 and the crossover spread under the isotonic K50, and the
    # held-out grammars. Targets resampled jointly in the two arms (B = 1000), so the spread difference is paired.
    def by_target(phase, exp):
        p_ = ROOT / f"data/rescored/{phase}.json"
        fl = set(json.loads(p_.read_text())) if p_.exists() else set()
        res = defaultdict(lambda: defaultdict(list))
        for k, r in fixed.results(phase).items():
            if r.get("exp") == exp:
                res[r["grammar"]][r["target_id"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl)))
        return res

    def iso_of(data, ops, sel):
        pts = np.array([q for g in ops for t in sel[g] for q in data[g][t]])
        return iso_k50(pts[:, 0], pts[:, 1])
    gp_t, nox_t = by_target("full_gp", "conf"), by_target("nox_gp", "confB")
    tgs = {g: sorted(set(gp_t[g]) & set(nox_t[g])) for g in six}
    spread_of = lambda data, sel: (lambda k: np.nanmax(k) - np.nanmin(k))([iso_of(data, [g], sel) for g in six])
    rng = np.random.default_rng(20261106)
    sp_reps, h1_reps = [], []
    for _ in range(B):
        s_ = {g: [tgs[g][i] for i in rng.integers(0, len(tgs[g]), len(tgs[g]))] for g in six}
        sp_reps.append(spread_of(nox_t, s_) - spread_of(gp_t, s_))
        pool = iso_of(gp_t, six, s_)
        h1_reps.append([iso_of(gp_t, [g], s_) - pool for g in six])
    iso_eff["spread_gp"], iso_eff["spread_nox"] = spread_of(gp_t, tgs), spread_of(nox_t, tgs)
    iso_eff["spread_diff_lo"], iso_eff["spread_diff_hi"] = (float(x) for x in np.nanquantile(sp_reps, [0.025, 0.975]))
    pool0 = iso_of(gp_t, six, tgs)
    h1 = {}
    for j, g in enumerate(six):
        lo_, hi_ = np.nanquantile(np.array(h1_reps)[:, j], [0.05, 0.95])
        h1[g[2:]] = dict(d=iso_of(gp_t, [g], tgs) - pool0, lo=float(lo_), hi=float(hi_), inside=bool(lo_ >= -1.5 and hi_ <= 1.5))
    iso_eff["h1"] = h1
    iso_eff["h1_outside"] = sum(not x["inside"] for x in h1.values())
    iso_eff["h1_holds"] = iso_eff["h1_outside"] == 0
    hd = by_target("heldout_gp", "conf")
    hd_iso, hd_log, hd_slope = {}, {}, {}
    for g in sorted(hd):
        pts = np.array([q for t in hd[g] for q in hd[g][t]])
        b_ = logit(pts[:, 0], pts[:, 1])
        hd_iso[g], hd_log[g], hd_slope[g] = iso_k50(pts[:, 0], pts[:, 1]), -b_[0] / b_[1], b_[1]
    iso_eff["heldout"] = hd_iso
    iso_eff["heldout_same_order"] = sorted(hd, key=hd_log.get) == sorted(hd, key=hd_iso.get)
    nox_iso, nox_slope = {}, {}
    for g in FAMILY:
        pts = np.array([q for t in nox_t[g] for q in nox_t[g][t]])
        b_ = logit(pts[:, 0], pts[:, 1])
        nox_iso[g], nox_slope[g] = iso_k50(pts[:, 0], pts[:, 1]), b_[1]
    iso_eff["nox"] = {g[2:]: nox_iso[g] for g in FAMILY}
    # EDL's share with the isotonic K50, targets resampled jointly in the two arms
    pen_t = by_target("pen_gp", "confP")
    tgj = {g: sorted(set(gp_t[g]) & set(pen_t[g])) for g in FAMILY}
    gap = lambda data, s_: iso_of(data, ["F-EDL"], s_) - iso_of(data, six, s_)
    rs_ = np.random.default_rng(20261107)
    sh = []
    for _ in range(B):
        s_ = {g: [tgj[g][i] for i in rs_.integers(0, len(tgj[g]), len(tgj[g]))] for g in FAMILY}
        sh.append(1 - gap(pen_t, s_) / gap(gp_t, s_))
    iso_eff["edl_share_lo"], iso_eff["edl_share_hi"] = (float(x) for x in np.nanquantile(sh, [0.025, 0.975]))
    # does the ES depend more on the operator than GP? Spread of the six under each estimator and penalty; each arm's own
    # targets (the ES covers sizes 3-15), resampled separately
    def log_of(data, ops, sel):
        pts = np.array([q for g in ops for t in sel[g] for q in data[g][t]])
        b0 = logit(pts[:, 0], pts[:, 1])
        return -b0[0] / b0[1] if b0[1] < 0 else float("nan")
    arms = dict(fixed=(gp_t, by_target("full_es", "es")), scaled=(pen_t, by_target("pen_es", "esP")))
    spread_by = lambda est, data, sel: (lambda k: np.nanmax(k) - np.nanmin(k))([est(data, [g], sel) for g in six])
    es_out, rs2 = {}, np.random.default_rng(20261108)
    for pen, (ga, ea) in arms.items():
        tga, tea = {g: sorted(ga[g]) for g in six}, {g: sorted(ea[g]) for g in six}
        for name, est in (("log", log_of), ("iso", iso_of)):
            reps_ = []
            for _ in range(B):
                sa = {g: [tga[g][i] for i in rs2.integers(0, len(tga[g]), len(tga[g]))] for g in six}
                se = {g: [tea[g][i] for i in rs2.integers(0, len(tea[g]), len(tea[g]))] for g in six}
                reps_.append(spread_by(est, ea, se) - spread_by(est, ga, sa))
            lo_, hi_ = np.nanquantile(reps_, [0.025, 0.975])
            es_out[f"{pen}_{name}"] = dict(es=spread_by(est, ea, tea), gp=spread_by(est, ga, tga), lo=float(lo_), hi=float(hi_))
    iso_eff["_es"] = es_out
    es_out = iso_eff.pop("_es")
    L += ["", "Key effects with the isotonic K₅₀ (nodes): crossover (pooled six, GP − GP without crossover) "
          f"{iso_eff['crossover']:+.2f}; EDL − pooled six {iso_eff['edl']:+.2f} (scaled penalty {iso_eff['edl_scaled']:+.2f}), so the "
          f"scaled penalty removes {iso_eff['edl_share']:.0%} of the deficit (95% CI {iso_eff['edl_share_lo']:.0%}–{iso_eff['edl_share_hi']:.0%}); spread of the six {iso_eff['spread_gp']:.2f} with crossover and "
          f"{iso_eff['spread_nox']:.2f} without (difference 95% CI {iso_eff['spread_diff_lo']:+.2f} to {iso_eff['spread_diff_hi']:+.2f}).", "",
          "H1's test applied with the isotonic K₅₀ (difference from the pooled six, 90% interval): "
          + "; ".join(f"{g} {x['d']:+.2f} [{x['lo']:+.2f}, {x['hi']:+.2f}]" for g, x in h1.items())
          + f". {iso_eff['h1_outside']} of the six leave ±1.5; this estimator's intervals are wider than the logistic ones.", "",
          "Held-out grammars, random targets: "
          + "; ".join(f"{g} isotonic K₅₀ {hd_iso[g]:.2f} (logistic {hd_log[g]:.2f}, slope {hd_slope[g]:+.2f})" for g in sorted(hd)) + ".", "",
          "GP without crossover: "
          + "; ".join(f"{g[2:]} isotonic K₅₀ {nox_iso[g]:.2f} (slope {nox_slope[g]:+.2f})" for g in FAMILY) + ".", "",
          "Spread of the six, (1+4)-ES against GP (95% CI of the difference, each arm's own targets): "
          + "; ".join(f"{dict(log='logistic', iso='isotonic')[k.split('_')[1]]} K₅₀, {k.split('_')[0]} penalty {x['es']:.2f} against {x['gp']:.2f} ({x['lo']:+.2f} to {x['hi']:+.2f})" for k, x in es_out.items()) + ".", ""]
    out.write_text("\n".join(L) + "\n")
    values.record("curve_shapes", dict(slope_min=min(p[1] for p in P.values()), slope_max=max(p[1] for p in P.values()),
                                       max_iso_gap=max(abs(lg[g] - iso[g]) for g in FAMILY),
                                       same_order=sorted(FAMILY, key=lg.get) == sorted(FAMILY, key=iso.get), iso=iso_eff, es=es_out))
    print("\n".join(L))


if __name__ == "__main__":
    main()
