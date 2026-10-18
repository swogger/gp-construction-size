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
    out.write_text("\n".join(L) + "\n")
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
    L += ["", "Key effects with the isotonic K₅₀ (nodes): crossover (pooled six, GP − GP without crossover) "
          f"{iso_eff['crossover']:+.2f}; EDL − pooled six {iso_eff['edl']:+.2f} (scaled penalty {iso_eff['edl_scaled']:+.2f}); "
          f"spread of the six {iso_eff['spread']:.2f}."]
    values.record("curve_shapes", dict(slope_min=min(p[1] for p in P.values()), slope_max=max(p[1] for p in P.values()),
                                       max_iso_gap=max(abs(lg[g] - iso[g]) for g in FAMILY),
                                       same_order=sorted(FAMILY, key=lg.get) == sorted(FAMILY, key=iso.get), iso=iso_eff))
    print("\n".join(L))


if __name__ == "__main__":
    main()
