"""Post hoc sensitivity analyses (D034; specified and committed before the runs).
1. Penalty scaled to the target: EDL - pooled six (K50) under GP, GP without crossover and the (1+4)-ES, with the
   penalty for undefined values scaled to the target (pen_gp, pen_nox, pen_es), beside the same quantity under
   EML-GP's fixed penalty (full_gp on the fixed engine, nox_gp, full_es) on the same targets and seeds; per-operator
   K50 under both penalties; failed runs ending undefined everywhere under the scaled penalty.
2. Headless-chicken crossover (hc_gp): pooled-six and per-operator K50 beside GP (full_gp, same targets, first
   two seeds) and GP without crossover (nox_gp, same targets, first two seeds); final tree sizes of failed runs
   under each (bloat). Added after the results: 95% intervals for the spread of the six operators' K50
   (max - min), subtree crossover against no crossover (five seeds) and against headless-chicken (two seeds).
Corrected scoring; 95% intervals from a bootstrap of targets within each operator (B = 1000).
Revised after the results (D036): the EDL deficits use edl_artefact.deficit (one seeded procedure, so S8 shows
one interval per estimate) and add the low-variance targets; H1's test and the crossover spread under the scaled
penalty were added.
-> paper/results/sensitivity.md"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.family_full import _k50
from reach.family import FAMILY
from reach.seeds import SEEDS
from reach.runner.phases import FULL_SEEDS
import edl_artefact as A  # noqa: E402
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
B, RNG = 1000, np.random.default_rng(20261016)
SIX = [g for g in FAMILY if g != "F-EDL"]
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def runs(phase, exp, seeds=None):
    p = ROOT / f"data/rescored/{phase}.json"
    fl = set(json.loads(p.read_text())) if p.exists() else set()
    by = defaultdict(lambda: defaultdict(list))
    for k, r in fixed.results(phase).items():
        if r.get("exp") == exp and (seeds is None or r["seed"] in seeds):
            by[r["grammar"]][r["target_id"]].append(
                (float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl), r.get("final_nmse")))
    return by


def k50(by, ops, sel):
    pts = np.array([p[:2] for g in ops for t in sel[g] for p in by[g][t]])
    return _k50(pts[:, 0], pts[:, 1])


def boot(by, stat):
    tg = {g: sorted(by[g]) for g in FAMILY}
    reps = np.array([stat({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY})
                     for _ in range(B)])
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return stat(tg), lo, hi


def fmt(v):
    return f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"


def main():
    var = {t["target_id"]: t["var_target"] for g in FAMILY
           for t in json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())}
    L = ["# Post hoc sensitivity analyses (D034, D036)", "",
         "## 1. Penalty for undefined values scaled to the target", "",
         "EDL − pooled six (K₅₀, nodes; 95% target-bootstrap interval), same targets and seeds:", "",
         "| Search | fixed penalty, all targets | scaled penalty, all targets | fixed penalty, var ≤ 1000 | scaled penalty, var ≤ 1000 |",
         "|---|---|---|---|---|"]
    low = lambda t: var[t] <= 1000
    V = {"deficit": {}, "k50": {}}
    for name, (pa, ea), (pb, eb) in (("GP", ("full_gp", "conf"), ("pen_gp", "confP")),
                                    ("GP without crossover", ("nox_gp", "confB"), ("pen_nox", "confBP")),
                                    ("(1+4)-ES", ("full_es", "es"), ("pen_es", "esP"))):
        ra, rb = A.runs(pa, ea), A.runs(pb, eb)
        cells = [A.deficit(r, k) for k in (lambda t: True, low) for r in (ra, rb)]
        V["deficit"][pa] = {lab: dict(d=c[0], lo=c[1], hi=c[2]) for lab, c in zip(("fixed", "scaled", "fixed_low", "scaled_low"), cells)}
        L.append(f"| {name} | " + " | ".join(fmt(c) for c in cells) + " |")
    pairs = [("GP", runs("full_gp", "conf"), runs("pen_gp", "confP")),
             ("GP without crossover", runs("nox_gp", "confB"), runs("pen_nox", "confBP")),
             ("(1+4)-ES", runs("full_es", "es"), runs("pen_es", "esP"))]
    succ = lambda by: np.mean([p[1] for t in by["F-EDL"] if var[t] > 1000 for p in by["F-EDL"][t]])
    L += ["", f"EDL's recovery rate on its targets with var > 1000 (GP): {succ(pairs[0][1]):.2f} with the fixed penalty, "
          f"{succ(pairs[0][2]):.2f} with the scaled penalty.", "",
          "K₅₀ per operator (fixed → scaled penalty):", "",
          "| Operator | GP | GP without crossover | (1+4)-ES |", "|---|---|---|---|"]
    for g in FAMILY:
        cells = []
        for _, a, b in pairs:
            ta, tb = {h: sorted(a[h]) for h in FAMILY}, {h: sorted(b[h]) for h in FAMILY}
            cells.append(f"{k50(a, [g], ta):.2f} → {k50(b, [g], tb):.2f}")
            V["k50"][f"{_}.{g[2:]}.fixed"], V["k50"][f"{_}.{g[2:]}.scaled"] = k50(a, [g], ta), k50(b, [g], tb)
        L.append(f"| {NAMES[g]} | " + " | ".join(cells) + " |")
    row = []
    for name, a, b in pairs:
        ta, tb = {h: sorted(a[h]) for h in FAMILY}, {h: sorted(b[h]) for h in FAMILY}
        row.append(f"{k50(a, SIX, ta):.2f} → {k50(b, SIX, tb):.2f}")
        V["k50"][f"{name}.six.fixed"], V["k50"][f"{name}.six.scaled"] = k50(a, SIX, ta), k50(b, SIX, tb)
    L.append("| pooled six | " + " | ".join(row) + " |")

    def undefined(by, g, scaled):
        """Failed runs on var > 1000 targets whose best tree is undefined at every validation point
        (final error 1000 under the scaled penalty, 1000/var under the fixed one)."""
        f = [(t, p) for t in by[g] if var[t] > 1000 for p in by[g][t] if not p[1]]
        n = sum(1 for t, p in f if p[2] not in (None, "")
                and np.isclose(float(p[2]), 1000.0 if scaled else 1000 / var[t], rtol=1e-6))
        return n, len(f)
    L += ["", "Failed EDL runs on targets with var > 1000 whose best tree is undefined at every validation point: "
          + "; ".join(f"{name}: fixed penalty {undefined(a, 'F-EDL', False)[0]} of {undefined(a, 'F-EDL', False)[1]}, "
                      f"scaled {undefined(b, 'F-EDL', True)[0]} of {undefined(b, 'F-EDL', True)[1]}"
                      for name, a, b in pairs) + ".", ""]

    # added after the results (D036): are the operator comparisons robust to the penalty?
    from reach.analysis import family_full as F
    F.RNG = np.random.default_rng(20261015)
    rows = [dict(op=g, tid=t, K=p[0], y=p[1]) for g in FAMILY for t in pairs[0][2][g] for p in pairs[0][2][g][t]]
    point, reps = F.k50_boot(rows, F.FAMILY)
    ok1, tab = F.equivalence(point, reps, 1.5)
    V["succ_high_fixed"], V["succ_high_scaled"] = succ(pairs[0][1]), succ(pairs[0][2])
    V["undef_scaled_gp"] = undefined(pairs[0][2], "F-EDL", True)[0]
    lowsel = lambda by: {g: [t for t in by[g] if var[t] <= 1000] for g in FAMILY}
    V["edl_low_fixed"], V["edl_low_scaled"] = (k50(by, ["F-EDL"], lowsel(by)) for by in (pairs[0][1], pairs[0][2]))
    V["move_gp"] = {g[2:]: V["k50"][f"GP.{g[2:]}.scaled"] - V["k50"][f"GP.{g[2:]}.fixed"] for g in FAMILY}
    V["gain_crossover"] = {g[2:]: V["k50"][f"GP.{g[2:]}.fixed"] - V["k50"][f"GP without crossover.{g[2:]}.fixed"] for g in FAMILY}
    V["range_seven_gp"] = max(V["k50"][f"GP.{g[2:]}.fixed"] for g in FAMILY) - min(V["k50"][f"GP.{g[2:]}.fixed"] for g in FAMILY)
    V["max_move_six_gp"] = max(abs(V["k50"][f"GP.{g[2:]}.scaled"] - V["k50"][f"GP.{g[2:]}.fixed"]) for g in SIX)
    V["h1_scaled"] = dict(holds=bool(ok1), **{g[2:]: dict(d=d, lo=lo, hi=hi) for g, k, d, lo, hi, p in tab})
    L += ["### Operator comparisons under the scaled penalty (added after the results)", "",
          f"H1's registered test (each of the six within ±1.5 nodes of their pooled K₅₀, 90% interval) applied to GP with "
          f"the scaled penalty: {'holds' if ok1 else 'fails'}.", "",
          "| Operator | K₅₀ | difference from pooled six | 90% interval | inside ±1.5 |", "|---|---|---|---|---|"]
    L += [f"| {NAMES[g]} | {k:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {'yes' if p else 'no'} |" for g, k, d, lo, hi, p in tab]
    gpx = (runs("nox_gp", "confB"), runs("pen_nox", "confBP"))
    sd = [spread_diff(gpx[0], pairs[0][1], 3), spread_diff(gpx[1], pairs[0][2], 4)]
    V["spread"] = {lab: dict(nox=x[0], gp=x[1], lo=x[2], hi=x[3]) for lab, x in zip(("fixed", "scaled"), sd)}
    L += ["", "Spread of the six (max − min K₅₀), GP without crossover against GP: "
          f"fixed penalty {sd[0][0]:.2f} against {sd[0][1]:.2f} (difference 95% CI {sd[0][2]:+.2f} to {sd[0][3]:+.2f}); "
          f"scaled penalty {sd[1][0]:.2f} against {sd[1][1]:.2f} (difference 95% CI {sd[1][2]:+.2f} to {sd[1][3]:+.2f}).", ""]

    two = set(SEEDS[FULL_SEEDS][:2])
    gp2, nox2, hc = runs("full_gp", "conf", two), runs("nox_gp", "confB", two), runs("hc_gp", "confH")
    L += ["## 2. Headless-chicken crossover", "",
          "Same targets and first two seeds; EML-GP's fixed penalty. K₅₀ (nodes; 95% target-bootstrap interval):", "",
          "| Operator | GP (subtree crossover) | headless-chicken crossover | GP without crossover |", "|---|---|---|---|"]
    for g in FAMILY + ["pooled six"]:
        ops = SIX if g == "pooled six" else [g]
        cells = [boot(d, lambda sel, d=d: k50(d, ops, sel)) for d in (gp2, hc, nox2)]
        if g == "pooled six":
            V["hc"] = dict(gp=cells[0][0], hc=cells[1][0], nox=cells[2][0])
        L.append(f"| {NAMES.get(g, g)} | " + " | ".join(f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]" for c in cells) + " |")
    d_gh = boot_pair(gp2, hc)
    d_hn = boot_pair(hc, nox2)
    L += ["", f"Pooled six, subtree − headless-chicken: {fmt(d_gh)} nodes; headless-chicken − no crossover: {fmt(d_hn)} nodes "
          "(paired bootstrap over targets).", ""]
    sizes = {name: [int(r["final_tree_size"]) for r in fixed.results(ph).values()
                    if r.get("exp") == exp and r["seed"] in two and r["exact_hash_match"] != "true"]
             for name, ph, exp in (("subtree crossover", "full_gp", "conf"), ("headless-chicken", "hc_gp", "confH"),
                                   ("no crossover", "nox_gp", "confB"))}
    b = spread_diff(hc, gp2, 2)
    L += [f"Spread of the six (max − min K₅₀): headless-chicken {b[0]:.2f} against subtree crossover {b[1]:.2f} "
          f"(two seeds; difference 95% CI {b[2]:+.2f} to {b[3]:+.2f}).", ""]
    L += ["Final tree size of failed runs (median [interquartile range], nodes): "
          + "; ".join(f"{n} {np.median(v):.0f} [{np.percentile(v, 25):.0f}–{np.percentile(v, 75):.0f}]" for n, v in sizes.items()) + ".", ""]
    V["hc"].update(gp_minus_hc=d_gh[0], gp_minus_hc_lo=d_gh[1], gp_minus_hc_hi=d_gh[2], hc_minus_nox=d_hn[0],
                   gp2_minus_nox=V["hc"]["gp"] - V["hc"]["nox"], spread_hc=b[0], spread_gp2=b[1], spread_lo=b[2], spread_hi=b[3],
                   **{f"size_{k.split()[0]}": float(np.median(v)) for k, v in sizes.items()})
    values.record("sensitivity", V)
    (ROOT / "paper/results/sensitivity.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def spread_diff(a, b, sel_seed):
    tg = {g: sorted(set(a[g]) & set(b[g])) for g in FAMILY}
    sp = lambda by, sel: (lambda k: max(k) - min(k))([k50(by, [g], sel) for g in SIX])
    rng = np.random.default_rng(sel_seed)
    reps = [sp(a, s) - sp(b, s) for s in ({g: [tg[g][i] for i in rng.integers(0, len(tg[g]), len(tg[g]))]
                                           for g in FAMILY} for _ in range(B))]
    return sp(a, tg), sp(b, tg), *np.quantile(reps, [0.025, 0.975])

def boot_pair(a, b):
    """Pooled-six K50 difference a - b, targets resampled jointly (same targets in both)."""
    tg = {g: sorted(set(a[g]) & set(b[g])) for g in FAMILY}
    stat = lambda sel: k50(a, SIX, sel) - k50(b, SIX, sel)
    reps = np.array([stat({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY})
                     for _ in range(B)])
    lo, hi = np.nanquantile(reps, [0.025, 0.975])
    return stat(tg), lo, hi


if __name__ == "__main__":
    main()
