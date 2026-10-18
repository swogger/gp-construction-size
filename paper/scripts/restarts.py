"""Post hoc (D038): independent restarts against single longer runs at equal total budget, on set A (the 700 targets
of the budget runs: seven operators, sizes 9-17, 20 per size). Specified before the restart_seeds runs.
- Single runs: K50 at E0, 4E0 and 16E0 (size_wall.outcomes; corrected scoring, fixed engine).
- Runs at E0 per target: full_gp's 5 seeds plus restart_seeds' 15 (when present). A run of L <= 200 generations
  succeeds iff the E0 run succeeded by generation L - 1 (no early stopping or length-dependent schedule in the
  study engine, so truncation is exact).
- Budget B (in E0) spent as r = B * 200 / L runs of L generations: P(at least one success) per target is estimated
  without bias as 1 - C(n - s, r)/C(n, r) when r <= n, and by the plug-in 1 - (1 - s/n)^r (conservative) when r > n.
- K50 from a binomial GLM of the per-target probabilities on K (one observation per target); 95% intervals from a
  bootstrap of targets within operators (B = 1000).
- Best allocation: the run length with the highest K50 at each budget; its gain per doubling of the budget.
- Timing: generation of success of every successful E0 run on the random targets (all sizes).
- Enumeration: per operator, GP's K50 at E0 under the best split against exhaustive enumeration's K50 at E0.
-> paper/results/restarts.md, values"""
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np
import statsmodels.api as sm

from reach.family import FAMILY
import fixed  # noqa: E402
import size_wall  # noqa: E402
import values  # noqa: E402
from reach.runner.phases import FULL_SEEDS  # noqa: E402
from reach.seeds import SEEDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from exchange_rate import enum_k50  # noqa: E402

B, RNG = 1000, np.random.default_rng(20261016)
SEEDS5 = set(SEEDS[FULL_SEEDS][:5])
LENGTHS, BUDGETS = (25, 50, 100, 200), (1, 2, 4, 8, 16)
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def k50(pairs):
    k = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
    if y.min() == y.max():
        return float("nan")
    m = sm.GLM(y, sm.add_constant(k), family=sm.families.Binomial()).fit()
    return float(-m.params[0] / m.params[1]) if m.params[1] < 0 else float("nan")


def p_restart(gens, L, r):
    """gens: generation of success per E0 run (None = no success). P(>= 1 success in r runs of L generations)."""
    n, s = len(gens), sum(1 for x in gens if x is not None and x <= L - 1)
    if r <= n:
        return 1 - comb(n - s, r) / comb(n, r)
    return 1 - (1 - s / n) ** r


def main():
    d = size_wall.outcomes()
    tg = {g: sorted({t for (o, t, m) in d if o == g and m == 16}) for g in FAMILY}
    sel_all = {t for g in FAMILY for t in tg[g]}
    K = {t: int(t.split("_K")[1][:2]) for t in sel_all}
    fl = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    runs = defaultdict(list)                      # target -> [generation of success or None]
    for ph in ("full_gp", "restart_seeds"):
        if not (ROOT / f"data/runs/{ph}/manifest.json").exists():
            continue
        for k, r in fixed.results(ph).items():
            if r.get("exp") == "conf" and r["target_id"] in sel_all:
                ok = r.get("exact_hash_match") == "true" or k in fl
                g = int(r["gens_run"]) if r.get("exact_hash_match") == "true" else (199 if ok else None)
                runs[r["target_id"]].append(g)
    nseeds = min(len(v) for v in runs.values())

    def table(sel):
        out = {(1, "single"): k50([(K[t], np.mean(d[(g, t, 1)])) for g in FAMILY for t in sel[g]])}
        for m in (4, 16):
            out[(m, "single")] = k50([(K[t], np.mean(d[(g, t, m)])) for g in FAMILY for t in sel[g]])
        for b in BUDGETS:
            for L in LENGTHS:
                r = b * 200 // L
                out[(b, L)] = k50([(K[t], p_restart(runs[t], L, r)) for g in FAMILY for t in sel[g]])
        return out
    pt = table(tg)
    reps = [table({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY}) for _ in range(B)]
    ci = lambda f: np.nanquantile([f(r) for r in reps], [0.025, 0.975])
    best = {b: max(LENGTHS, key=lambda L: pt[(b, L)]) for b in BUDGETS}
    bestk = {b: pt[(b, best[b])] for b in BUDGETS}
    slope = float(np.polyfit(np.log2(BUDGETS), [bestk[b] for b in BUDGETS], 1)[0])
    seg = {f"{a}-{b}": (bestk[b] - bestk[a]) / np.log2(b / a) for a, b in zip(BUDGETS, BUDGETS[1:])}
    d4, d16 = (lambda r: r[(4, 200)] - r[(4, "single")]), (lambda r: r[(16, 200)] - r[(16, "single")])
    d1 = lambda r: r[(1, best[1])] - r[(1, "single")]      # added (D039): E0 split at its best run length vs one run

    gens = np.array([int(r["gens_run"]) for r in fixed.results("full_gp").values()
                     if r.get("exp") == "conf" and r.get("exact_hash_match") == "true"])
    L = ["# Post hoc: restarts against longer runs at equal budget (set A)", "",
         f"700 targets (seven operators, sizes 9–17); {nseeds} E0 runs per target; K₅₀ in nodes. Single run of B × 200 "
         "generations, or B × 200 / L independent runs of L generations. Values with more runs than seeds use the "
         "conservative plug-in (marked *).", "",
         "| Budget | single run | " + " | ".join(f"runs of {L_} gen." for L_ in LENGTHS) + " | best |",
         "|---|---|" + "---|" * len(LENGTHS) + "---|"]
    for b in BUDGETS:
        single = f"{pt[(b, 'single')]:.2f}" if (b, "single") in pt else "–"
        cells = [f"{pt[(b, L_)]:.2f}{'*' if b * 200 // L_ > nseeds else ''}" for L_ in LENGTHS]
        L.append(f"| {b} E0 | {single} | " + " | ".join(cells) + f" | {bestk[b]:.2f} ({best[b]} gen.) |")
    c4, c16, c1 = ci(d4), ci(d16), ci(d1)
    L += ["", f"Restarts of full E0 runs minus one long run at the same budget: 4 E0 {d4(pt):+.2f} [{c4[0]:+.2f}, {c4[1]:+.2f}]; "
          f"16 E0 {d16(pt):+.2f} [{c16[0]:+.2f}, {c16[1]:+.2f}] (95% target bootstrap)." if nseeds >= 16 else
          f"", f"Best allocation: reach gains {slope:.2f} nodes per doubling of the budget from E0 to 16 E0 (by segment: "
          + ", ".join(f"{k} E0 {v:.2f}" for k, v in seg.items()) + "); a long single run gains "
          f"{(pt[(16, 'single')] - pt[(1, 'single')]) / 4:.2f}.", "",
          f"Timing of the {len(gens)} successful E0 runs on random targets (all sizes): median generation {np.median(gens):.0f}; "
          f"{(gens <= 10).mean():.0%} by generation 10, {(gens <= 50).mean():.0%} by 50, {(gens > 100).mean():.0%} after 100.", ""]
    # added (D040): the best split measured without bias (r <= n runs per target) up to 4 E0, over finer run lengths
    fine = (10, 20, 25, 40, 50, 100, 200)
    ub = {}
    for b in (1, 2, 4):
        cells = {Lf: k50([(K[t], p_restart(runs[t], Lf, b * 200 // Lf)) for g in FAMILY for t in tg[g]])
                 for Lf in fine if b * 200 % Lf == 0 and b * 200 // Lf <= nseeds}
        ub[b] = max(cells.items(), key=lambda kv: kv[1])
    slope_ub = (ub[4][1] - ub[1][1]) / 2
    # added (D043): interval for the unbiased best-split rate, re-selecting the best split in every bootstrap replicate
    rb = np.random.default_rng(20261018)
    def ub_slope(sel):
        best = {}
        for b in (1, 4):
            best[b] = max(k50([(K[t], p_restart(runs[t], Lf, b * 200 // Lf)) for g in FAMILY for t in sel[g]])
                          for Lf in fine if b * 200 % Lf == 0 and b * 200 // Lf <= nseeds)
        return (best[4] - best[1]) / 2
    ub_reps = [ub_slope({g: [tg[g][i] for i in rb.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY}) for _ in range(300)]
    ub_lo, ub_hi = np.nanquantile(ub_reps, [0.025, 0.975])
    L += [f"Best split measured without bias (at most {nseeds} runs per target): "
          + ", ".join(f"{b} E0 {ub[b][1]:.2f} ({b * 200 // ub[b][0]} × {ub[b][0]} gen.)" for b in (1, 2, 4))
          + f"; {slope_ub:.2f} nodes per doubling (95% interval {ub_lo:.2f}–{ub_hi:.2f}, re-selecting the best split in each of 300 "
          f"bootstrap replicates; still optimistic, since the lengths are chosen on the same data), so one node ≈ {2 ** (1 / slope_ub):.1f}× the budget. Above 4 E0 the best "
          "split needs more runs per target than there are seeds, so the rate over E0–16 E0 above is a lower bound.", ""]
    # enumeration against GP's best split of E0, per operator (set A targets)
    gp_best = {g: max(k50([(K[t], p_restart(runs[t], L_, 200 // L_)) for t in tg[g]]) for L_ in LENGTHS) for g in FAMILY}
    gp_single = {g: k50([(K[t], np.mean(d[(g, t, 1)])) for t in tg[g]]) for g in FAMILY}
    gaps = {g: enum_k50(g) - gp_best[g] for g in FAMILY}
    gp16 = {g: k50([(K[t], p_restart(runs[t], 200, 16)) for t in tg[g]]) for g in FAMILY}       # 16 runs of E0
    gaps16 = {g: enum_k50(g, E=16 * 2e4) - gp16[g] for g in FAMILY}
    L += ["Exhaustive enumeration against GP at E0 (set A; K₅₀):", "", "| Operator | GP, one run | GP, best split of E0 | enumeration | gap to best split |",
          "|---|---|---|---|---|"]
    L += [f"| {NAMES[g]} | {gp_single[g]:.2f} | {gp_best[g]:.2f} | {enum_k50(g):.2f} | {gaps[g]:+.2f} |" for g in FAMILY]
    L += ["", f"Median gap {np.median(list(gaps.values())):.2f} nodes (range {min(gaps.values()):.2f} to {max(gaps.values()):.2f}). "
          f"At 16 E0 (enumeration at 16 E0 against 16 runs of E0): median {np.median(list(gaps16.values())):.2f} "
          f"(range {min(gaps16.values()):.2f} to {max(gaps16.values()):.2f}); there GP's best split could not be measured and "
          "several operators' K50 lie beyond the 17-node window.", ""]
    # added after the design, on existing data only (D039): crossover's advantage and the operator spread under restarts
    six = [g for g in FAMILY if g != "F-EDL"]
    allg = defaultdict(lambda: defaultdict(list))
    for ph, exp in (("full_gp", "conf"), ("nox_gp", "confB")):
        for k, r in fixed.results(ph).items():
            if r.get("exp") == exp and r["seed"] in SEEDS5:
                allg[ph][(r["grammar"], r["target_id"], int(r["K"]))].append(float(r.get("exact_hash_match") == "true" or k in fl))
    cx = {rr: k50([(Kt, 1 - comb(len(ys) - int(sum(ys)), rr) / comb(len(ys), rr)) for (g, t, Kt), ys in allg["full_gp"].items() if g in six])
          - k50([(Kt, 1 - comb(len(ys) - int(sum(ys)), rr) / comb(len(ys), rr)) for (g, t, Kt), ys in allg["nox_gp"].items() if g in six])
          for rr in (1, 4)}
    spread_best = max(gp_best[g] for g in six) - min(gp_best[g] for g in six)
    spread_single = max(gp_single[g] for g in six) - min(gp_single[g] for g in six)
    L += [f"Crossover (pooled six, all random targets, same 5 seeds): GP − GP without crossover {cx[1]:+.2f} nodes with one E0 run "
          f"each, {cx[4]:+.2f} with four E0 restarts each. Spread of the six on set A at E0: {spread_single:.2f} (one run), "
          f"{spread_best:.2f} (best split).", ""]
    (ROOT / "paper/results/restarts.md").write_text("\n".join(L) + "\n")
    V = dict(nseeds=nseeds, single={str(b): pt[(b, "single")] for b in (1, 4, 16)},
             grid={f"{b}.{L_}": pt[(b, L_)] for b in BUDGETS for L_ in LENGTHS}, best={str(b): bestk[b] for b in BUDGETS},
             best_len={str(b): best[b] for b in BUDGETS}, slope_best=slope, seg=seg,
             slope_single=(pt[(16, "single")] - pt[(1, "single")]) / 4,
             d4=d4(pt), d4_lo=c4[0], d4_hi=c4[1], d1=d1(pt), d1_lo=c1[0], d1_hi=c1[1], fold_best=2 ** (1 / slope), gen_median=float(np.median(gens)), by10=float((gens <= 10).mean()),
             by50=float((gens <= 50).mean()), after100=float((gens > 100).mean()),
             best_ub={str(b): ub[b][1] for b in (1, 2, 4)}, slope_ub=slope_ub, fold_ub=2 ** (1 / slope_ub),
             slope_ub_lo=ub_lo, slope_ub_hi=ub_hi, fold_ub_lo=2 ** (1 / ub_hi), fold_ub_hi=2 ** (1 / ub_lo),
             enum_gap16_median=float(np.median(list(gaps16.values()))),
             crossover_1=cx[1], crossover_4=cx[4], spread_best=spread_best, spread_single=spread_single,
             enum_gap_median=float(np.median(list(gaps.values()))), enum_gap_min=min(gaps.values()), enum_gap_max=max(gaps.values()))
    if nseeds >= 16:
        V.update(d16=d16(pt), d16_lo=c16[0], d16_hi=c16[1])
    values.record("restarts", V)
    print("\n".join(L))


if __name__ == "__main__":
    main()
