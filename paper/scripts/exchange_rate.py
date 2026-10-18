"""The exchange rate: what each effect is worth in compute (post hoc interpretation of registered effects).

Rate: s = increase of pooled K50 per doubling of the budget, like-for-like (size_wall.py, set A). One node is
worth 2^(1/s) times the compute; an effect of D nodes is worth 2^(D/s). Effects (nodes, corrected scoring, fixed engine):
  operator spread  max - min K50 of the six non-EDL operators (GP, E0; full_gp)
  EDL              pooled six - EDL (GP, E0); also with the penalty for undefined values scaled to the target
                   (pen_gp, D034); and the penalty's own effect on EDL's K50 (scaled - fixed, same targets and seeds),
                   the value Figure 1 shows, so that every bar of Figure 1 uses the registered penalty (D036)
  spread, scaled   the operator spread with the scaled penalty (reported beside the registered one)
  crossover        pooled six with crossover (full_gp) - without (nox_gp), same targets and seeds (paired)
  enumeration      K50 of exhaustive enumeration at E0 - GP K50, per operator (enumeration is deterministic)
Intervals: target bootstrap of each effect (B = 1000) combined with the bootstrap of s. Values above 64x
extrapolate s beyond the budgets measured (1x-64x). -> paper/results/exchange_rate.md, paper/figures/fig_exchange.*"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reach.analysis.family_full import _k50
from reach.family import FAMILY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import size_wall                                                    # noqa: E402
import values                                                       # noqa: E402
import fixed  # noqa: E402

B, RNG = 1000, np.random.default_rng(20261017)
SIX = [g for g in FAMILY if g != "F-EDL"]
INK, INK2, GRID, BLUE, LIGHT, PALE = "#1f1f1e", "#5f5e58", "#e6e5e0", "#2a78d6", "#a9cbf3", "#f2f1ec"


def sig2(x):
    """Round to two significant figures for display."""
    from math import floor, log10
    return int(round(x, -int(floor(log10(abs(x)))) + 1)) if x >= 10 else int(round(x))


def slope_samples():
    d = size_wall.outcomes()
    tg = {g: sorted({t for (o, t, m) in d if o == g and m == 16}) for g in FAMILY}
    K = {t: int(t.split("_K")[1][:2]) for g in FAMILY for t in tg[g]}
    x = np.log2([1, 4, 16])

    def pooled(sel, m):
        ks, ys = [], []
        for g in FAMILY:
            for t in sel[g]:
                ys += d[(g, t, m)]; ks += [K[t]] * len(d[(g, t, m)])
        return _k50(np.array(ks, float), np.array(ys))
    sl = lambda sel: float(np.polyfit(x, [pooled(sel, m) for m in (1, 4, 16)], 1)[0])
    point = sl(tg)
    reps = [sl({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY}) for _ in range(B)]
    return point, np.array(reps)


def runs(phase, exp, flips=frozenset()):
    by = defaultdict(lambda: defaultdict(list))
    for k, r in fixed.results(phase).items():
        if r.get("exp") == exp:
            by[r["grammar"]][r["target_id"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in flips)))
    return by


def k50(by, g, tids):
    pts = [p for t in tids for p in by[g][t]]
    return _k50(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def pooled(by, ops, sel):
    pts = [p for g in ops for t in sel[g] for p in by[g][t]]
    return _k50(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))


def enum_k50(g, E=2e4):
    lv = json.loads((ROOT / f"data/enumeration/{g}/counts.json").read_text())["levels"]
    before, pts = 0.0, []
    for l in lv:
        c = l["candidates"]
        if c > 0:
            pts.append((l["K"], float(np.clip((E - before) / c, 0, 1))))
        before += c
    pts = [p for p in pts if p[0] % 2 == 1]
    for (k0, p0), (k1, p1) in zip(pts, pts[1:]):
        if p0 >= 0.5 > p1:
            return k0 + (p0 - 0.5) / (p0 - p1) * (k1 - k0)
    return float("nan")


def main():
    s, s_reps = slope_samples()
    flips = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    gp, nox, pen = runs("full_gp", "conf", flips), runs("nox_gp", "confB"), runs("pen_gp", "confP")
    tg = {g: sorted(gp[g]) for g in FAMILY}

    def effects(sel):
        k = {g: k50(gp, g, sel[g]) for g in FAMILY}
        six = pooled(gp, SIX, sel)
        return dict(spread=max(k[g] for g in SIX) - min(k[g] for g in SIX), edl=six - k["F-EDL"],
                    edl_scaled=pooled(pen, SIX, sel) - k50(pen, "F-EDL", sel["F-EDL"]),
                    spread_scaled=(lambda kk: max(kk) - min(kk))([k50(pen, g, sel[g]) for g in SIX]),
                    penalty_edl=k50(pen, "F-EDL", sel["F-EDL"]) - k["F-EDL"],
                    crossover=six - pooled(nox, SIX, sel))
    pt = effects(tg)
    reps = [effects({g: [tg[g][i] for i in RNG.integers(0, len(tg[g]), len(tg[g]))] for g in FAMILY}) for _ in range(B)]
    gaps = {g: enum_k50(g) - k50(gp, g, tg[g]) for g in FAMILY}

    fold = lambda d, sl: 2 ** (d / sl)
    rows = [("one extra node of target size", 1.0, np.ones(B))]
    rows += [("spread among six operators", pt["spread"], np.array([r["spread"] for r in reps])),
             ("EDL against the other six", pt["edl"], np.array([r["edl"] for r in reps])),
             ("EDL against the other six, penalty scaled to the target", pt["edl_scaled"], np.array([r["edl_scaled"] for r in reps])),
             ("spread among six operators, penalty scaled to the target", pt["spread_scaled"], np.array([r["spread_scaled"] for r in reps])),
             ("penalty for undefined values, EDL (scaled − fixed, same runs)", pt["penalty_edl"], np.array([r["penalty_edl"] for r in reps])),
             ("crossover (with vs without)", pt["crossover"], np.array([r["crossover"] for r in reps]))]
    out = []
    for name, d, dr in rows:
        f = fold(d, s); fr = fold(dr, s_reps)
        out.append((name, d, f, np.quantile(fr, [0.025, 0.975])))
    g_med = float(np.median(list(gaps.values())))
    g_lo, g_hi = min(gaps.values()), max(gaps.values())
    out.append(("GP to match exhaustive enumeration", g_med, fold(g_med, s), np.array([fold(g_lo, s), fold(g_hi, s)])))

    L = ["# Exchange rate: what each effect is worth in compute (post hoc)", "",
         f"Rate: {s:.3f} nodes per doubling (95% CI {np.quantile(s_reps, 0.025):.3f}–{np.quantile(s_reps, 0.975):.3f}); "
         f"one node ≈ {fold(1, s):.1f}× compute. Measured budgets: 1×–64×; larger multiples extrapolate.", "",
         "| Effect | nodes | compute equivalent | 95% interval |", "|---|---|---|---|"]
    L += [f"| {n} | {d:.3f} | {f:.1f}× | {lo:,.0f}–{hi:,.0f}× |" if f < 10 else f"| {n} | {d:.3f} | {f:,.0f}× | {lo:,.0f}–{hi:,.0f}× |"
          for n, d, f, (lo, hi) in out]
    L += ["", "Enumeration row: median over the seven operators of (enumeration K50 at E0 − GP K50); the interval is the "
          "range over operators at the point rate. Per operator: " + ", ".join(f"{g[2:]} {v:+.2f}" for g, v in gaps.items()) + ".", ""]
    (ROOT / "paper/results/exchange_rate.md").write_text("\n".join(L) + "\n")
    short = {"one extra node of target size": "node", "spread among six operators": "spread",
             "EDL against the other six": "edl", "EDL against the other six, penalty scaled to the target": "edl_scaled",
             "spread among six operators, penalty scaled to the target": "spread_scaled",
             "penalty for undefined values, EDL (scaled − fixed, same runs)": "penalty_edl",
             "crossover (with vs without)": "crossover", "GP to match exhaustive enumeration": "enum"}
    v = {"h1_margin_fold": 2 ** (1.5 / s), "s": s, "s_lo": np.quantile(s_reps, 0.025), "s_hi": np.quantile(s_reps, 0.975), "gap_min": g_lo, "gap_max": g_hi}
    for n, d, f, (lo, hi) in out:
        v[short[n]] = dict(nodes=d, fold=f, lo=lo, hi=hi)
    v["gap"] = {g[2:]: x for g, x in gaps.items()}
    values.record("exchange_rate", v)
    print("\n".join(L))
    plot(out)


def plot(out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    out = sorted(out, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axvspan(1, 64, color=PALE, zorder=0, lw=0)
    ax.text(1.12, -0.62, "budgets measured (1–64×)", color=INK2, fontsize=6.5, va="center")
    for i, (name, d, f, (lo, hi)) in enumerate(out):
        ax.barh(i, min(f, 64) - 1, left=1, height=0.52, color=BLUE, zorder=2)
        if f > 64:
            ax.barh(i, f - 64, left=64, height=0.52, color=LIGHT, zorder=2)
        enum = name.startswith("GP to match")
        ax.plot([lo, hi], [i, i], color=INK, lw=1, zorder=3, ls=(0, (2, 1.5)) if enum else "-")
        note = "; range over operators" if enum else ""
        ax.text(max(hi, f) * 1.35, i, f"≈ {sig2(f):,}×   ({d:.1f} node{'s' if abs(d - 1) > 0.05 else ''}{note})", va="center",
                fontsize=7.5, color=INK)
    ax.set_yticks(range(len(out))); ax.set_yticklabels([r[0] for r in out])
    ax.set_xscale("log"); ax.set_xlim(1, 6e5)
    ax.set_ylim(-0.95, len(out) - 0.5)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_xticklabels(["1×", "10×", "100×", "1,000×", "10,000×", "100,000×"])
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("equivalent increase in GP's evaluation budget (log scale). Lines: 95% intervals; light bars: extrapolated beyond 64×")
    ax.set_title("What each factor is worth in compute", loc="left", fontsize=9, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/figures/fig_exchange_horizontal.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plot_vertical(out)


SHORT = {"one extra node of target size": "one extra node\nof target size", "spread among six operators": "spread among\nsix operators",
         "EDL against the other six": "EDL against\nits six peers", "GP to match exhaustive enumeration": "GP to match\nenumeration",
         "EDL against the other six, penalty scaled to the target": "EDL, scaled\npenalty",
         "spread among six operators, penalty scaled to the target": "spread, scaled\npenalty",
         "penalty for undefined values, EDL (scaled − fixed, same runs)": "penalty\n(EDL)",
         "crossover (with vs without)": "crossover\n(with vs without)"}


def plot_vertical(out):
    out = sorted(out, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(out))
    for i, (name, d, f, (lo, hi)) in enumerate(out):
        ax.bar(i, f - 1, bottom=1, width=0.62, color=BLUE, zorder=2)
        enum = name.startswith("GP to match")
        ax.plot([i, i], [lo, hi], color=INK, lw=1.1, zorder=3, ls=(0, (2, 1.5)) if enum else "-")
        ax.text(i, max(hi, f) * 1.6, f"≈ {sig2(f):,}×", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(i, max(hi, f) * 1.6 / 1.9, f"{d:.1f} node{'s' if abs(d - 1) > 0.05 else ''}", ha="center", va="bottom",
                fontsize=7, color=INK2)
    ax.axhline(64, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.45, 2.2e5, "dashed line: largest budget measured (64×); taller bars extrapolate", fontsize=6.8, color=INK2, va="center")
    ax.set_yscale("log"); ax.set_ylim(1, 4e5)
    ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_yticklabels(["1×", "10×", "100×", "1,000×", "10,000×", "100,000×"])
    ax.set_xticks(x); ax.set_xticklabels([SHORT[r[0]] for r in out], fontsize=7.6)
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("equivalent increase in GP's evaluation budget")
    ax.set_title("What each factor is worth in compute", loc="left", fontsize=10, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/figures/fig_exchange.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
