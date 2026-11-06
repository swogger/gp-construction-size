"""Figures 3 and 4 (post hoc views of registered data; corrected scoring for the first study's phases).
Figure 3: reach (K50) against budget, GP beside exhaustive enumeration. GP: pooled K50 on the like-for-like targets
(seven operators, sizes 9-17: E0, 4E0, 16E0; and EML, LDE c=0, PLM, sizes 11-17: E0..64E0), with 95% target-bootstrap
intervals. Enumeration: K50 of exhaustive enumeration at each budget (range over the seven operators and median).
Figure 4: K50 per operator under GP, GP without crossover and the (1+4)-ES (E0), with 95% target-bootstrap intervals.
-> paper/figures/fig3_reach_budget.*, paper/figures/fig4_regimes.*, paper/results/fig_reach.md"""
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
import size_wall  # noqa: E402
from exchange_rate import enum_k50  # noqa: E402
import fixed  # noqa: E402

B, RNG = 1000, np.random.default_rng(20261018)
INK, INK2, GRID, GREY = "#1f1f1e", "#5f5e58", "#e6e5e0", "#b9b8b2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def ci_k50(pts_by_target, sel_fn):
    """Point and 95% interval of K50 over a dict {target: [(K, y), ...]} with target resampling."""
    tids = list(pts_by_target)
    def k(ts):
        p = [q for t in ts for q in pts_by_target[t]]
        return _k50(np.array([q[0] for q in p], float), np.array([q[1] for q in p], float))
    reps = [k([tids[i] for i in RNG.integers(0, len(tids), len(tids))]) for _ in range(B)]
    return k(tids), np.nanquantile(reps, [0.025, 0.975])


def fig3(L):
    d = size_wall.outcomes()
    sets = {"seven operators, sizes 9–17": (FAMILY, 16, [1, 4, 16]),
            "EML, LDE (c=0), PLM, sizes 11–17": (["F-EML", "F-LDE0", "F-PLM"], 64, [1, 4, 16, 64])}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    E = 2e4 * 2 ** np.linspace(0, 6, 61)
    ek = np.array([[enum_k50(g, e) for e in E] for g in FAMILY])
    ax.fill_between(E / 2e4, np.nanmin(ek, 0), np.nanmax(ek, 0), color="#ece9df", lw=0, zorder=0)
    ax.plot(E / 2e4, np.nanmedian(ek, 0), color=INK2, lw=1.6, ls=(0, (5, 2)), zorder=1)
    L += ["## Figure 3", "", "| Set | budget | GP K50 [95% CI] | enumeration K50 (median over operators) |", "|---|---|---|---|"]
    styles = {"seven operators, sizes 9–17": dict(color=BLUE, marker="o"), "EML, LDE (c=0), PLM, sizes 11–17": dict(color=ORANGE, marker="s")}
    for name, (ops, ref, mults) in sets.items():
        tg = {g: sorted({t for (o, t, m) in d if o == g and m == ref}) for g in ops}
        pts, los, his = [], [], []
        for m in mults:
            by = {(g, t): [(int(t.split("_K")[1][:2]), y) for y in d[(g, t, m)]] for g in ops for t in tg[g]}
            k, (lo, hi) = ci_k50(by, None)
            pts.append(k); los.append(lo); his.append(hi)
            L.append(f"| {name} | {m}× | {k:.2f} [{lo:.2f}, {hi:.2f}] | {np.nanmedian([enum_k50(g, 2e4 * m) for g in FAMILY]):.2f} |")
        st = styles[name]
        ax.fill_between(mults, los, his, color=st["color"], alpha=0.15, lw=0, zorder=2)
        ax.plot(mults, pts, color=st["color"], lw=2, marker=st["marker"], ms=5, markeredgecolor="white", zorder=3, label=f"GP, {name}")
    ax.plot([], [], color=INK2, lw=1.6, ls=(0, (5, 2)), label="exhaustive enumeration (median; band: range over operators)")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["E₀ = 2×10⁴", "4×", "16×", "64×"])
    ax.set_xlim(0.9, 70); ax.set_ylim(10, 23)
    ax.set_xlabel("evaluation budget"); ax.set_ylabel("reach K₅₀ (nodes of shortest construction)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    em = np.nanmedian(ek, 0)
    es = (em[-1] - em[0]) / 6
    ax.text(5.0, 11.35, "GP: +0.30 nodes per doubling\n(one node ≈ 10× the budget)", color=BLUE, fontsize=7.5)
    ax.text(6.0, 17.0, f"enumeration: about +{es:.1f} nodes per doubling", color=INK2, fontsize=7.5)
    L.append(f"\nEnumeration median K50 rises {es:.2f} nodes per doubling over E0..64E0.")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.set_title("GP's reach grows slowly and stays below enumeration", loc="left", fontsize=10, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/figures/fig3_reach_budget.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def runs(phase, exp, flips_phase=None):
    flips = set(json.loads((ROOT / f"data/rescored/{flips_phase}.json").read_text())) if flips_phase else set()
    by = defaultdict(lambda: defaultdict(list))
    for k, r in fixed.results(phase).items():
        if r.get("exp") == exp:
            by[r["grammar"]][r["target_id"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in flips)))
    return by


def fig4(L):
    regimes = [("GP", runs("full_gp", "conf", "full_gp"), BLUE, "o"),
               ("GP without crossover", runs("nox_gp", "confB"), ORANGE, "s"),
               ("(1+4)-ES", runs("full_es", "es", "full_es"), AQUA, "D")]
    est = {}
    L += ["", "## Figure 4", "", "| Operator | " + " | ".join(r[0] for r in regimes) + " |", "|---|---|---|---|"]
    for g in FAMILY:
        est[g] = [ci_k50(by[g], None) for _, by, _, _ in regimes]
        L.append(f"| {NAMES[g]} | " + " | ".join(f"{k:.2f} [{lo:.2f}, {hi:.2f}]" for k, (lo, hi) in est[g]) + " |")
    order = sorted(FAMILY, key=lambda g: est[g][0][0])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for j, (name, _, col, mk) in enumerate(regimes):
        for i, g in enumerate(order):
            k, (lo, hi) = est[g][j]
            y = i + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=col, lw=1.2, alpha=0.8, zorder=2)
            ax.plot(k, y, marker=mk, color=col, ms=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                    label=name if i == 0 else None)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[g] for g in order])
    for t in ax.get_yticklabels():
        if t.get_text() == "EDL":
            t.set_fontweight("bold")
    ax.set_xlabel("reach K₅₀ at E₀ (nodes; bars: 95% intervals)")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=7.5)
    ax.set_title("Crossover raises reach and makes operators alike", loc="left", fontsize=10, fontweight="bold", color=INK)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"paper/figures/fig4_regimes.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    L = ["# Figures 3 and 4: values (post hoc views; 95% target-bootstrap intervals, B = 1000)", ""]
    fig3(L); fig4(L)
    (ROOT / "paper/results/fig_reach.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
