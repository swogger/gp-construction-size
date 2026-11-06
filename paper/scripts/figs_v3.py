"""Figures, editorial style (v3). One blue ramp (monotone in lightness, so greyscale-safe) plus one red accent;
direct labels, dotted leader lines, big numbers on the marks; values from frozen data and results files.
  Figure 1  fig1v3_currency  what each factor is worth in compute (paper/results/exchange_rate.md)
  Figure 2  fig2v3_matrix    what each operator makes short: shortest size per (function, operator), shaded by the
                             fraction of GP runs that found it (calculator runs of the first study, corrected scoring)
  Figure 3  fig3v3_predict   (a) same function under different operators; (b) calibration on unseen grammars
  Figure 4  fig4v3_crossover reach per operator under the ES, GP without crossover and GP
-> paper/figures/*.png|pdf"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from reach.analysis.family_full import _k50
from reach.family import CALC, CALC2, FAMILY
import fixed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures"


def register_fonts():
    """Matplotlib sees only the first face of a .ttc; extract Avenir Next's faces into a temp dir and register them
    (system fonts; nothing is copied into the repository). Falls back silently if the collection is absent."""
    import tempfile
    from matplotlib import font_manager as fm
    src = Path("/System/Library/Fonts/Avenir Next.ttc")
    if not src.exists():
        return
    from fontTools.ttLib import TTCollection
    d = Path(tempfile.gettempdir()) / "reach_fonts"; d.mkdir(exist_ok=True)
    for i, tag in ((7, "Regular"), (4, "Italic"), (5, "Medium"), (2, "DemiBold"), (0, "Bold")):
        f = d / f"AvenirNext-{tag}.ttf"
        if not f.exists():
            TTCollection(str(src)).fonts[i].save(str(f))
        fm.fontManager.addfont(str(f))


register_fonts()
INK, INK2, INK3, RULE = "#121212", "#4d4d4d", "#8a8a8a", "#dcdcdc"
RAMP = ["#e8f1fd", "#cde2fb", "#9cc3f2", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0f3466"]
RED = "#e34948"
CMAP = LinearSegmentedColormap.from_list("ramp", RAMP)
SANS, SERIF = "Avenir Next", "Georgia"
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE$_0$", "F-LDE1": "LDE$_1$", "F-PLI": "PLI", "F-PLM": "PLM"}
RNG = np.random.default_rng(20261020)
plt.rcParams.update({"font.family": SANS, "font.size": 9.5, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": SANS,
                     "mathtext.it": SANS + ":italic", "mathtext.bf": SANS + ":bold", "mathtext.default": "regular", "savefig.facecolor": "white", "legend.frameon": False})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def headline(fig, title, sub, x=0.02, y=1.0):
    fig.text(x, y + 0.035, title, fontfamily=SERIF, fontsize=17, color=INK, va="bottom")
    fig.text(x, y, sub, fontsize=10, color=INK2, va="bottom")


def rescored(ph):
    p = ROOT / f"data/rescored/{ph}.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def sig2(x):
    from math import floor, log10
    return int(round(x, -int(floor(log10(abs(x)))) + 1)) if x >= 10 else int(round(x))


# ------------------------------------------------------------------ Figure 1: the currency chart
def fig1():
    rows = []
    for line in (ROOT / "paper/results/exchange_rate.md").read_text().splitlines():
        m = re.match(r"\| (.+?) \| ([\d.]+) \| ([\d,.]+)× \| ([\d,]+)–([\d,]+)× \|", line)
        if m:
            rows.append((m.group(1), float(m.group(2)), *(float(m.group(i).replace(",", "")) for i in (3, 4, 5))))
    show = ("one extra node of target size", "spread among six operators", "penalty for undefined values, EDL (scaled − fixed, same runs)",
            "crossover (with vs without)", "GP to match exhaustive enumeration")
    rows = [r for r in rows if r[0] in show]                             # one penalty throughout (D036)
    import values as V_                                                  # allocation: E0 split into short runs (D039)
    v = V_.load(); s = v["exchange_rate.s"]
    rows.append(("restarts", v["restarts.d1"], 2 ** (v["restarts.d1"] / s), 2 ** (v["restarts.d1_lo"] / s), 2 ** (v["restarts.d1_hi"] / s)))
    rows.sort(key=lambda r: r[2])
    lab = {"one extra node of target size": "One more\nnode", "spread among six operators": "Six\noperators",
           "penalty for undefined values, EDL (scaled − fixed, same runs)": "Fitness\npenalty (EDL)",
           "GP to match exhaustive enumeration": "GP vs\nenumeration",
           "crossover (with vs without)": "Mutation\nonly", "restarts": "Short\nrestarts"}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cols = [RAMP[3], RAMP[4], RAMP[5], RAMP[6], RAMP[7], RAMP[8]]
    X0 = 1.0                                                                              # bars shifted right: room for the 64× note
    for i, ((name, d, f, lo, hi), c) in enumerate(zip(rows, cols)):
        i = i + X0
        ax.bar(i, f - 1, bottom=1, width=0.66, color=c, zorder=2, linewidth=0)          # solid, one colour per bar
        xe = i + 0.45                                                                     # interval beside the bar
        ax.plot([xe, xe], [lo, hi], color=INK2, lw=1.1, zorder=3, ls=(0, (1.5, 1.5)) if name.startswith("GP to match") else "-")
        for yc in (lo, hi):
            ax.plot([xe - 0.035, xe + 0.035], [yc, yc], color=INK2, lw=1.1, zorder=3)
        tc = INK if i == X0 else "white"
        ax.text(i, f * 1.15, f"×{f:.1f}" if f < 10 else f"×{sig2(f):,}", ha="center", va="bottom", fontsize=12.6, fontweight="semibold", color=INK, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        ax.text(i, f / 1.25, f"{d:.1f} node{'s' if abs(d - 1) > .05 else ''}", ha="center", va="top", fontsize=9,
                fontweight="semibold", color=tc, zorder=4)
    ax.axhline(64, color=INK2, lw=1, ls=(0, (1, 2.2)), zorder=1)
    kx, ky = -0.45, 6.2e4                                   # key for the interval lines
    ax.plot([kx, kx], [ky / 1.6, ky * 1.6], color=INK2, lw=1.1); ax.text(kx + 0.1, ky, "95% interval", fontsize=8, color=INK2, va="center")
    ax.plot([kx + 1.25, kx + 1.25], [ky / 1.6, ky * 1.6], color=INK2, lw=1.1, ls=(0, (1.5, 1.5)))
    ax.text(kx + 1.35, ky, "range over the seven operators", fontsize=8, color=INK2, va="center")
    ax.text(-0.5, 64 * 1.12, "largest budget we ran (64×)", fontsize=8, color=INK2, fontstyle="italic", va="bottom", ha="left")
    ax.text(-0.5, 64 / 1.12, "above: extrapolated", fontsize=8, color=INK3, fontstyle="italic", va="top", ha="left")
    ax.set_yscale("log"); ax.set_ylim(1, 1.2e5)
    ax.set_yticks([1, 10, 100, 1e3, 1e4, 1e5]); ax.set_yticklabels(["1×", "10×", "100×", "1,000×", "10,000×", "100,000×"], fontsize=8.5)
    ax.yaxis.grid(True, color="#eeeeee", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xticks([i + X0 for i in range(len(rows))]); ax.set_xticklabels([lab[r[0]] for r in rows], fontsize=9.5, color=INK, linespacing=1.25)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(rows) + X0 - 0.25)
    ax.minorticks_off()
    headline(fig, "Shortest size has a price", "What each factor is worth, in multiples of the length of one GP run", y=0.95)
    save(fig, "fig1v3_currency")


# ------------------------------------------------------------------ Figure 2: what each operator makes short
PRETTY = {"exp(x)": "e$^x$", "ln(x)": "ln x", "-x": "−x", "1/x": "1/x", "x^2": "x$^2$", "x+1": "x + 1", "x-1": "x − 1",
          "2x": "2x", "exp(-x)": "e$^{-x}$", "x^x": "x$^x$", "exp(exp(x))": "exp(e$^x$)", "ln(ln(x))": "ln(ln x)",
          "x+y": "x + y", "x-y": "x − y", "x*y": "x · y", "x/y": "x / y", "x^y": "x$^y$", "log_x(y)": "log$_x$ y"}


def fig2():
    size = {}
    for g in FAMILY:
        for p in (g, "F2-" + g[2:]):
            for fn, hit in json.loads((ROOT / f"data/family/minimal_{p}.json").read_text())["table"].items():
                if hit and hit.get("verified"):
                    size[(fn, g)] = hit["K"]
    fl = rescored("full_gp")
    obs = defaultdict(lambda: [0, 0])
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") in ("calc", "calc2"):
            fn = r["target_id"].split("_calc_")[1]; g = r["grammar"].replace("F2-", "F-")
            c = obs[(fn, g)]; c[0] += r.get("exact_hash_match") == "true" or k in fl; c[1] += 1
    fns1 = [f for f in CALC if f in PRETTY and any((f, g) in size for g in FAMILY)]
    fns2 = [f for f in CALC2 if f in PRETTY and any((f, g) in size for g in FAMILY)]
    rows = fns1 + fns2
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    for i, fn in enumerate(rows):
        y = i + (0.6 if fn in fns2 else 0)
        best = min(size[(fn, g)] for g in FAMILY if (fn, g) in size)
        for j, g in enumerate(FAMILY):
            if (fn, g) not in size:
                ax.add_patch(plt.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84, color="#f4f4f2", lw=0))
                ax.text(j, y, "–", ha="center", va="center", fontsize=9, color="#b5b5b5")
                continue
            k = size[(fn, g)]
            s, n = obs.get((fn, g), (0, 0))
            if n:
                v = s / n
                ax.add_patch(plt.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84, color=CMAP(0.06 + 0.94 * v), lw=0))
                col = "white" if v > 0.55 else INK
            else:
                ax.add_patch(plt.Rectangle((j - 0.46, y - 0.42), 0.92, 0.84, facecolor="white", edgecolor=RULE, lw=0.8))
                col = INK2
            ax.text(j, y, str(k), ha="center", va="center", fontsize=10.5 if k == best else 10, color=col,
                    fontweight="bold" if k == best else "normal")
    ax.set_xlim(-0.6, len(FAMILY) - 0.4); ax.set_ylim(len(rows) + 0.2, -1.3)
    ax.set_xticks(range(len(FAMILY))); ax.set_xticklabels([NAMES[g] for g in FAMILY], fontsize=10, color=INK)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    ax.set_yticks([i + (0.6 if f in fns2 else 0) for i, f in enumerate(rows)])
    ax.set_yticklabels([PRETTY[f] for f in rows], fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-1.55, -0.95, "ONE VARIABLE", fontsize=7.5, color=INK3, fontweight="bold")
    ax.text(-1.55, len(fns1) - 0.05, "TWO VARIABLES", fontsize=7.5, color=INK3, fontweight="bold")
    # colour key
    cax = fig.add_axes([0.30, 0.055, 0.42, 0.013])
    grad = np.linspace(0, 1, 256)[None, :]
    from matplotlib.colors import ListedColormap
    cax.imshow(grad, aspect="auto", cmap=ListedColormap([CMAP(0.06 + 0.94 * t) for t in np.linspace(0, 1, 256)]))
    cax.set_yticks([]); cax.set_xticks([0, 127.5, 255]); cax.set_xticklabels(["0%", "50%", "100%"], fontsize=8)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.tick_params(length=0)
    fig.text(0.51, 0.075, "Share of 30 GP runs that found the function (2 × 10$^4$ evaluations)", ha="center", fontsize=8.5, color=INK2)
    fig.text(0.51, 0.018, "Numbers: shortest construction in nodes (bold: shortest across operators). "
             "Outlined: not searched (fails the viability filter).", ha="center", fontsize=7.5, color=INK3)
    headline(fig, "What each operator makes short", "Shortest construction of calculator functions, and how often GP found them", y=0.955)
    save(fig, "fig2v3_matrix")


# ------------------------------------------------------------------ Figure 3: prediction
def size_curve():
    from reach.analysis.k50 import logit
    fl = rescored("full_gp"); K, y = [], []
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") == "conf":
            K.append(float(r["K"])); y.append(float(r.get("exact_hash_match") == "true" or k in fl))
    b = logit(np.array(K), np.array(y)); return float(b[0]), float(b[1])


SHOW = [("exp(x)", "e$^x$"), ("ln(ln(x))", "ln(ln x)"), ("log_x(y)", "log$_x$ y"), ("x-1", "x − 1")]


def fig3():
    fl = rescored("full_gp"); calc = defaultdict(dict)
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") in ("calc", "calc2"):
            fn, op = r["target_id"].split("_calc_")[1], r["grammar"].replace("F2-", "F-")
            c = calc[fn].setdefault(op, [int(r["K"]), 0, 0]); c[1] += r.get("exact_hash_match") == "true" or k in fl; c[2] += 1
    b0, b1 = size_curve()
    fig = plt.figure(figsize=(7.4, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3], hspace=0.8, wspace=0.14, top=0.86)
    for j, (fn, nice) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, j])
        kk = np.linspace(2, 24, 200)
        ax.plot(kk, 1 / (1 + np.exp(-(b0 + b1 * kk))), color=RAMP[2], lw=3, zorder=1, solid_capstyle="round")
        groups = defaultdict(list)
        for op, (k, s, n) in calc[fn].items():
            groups[(k, round(s / n, 2))].append(NAMES[op])
        for (k, v), ops in groups.items():
            ax.plot(k, v, "o", ms=7.5, color=RAMP[7], mec="white", mew=1.2, zorder=3)
        key = defaultdict(list)
        for (k, v), ops in groups.items():
            key[k] += ops
        ax.text(0.0, -0.36, "\n".join(f"{k}  {', '.join(sorted(key[k]))}" for k in sorted(key)), transform=ax.transAxes,
                ha="left", va="top", fontsize=7.4, color=INK2, linespacing=1.45)
        ax.set_xlim(1.5, 25); ax.set_ylim(-0.06, 1.1); ax.set_xticks([3, 11, 19]); ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50%", "100%"] if j == 0 else [])
        ax.set_title(nice, fontsize=11, color=INK, pad=6, fontfamily=SERIF)
        ax.spines["left"].set_visible(j == 0)
        ax.tick_params(axis="y", length=0 if j else 3)
        ax.yaxis.grid(True, color="#f0f0f0", lw=0.8); ax.set_axisbelow(True)
        y0 = ax.get_position().y0
        if j == 0:
            k, s, n = calc["exp(x)"]["F-PLM"]
            ax.annotate("PLM: 15 nodes,\nfound in 17%", xy=(k, s / n), xytext=(13.5, 0.62), fontsize=7.6, color=INK, ha="center",
                        arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8, ls=(0, (1, 1.5))))
    fig.text(0.5, y0 - 0.05, "Shortest construction in each operator (nodes); key: nodes and operators", ha="center", fontsize=8.5, color=INK2)
    fig.text(0.02, 0.885, "a", fontsize=13, fontweight="bold", color=INK)
    ax = fig.add_subplot(gs[1, 1:3])
    pred = json.loads((ROOT / "data/heldout/predictions.json").read_text())
    by = defaultdict(list)
    for r in fixed.results("heldout_gp").values():
        by[r["target_id"]].append(float(r.get("exact_hash_match") == "true"))
    tids = list(by); p = np.array([pred[t]["p"] for t in tids])
    edges = np.quantile(p, np.linspace(0, 1, 11)); edges[-1] += 1e-9
    ax.plot([0, 1], [0, 1], color=INK3, lw=1, ls=(0, (1, 2.2)), zorder=1)
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [t for t, q in zip(tids, p) if a <= q < b]
        o = lambda ts: np.mean([y for t in ts for y in by[t]])
        reps = [o([sel[i] for i in RNG.integers(0, len(sel), len(sel))]) for _ in range(1000)]
        x = np.mean([pred[t]["p"] for t in sel]); yv = o(sel); lo, hi = np.quantile(reps, [0.025, 0.975])
        ax.plot([x, x], [lo, hi], color=RAMP[6], lw=1.6, alpha=0.6, zorder=2, solid_capstyle="round")
        ax.plot(x, yv, "o", ms=8.5, color=RAMP[7], mec="white", mew=1.3, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_xlabel("Predicted, frozen before the runs"); ax.set_ylabel("Observed share of runs")
    ax.text(0.62, 0.69, "perfect\ncalibration", fontsize=7.6, color=INK3, rotation=45, ha="center", va="center", fontstyle="italic")
    ax.text(1.06, 0.30, "44.5%", fontsize=18, fontweight="bold", color=INK, transform=ax.transAxes)
    ax.text(1.06, 0.22, "predicted", fontsize=8.5, color=INK2, transform=ax.transAxes)
    ax.text(1.06, 0.08, "44.7%", fontsize=18, fontweight="bold", color=RAMP[7], transform=ax.transAxes)
    ax.text(1.06, 0.00, "observed", fontsize=8.5, color=INK2, transform=ax.transAxes)
    pb = ax.get_position()
    fig.text(0.02, pb.y1 - 0.005, "b", fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(-0.75, 0.80, "Four unseen grammars\n859 targets · 4,745 runs\nAUC 0.86 · slope 0.90\n\nDots: deciles of the\nfrozen prediction,\n95% intervals", fontsize=8,
            color=INK2, transform=ax.transAxes, va="top", linespacing=1.45)
    headline(fig, "Shortest size predicts what GP finds", "a  The same function under different operators   b  The frozen predictor on unseen grammars", y=0.935)
    save(fig, "fig3v3_predict")


# ------------------------------------------------------------------ Figure 4: crossover
def k50_by(phase, exp, fl, ci=False, B=500):
    """K50 per operator; with ci=True also a 95% target-bootstrap interval (fixed seed)."""
    by = defaultdict(lambda: defaultdict(list))
    for k, r in fixed.results(phase).items():
        if r.get("exp") == exp:
            by[r["grammar"]][r["target_id"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl)))
    est = lambda ts: _k50(np.array([q[0] for t in ts for q in t]), np.array([q[1] for t in ts for q in t]))
    point = {g: est(list(v.values())) for g, v in by.items()}
    if not ci:
        return point
    rng, out = np.random.default_rng(20261016), {}
    for g, v in by.items():
        ts = list(v.values())
        reps = [est([ts[i] for i in rng.integers(0, len(ts), len(ts))]) for _ in range(B)]
        out[g] = (point[g], *np.nanquantile(reps, [0.025, 0.975]))
    return out


def fig4():
    raw = [("Mutation-only (1+4)-ES", k50_by("full_es", "es", rescored("full_es"), ci=True), RAMP[2]),
           ("GP without crossover", k50_by("nox_gp", "confB", set(), ci=True), RAMP[5]),
           ("GP", k50_by("full_gp", "conf", rescored("full_gp"), ci=True), RAMP[8])]
    cols = [(n, {g: v[0] for g, v in k.items()}, c) for n, k, c in raw]
    six = [g for g in FAMILY if g != "F-EDL"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    w = 0.27
    for j, (name, k, c) in enumerate(cols):
        xs = np.arange(len(FAMILY)) + (j - 1) * w
        spread = max(k[g] for g in six) - min(k[g] for g in six)
        ax.bar(xs, [k[g] for g in FAMILY], width=w, color=c, edgecolor="white", linewidth=1.2, zorder=2,
               label=f"{name}   (spread {spread:.1f})")
        lo = [raw[j][1][g][1] for g in FAMILY]; hi = [raw[j][1][g][2] for g in FAMILY]
        ax.errorbar(xs, [k[g] for g in FAMILY], yerr=[np.array([k[g] for g in FAMILY]) - lo, np.array(hi) - [k[g] for g in FAMILY]],
                    fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
    import values as V_                                           # random sampling as a reference (D040)
    rv = V_.load()
    for i, g in enumerate(FAMILY):
        ax.plot([i - 1.6 * w, i + 1.6 * w], [rv[f"random_baseline.per.{g[2:]}.rand"]] * 2, color=INK, lw=1.4,
                ls=(0, (3, 2)), zorder=4, label="Random sampling (2 × 10$^4$ trees)" if i == 0 else None)
    ax.set_xticks(np.arange(len(FAMILY)))
    ax.set_xticklabels([NAMES[g] for g in FAMILY], fontsize=10, color=INK)
    for t, g in zip(ax.get_xticklabels(), FAMILY):
        if g == "F-EDL":
            t.set_color(RED); t.set_fontweight("bold")
    ax.set_ylim(0, 15.2); ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_ylabel("Reach K$_{50}$ at 2 × 10$^4$ evaluations (nodes)")
    ax.yaxis.grid(True, color="#efefef", lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(FAMILY) - 0.4)
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.955), ncol=2, frameon=False, fontsize=8.5, handlelength=1.2,
              columnspacing=1.4, handletextpad=0.5)
    headline(fig, "Crossover lifts GP well above random sampling",
             "Reach per operator under three ways of searching; spread: range of the six non-EDL operators (nodes)", y=0.97)
    save(fig, "fig4v3_crossover")


def main():
    fig1(); fig2(); fig3(); fig4(); print("ok")


if __name__ == "__main__":
    main()
