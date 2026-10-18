"""Tools for Section 5 (one per subsection), generated from frozen data and results files.
-> paper/results/tools.md with blocks <!-- TOOL name --> ... <!-- /TOOL --> that assemble.py inserts."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from reach.analysis.family_full import _k50
from reach.family import FAMILY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from exchange_rate import enum_k50  # noqa: E402
import fixed  # noqa: E402

NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def flips(ph):
    p = ROOT / f"data/rescored/{ph}.json"
    return set(json.loads(p.read_text())) if p.exists() else set()


def k50s(phase, exp):
    fl = flips(phase)
    by = defaultdict(list)
    for k, r in fixed.results(phase).items():
        if r.get("exp") == exp:
            by[r["grammar"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl)))
    return {g: _k50(np.array([p[0] for p in v]), np.array([p[1] for p in v])) for g, v in by.items()}


def calc_rate(phase, exp, grammar, fn):
    fl = flips(phase)
    v = [r.get("exact_hash_match") == "true" or k in fl for k, r in fixed.results(phase).items()
         if r.get("exp") == exp and r["grammar"] == grammar and r["target_id"].endswith("_calc_" + fn)]
    return sum(v), len(v)


def block(name, body):
    return f"<!-- TOOL {name} -->\n{body.strip()}\n<!-- /TOOL -->\n"


def main():
    gp, nox, es = k50s("full_gp", "conf"), k50s("nox_gp", "confB"), k50s("full_es", "es")
    pen = k50s("pen_gp", "confP")
    out = []

    # 5.1 budget estimator (Equation 3), worked example
    s0, n0 = calc_rate("full_gp", "calc", "F-EML", "x^2"); s16, n16 = calc_rate("full_budget", "budget_calc", "F-EML", "x^2")
    m0, mn0 = calc_rate("full_gp", "calc", "F-EML", "-x")
    out.append(block("budget", f"""
**Algorithm 1.** Enumerate or search? For a target whose form is known (benchmark design, or a model class anticipated in advance).

*Input:* a single-operator grammar G, a known target function t, a budget E. *Output:* enumerate, search, or change the operator.

1. Enumerate G and read off K*, the shortest construction of t (Figure 2).
2. If exhaustive enumeration covers K* within E evaluations (Table 3), enumerate: it finds t with certainty.
3. Otherwise estimate GP's chance per run from the size curve (Figure 3a) or the predictor of Section 4.2, and spend E on independent short runs rather than one long run (Section 4.3).
4. If K* lies beyond both, choose an operator that makes t shorter (Figure 2).

<!-- /algo -->

For example, x² under EML has K* = 17. With a budget of 1.3 × 10⁶ evaluations (64E₀), exhaustive enumeration covers 17 nodes (Table 3), so step 2 applies; at E₀ it does not, and step 3 applies. Consistently, x² was never found by GP under EML at E₀ ({s0} of {n0} runs) or at 16E₀ ({s16} of {n16}). The size curve describes a typical target of a given size, and individual functions vary around it: −x, 15 nodes under EML and so expected to be found rarely at E₀, was found in {m0} of {mn0} runs.
"""))
    # 5.2 crossover table
    six = [g for g in FAMILY if g != "F-EDL"]
    rows = "\n".join(f"| {NAMES[g]} | {gp[g]:.2f} | {nox[g]:.2f} | +{gp[g] - nox[g]:.2f} |" for g in FAMILY)
    sp = lambda d: max(d[g] for g in six) - min(d[g] for g in six)
    out.append(block("crossover", f"""
**Table 4.** What crossover adds, per operator (K₅₀ at E₀, nodes; same targets and seeds).

| Operator | GP | GP without crossover | Gain from crossover |
|---|---|---|---|
{rows}
| Spread of the six non-EDL operators | {sp(gp):.2f} | {sp(nox):.2f} | |
"""))

    # 5.3 EDL profile
    def U(g, k):
        return {l["K"]: l for l in json.loads((ROOT / f"data/enumeration/{g}/counts.json").read_text())["levels"]}[k]["U"]
    var = [t for g in ("F-EML", "F-EDL") for t in json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())]
    def md_table(name):                                   # {row label: {column: cell}} of a results table
        ls = [l for l in (ROOT / f"paper/results/{name}.md").read_text().splitlines() if l.startswith("| ")]
        head = [c.strip() for c in ls[0].split("|")[1:-1]]
        return {r[0]: dict(zip(head[1:], r[1:])) for r in ([c.strip(" *") for c in l.split("|")[1:-1]] for l in ls[1:])}
    t3, tc = md_table("table3"), md_table("table_constants")
    cover = lambda op: sum(row[op] != "–" for row in t3.values())
    consts = lambda op: ", ".join(tc[c][op] for c in ("0", "-1", "2"))
    hv = lambda g: f"{100 * np.mean([t['var_target'] > 1000 for t in var if t['target_id'].startswith(g + '_c_')]):.0f}%"
    nb = {g: np.mean([x["nb_good"] for x in json.loads((ROOT / f"data/family/locality_conf_{g}.json").read_text())]) for g in ("F-EML", "F-EDL")}
    out.append(block("edl", f"""
**Table 4.** A diagnostic profile: EML and EDL on every enumeration-derived measure, and in search (K₅₀ at E₀, fixed engine).

| Measure | EML | EDL |
|---|---|---|
| Distinct functions of size 13 / 15 | {U('F-EML', 13):,} / {U('F-EML', 15):,} | {U('F-EDL', 13):,} / {U('F-EDL', 15):,} |
| Size exhaustive enumeration reaches at E₀ (K₅₀) | {enum_k50('F-EML'):.1f} | {enum_k50('F-EDL'):.1f} |
| Calculator functions constructed within the ceilings | {cover('EML')} of 22 | {cover('EDL')} of 22 |
| Shortest constructions: exp, ln, 1/x, x², x − 1 | 3, 7, 15, 17, 11 | 3, 7, 17, 19, 11 |
| Shortest constructions of the constants 0, −1, 2 | {consts('EML')} | {consts('EDL')} |
| Random trees defined everywhere on the domain | 22% | 16% |
| Random targets with variance above 1,000 (Section 4.4) | {hv('F-EML')} | {hv('F-EDL')} |
| Near-correct one-edit neighbours (mean share) | {nb['F-EML']:.3f} | {nb['F-EDL']:.3f} |
| GP reach K₅₀ | {gp['F-EML']:.2f} | {gp['F-EDL']:.2f} |
| GP reach K₅₀, penalty scaled to the target (Section 4.4) | {pen['F-EML']:.2f} | {pen['F-EDL']:.2f} |
| GP without crossover | {nox['F-EML']:.2f} | {nox['F-EDL']:.2f} |
| (1+4)-ES | {es['F-EML']:.2f} | {es['F-EDL']:.2f} |
"""))

    # 5.4 decision matrix (streamlined)
    fl = flips("full_gp")
    rate = defaultdict(lambda: [0, 0])
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") == "conf":
            c = rate[int(r["K"])]; c[0] += r.get("exact_hash_match") == "true" or k in fl; c[1] += 1
    band = lambda ks: f"{100 * sum(rate[k][0] for k in ks) / sum(rate[k][1] for k in ks):.0f}%"
    out.append(block("decision", f"""
**Table 5.** Search or enumerate? A decision matrix for a target whose form is known, in a single-operator grammar (GP success: one run of 2 × 10⁴ evaluations, random targets of the first study, all seven operators; the same budget spent on short restarts reaches about 2.5 nodes further, Section 4.3; enumeration: exact, range over operators).

| Target's shortest construction | GP finds it | Exhaustive enumeration | Do this |
|---|---|---|---|
| up to 13 nodes | {band([3, 5, 7, 9])} (≤ 9), {band([11, 13])} (11–13) | certain at 2 × 10⁴; under a second | **enumerate** |
| 15–19 nodes | {band([15])} (15), {band([17, 19])} (17–19) | certain within 1.3 × 10⁶ (19 nodes: five of the seven operators); seconds | **enumerate, with a larger budget** |
| over 19 nodes | rare; each node ≈ 10× the length of a single run, ≈ 2× the budget with short restarts | beyond our ceilings | **change operator** to one that makes the target shorter (Figure 2) |
"""))
    (ROOT / "paper/results/tools.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
