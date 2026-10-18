"""Supplement S7 (post hoc): GP against exhaustive enumeration at E0, on the same basis as the main text.
GP K50: all random targets of the first study at E0 (corrected scoring; as Table 1). Enumeration K50: the size at
which exhaustive enumeration finds a random target with probability 1/2 within E0 = 2e4 candidate evaluations
(paper/scripts/exchange_rate.py, enum_k50). Gap = enumeration K50 - GP K50 (the basis of the enumeration bar of
Figure 1). CPU: enumeration to complete sizes 13, 15, 17 (data/enumeration/*/counts.json) and GP per run at E0.
-> paper/results/enumeration_frontier.md"""
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
import values  # noqa: E402

NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)", "F-PLI": "PLI", "F-PLM": "PLM"}


def main():
    fl = set(json.loads((ROOT / "data/rescored/full_gp.json").read_text()))
    pts, cpu = defaultdict(list), defaultdict(list)
    for k, r in fixed.results("full_gp").items():
        if r.get("exp") == "conf":
            pts[r["grammar"]].append((float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl)))
            if r["terminal_state"] != "error":
                cpu[r["grammar"]].append(float(r["cpu_s"]))
    L = ["One GP evaluation is one tree scored on a 75-point batch; one enumeration evaluation is one candidate tree "
         "evaluated on the 100-point grid.", "",
         "| Operator | GP K₅₀ at E₀ | Enumeration K₅₀ at E₀ | Gap (nodes) | Enumeration CPU to complete sizes 13 / 15 / 17 | GP CPU per run at E₀ (mean) |",
         "|---|---|---|---|---|---|"]
    gaps, c17, gpcpu, growth = [], [], [], {}
    for g in FAMILY:
        gk = _k50(np.array([p[0] for p in pts[g]]), np.array([p[1] for p in pts[g]]))
        ek = enum_k50(g); gaps.append(ek - gk)
        lv = json.loads((ROOT / f"data/enumeration/{g}/counts.json").read_text())["levels"]
        c = np.cumsum([l["cpu_s"] for l in lv]); ct = {l["K"]: c[i] for i, l in enumerate(lv)}
        c17.append(ct[17]); gpcpu.append(np.mean(cpu[g]))
        cand = {l["K"]: l["candidates"] for l in lv}
        growth[g] = [(cand[k + 2] / cand[k]) ** 0.5 for k in (13, 15, 17) if k + 2 in cand and cand[k] > 0]
        L.append(f"| {NAMES[g]} | {gk:.2f} | {ek:.2f} | +{ek - gk:.2f} | {ct[13]:.1f} / {ct[15]:.1f} / {ct[17]:.1f} s | {np.mean(cpu[g]):.1f} s |")
    allg = [x for v in growth.values() for x in v]
    L += ["", f"Gaps range from {min(gaps):.1f} to {max(gaps):.1f} nodes (median {np.median(gaps):.2f}).", "",
          "Growth of enumeration's cost per node (candidates per level, square root of the ratio per two nodes, 13 to 19 nodes): "
          + ", ".join(f"{NAMES[g]} {min(v):.2f}–{max(v):.2f}×" for g, v in growth.items() if v) + "."]
    (ROOT / "paper/results/enumeration_frontier.md").write_text("\n".join(L) + "\n")
    values.record("enumeration_frontier", dict(enum17_min=min(c17), enum17_max=max(c17), gpcpu_min=min(gpcpu), gpcpu_max=max(gpcpu),
                                               growth_min=min(allg), growth_max=max(allg)))
    print("\n".join(L))


if __name__ == "__main__":
    main()
