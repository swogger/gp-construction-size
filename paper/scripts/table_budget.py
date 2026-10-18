"""Table 3: reach by budget. GP K50 on the same targets at every budget (from paper/results/fig_reach.md, 95%
target-bootstrap intervals) beside the largest size exhaustive enumeration covers completely at the same number of
evaluations (exact, from data/enumeration/*/counts.json; range over the seven operators). -> paper/results/table_budget.md"""
import json
import re
from pathlib import Path

from reach.family import FAMILY

ROOT = Path(__file__).resolve().parents[2]


def main():
    gp = {}
    for line in (ROOT / "paper/results/fig_reach.md").read_text().splitlines():
        m = re.match(r"\| (seven operators|EML, LDE \(c=0\), PLM)[^|]* \| (\d+)× \| ([\d.]+) \[([\d.]+), ([\d.]+)\] \|", line)
        if m:
            gp[(m.group(1)[:5], int(m.group(2)))] = f"{m.group(3)} [{m.group(4)}, {m.group(5)}]"
    full = {}
    for g in FAMILY:
        lv = json.loads((ROOT / f"data/enumeration/{g}/counts.json").read_text())["levels"]
        cum = 0
        for l in lv:
            cum += l["candidates"]
            for m in (1, 4, 16, 64):
                if cum <= 2e4 * m and l["K"] % 2 == 1:
                    full.setdefault(m, {})[g] = l["K"]
    lab = {1: "E₀ = 2 × 10⁴", 4: "4E₀", 16: "16E₀", 64: "64E₀"}
    L = ["| Budget (evaluations) | GP K₅₀, seven operators, sizes 9–17 | GP K₅₀, EML, LDE₀, PLM, sizes 11–17 | Enumeration finds every function up to |",
         "|---|---|---|---|"]
    for m in (1, 4, 16, 64):
        v = full[m].values()
        L.append(f"| {lab[m]} | {gp.get(('seven', m), '–')} | {gp.get(('EML, ', m), '–')} | {min(v)}–{max(v)} nodes |")
    (ROOT / "paper/results/table_budget.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
