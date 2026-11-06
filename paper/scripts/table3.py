"""Table 3: shortest construction (nodes) of each calculator function under each operator, from the enumeration
tables (data/family/minimal_<op>.json; proven on the study grid within each ceiling). '–' = none within the ceiling
(one variable: 21 for EML and EDL, 23 for the others; two variables: 17). -> paper/results/table3.md"""
import json
from pathlib import Path

from reach.family import CALC, CALC2, FAMILY

ROOT = Path(__file__).resolve().parents[2]
PRETTY = {"exp(x)": "eˣ", "ln(x)": "ln x", "-x": "−x", "1/x": "1/x", "x^2": "x²", "sqrt(x)": "√x", "x+1": "x + 1",
          "x-1": "x − 1", "2x": "2x", "x/2": "x/2", "exp(-x)": "e⁻ˣ", "x^x": "xˣ", "exp(exp(x))": "exp(eˣ)",
          "ln(ln(x))": "ln(ln x)", "x+y": "x + y", "x-y": "x − y", "x*y": "x·y", "x/y": "x/y", "x^y": "xʸ",
          "log_x(y)": "log_x y", "(x+y)/2": "(x + y)/2", "sqrt(x^2+y^2)": "√(x² + y²)"}
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def main():
    tab = {}
    for g in FAMILY:
        for p in (g, "F2-" + g[2:]):
            t = json.loads((ROOT / f"data/family/minimal_{p}.json").read_text())["table"]
            for fn, hit in t.items():
                tab[(g, fn)] = hit["K"] if hit and hit.get("verified") else None
    L = ["| Function | " + " | ".join(NAMES[g] for g in FAMILY) + " |", "|---|" + "---|" * len(FAMILY)]
    for fn in list(CALC) + list(CALC2):
        row = [tab.get((g, fn)) for g in FAMILY]
        best = min([k for k in row if k is not None], default=None)
        cells = [("**%d**" % k if k == best else str(k)) if k is not None else "–" for k in row]
        L.append(f"| {PRETTY.get(fn, fn)} | " + " | ".join(cells) + " |")
    out = ROOT / "paper/results/table3.md"
    out.write_text("\n".join(L) + "\n\nBold: shortest across operators. '–': no construction within the enumeration ceiling.\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
