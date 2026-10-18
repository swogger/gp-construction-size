"""Supplement S2: shortest construction (nodes) of constants under each operator (one-variable grammars; enumeration
tables). -> paper/results/table_constants.md"""
import json
from pathlib import Path

from reach.family import CONSTANTS, FAMILY

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}


def main():
    tabs = {g: json.loads((ROOT / f"data/family/minimal_{g}.json").read_text())["table"] for g in FAMILY}
    L = ["| Constant | " + " | ".join(NAMES[g] for g in FAMILY) + " |", "|---|" + "---|" * len(FAMILY)]
    for c in CONSTANTS:
        row = [tabs[g].get(c) for g in FAMILY]
        L.append(f"| {c} | " + " | ".join(str(h['K']) if h and h.get('verified') else "–" for h in row) + " |")
    out = ROOT / "paper/results/table_constants.md"
    out.write_text("\n".join(L) + "\n\n–: none within the enumeration ceiling. A grammar's own constant is a single leaf (1 node).\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
