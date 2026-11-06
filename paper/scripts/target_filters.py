"""Post hoc (D039): how many drawn candidates each target filter rejected, per operator (random targets; from the
target-generation logs data/targets/<grammar>/targets_conf_log.json). Drawn = kept + rejected.
-> paper/results/target_filters.md, values"""
import json
from collections import Counter
from pathlib import Path

from reach.family import FAMILY
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"F-EML": "EML", "F-EDL": "EDL", "F-NEML": "−EML", "F-LDE0": "LDE (c=0)", "F-LDE1": "LDE (c=1)",
         "F-PLI": "PLI", "F-PLM": "PLM"}
REASONS = [("nonfinite", "not finite"), ("variance", "near-constant"), ("magnitude_ratio", "zero or magnitude ratio"),
           ("dead_subtree", "dead subtree"), ("near_duplicate", "near-duplicate of a smaller function")]


def main():
    L = ["# Post hoc: rejections by the target filters (random targets)", "",
         "Share of drawn candidates rejected by each filter (drawn = kept + rejected), all sizes pooled.", "",
         "| Operator | drawn | kept | " + " | ".join(r[1] for r in REASONS) + " |", "|---|---|---|" + "---|" * len(REASONS)]
    V = {}
    for g in FAMILY:
        log = json.loads((ROOT / f"data/targets/{g}/targets_conf_log.json").read_text())["levels"]
        c, kept = Counter(), 0
        for lv in log.values():
            c.update(lv.get("rejected", {}))
            c["near_duplicate"] += lv.get("near_duplicate_rejected", 0)
            kept += lv.get("kept", 0)
        drawn = kept + sum(c.values())
        L.append(f"| {NAMES[g]} | {drawn:,} | {kept} | " + " | ".join(f"{100 * c[r] / drawn:.0f}%" for r, _ in REASONS) + " |")
        V[g[2:]] = dict(drawn=drawn, kept=kept, **{r: c[r] / drawn for r, _ in REASONS})
    (ROOT / "paper/results/target_filters.md").write_text("\n".join(L) + "\n")
    values.record("target_filters", V)
    print("\n".join(L))


if __name__ == "__main__":
    main()
