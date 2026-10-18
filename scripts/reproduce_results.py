"""Rerun every analysis behind the paper from the released data and check that each quoted number is reproduced.

The analyses write paper/results/*.md (reports) and paper/results/values/*.json (every number the paper quotes).
This script runs them in dependency order and compares the regenerated values with the released ones.
Run from the repository root with the pinned environment (requirements.txt):

    python scripts/reproduce_results.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["validate_measure", "size_wall", "exchange_rate", "law", "fig_reach", "table_budget", "curve_shapes",
         "within_function", "size_vs_operator", "enumeration_frontier", "domain_check", "edl_mechanism", "edl_artefact",
         "h6_intervals", "sensitivity", "restarts", "random_baseline", "target_filters", "table3", "table_constants",
         "discussion_tools", "figs_v3"]


def main():
    vals = ROOT / "paper/results/values"
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "values"
        shutil.copytree(vals, saved)
        for s in ORDER:
            r = subprocess.run([sys.executable, "-W", "ignore", f"paper/scripts/{s}.py"], cwd=ROOT, capture_output=True, text=True)
            print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {s}")
            if r.returncode:
                print(r.stderr[-1500:])
                sys.exit(1)
        bad = []
        for f in sorted(saved.glob("*.json")):
            a, b = json.loads(f.read_text()), json.loads((vals / f.name).read_text())
            bad += [f"{f.stem}.{k}: released {a.get(k)} regenerated {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print("every quoted value reproduced" if not bad else f"{len(bad)} value(s) differ:\n  " + "\n  ".join(bad[:50]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
