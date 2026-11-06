"""Rerun every analysis behind the paper from the released data and check that each quoted number is reproduced.

Three steps, all from the released files:
1. The registered analyses (reach.analysis.family_full, reach.analysis.heldout_analysis) and the rerun of the first
   on the corrected outcomes (reach.analysis.rescore). Their reports must match registration/analysis_report_checksums.txt.
2. Every post hoc analysis, table and figure script (paper/scripts/), in dependency order.
3. A comparison of the regenerated values with the released ones (paper/results/values/*.json).

The script makes this copy's src/ importable and points the code at this copy, so nothing needs installing beyond
requirements.txt. Reports are written to docs/results/ and paper/results/ as local output. Run from anywhere with the
pinned environment:

    python scripts/reproduce_results.py
"""
import hashlib
import json
import os
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
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "REACH_ROOT": str(ROOT)}


def run(args, label):
    r = subprocess.run([sys.executable, "-W", "ignore", *args], cwd=ROOT, env=ENV, capture_output=True, text=True)
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {label}", flush=True)
    if r.returncode:
        print(r.stderr[-1500:])
        sys.exit(1)


def main():
    (ROOT / "docs/results").mkdir(parents=True, exist_ok=True)
    vals = ROOT / "paper/results/values"
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "values"
        shutil.copytree(vals, saved)
        run(["-m", "reach.analysis.family_full"], "registered analysis, first study")
        run(["-m", "reach.analysis.rescore"], "registered analysis rerun on the corrected outcomes")
        run(["-m", "reach.analysis.heldout_analysis"], "registered analysis, second study")
        bad = []
        for line in (ROOT / "registration/analysis_report_checksums.txt").read_text().splitlines():
            want, name = line.split()
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != want:
                bad.append(f"{name}: report differs from the released checksum")
        print(f"{'FAIL' if bad else 'ok  '} registered reports match the released checksums", flush=True)
        for s in ORDER:
            run([f"paper/scripts/{s}.py"], s)
        for f in sorted(saved.glob("*.json")):
            a, b = json.loads(f.read_text()), json.loads((vals / f.name).read_text())
            bad += [f"{f.stem}.{k}: released {a.get(k)} regenerated {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    print("every report and quoted value reproduced" if not bad else f"{len(bad)} difference(s):\n  " + "\n  ".join(bad[:50]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
