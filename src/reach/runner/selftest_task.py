"""A fake task for testing the queue: burns CPU, then reports, crashes or never finishes."""
import argparse
import json
import time

from reach.paths import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["ok", "timeout", "error", "noresult"], default="ok")
ap.add_argument("--cpu", type=float, default=0.2, help="CPU seconds to burn first")
a = ap.parse_args()

t0 = time.process_time()
x = 0
while a.mode == "timeout" or time.process_time() - t0 < a.cpu:
    x += 1
if a.mode == "error":
    raise RuntimeError("deliberate failure")
if a.mode == "ok":
    print("RESULT " + json.dumps({"terminal_state": "converged", "module_file": __file__, "root": str(ROOT)}), flush=True)
