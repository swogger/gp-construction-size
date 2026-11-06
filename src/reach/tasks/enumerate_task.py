"""Enumerate one grammar up to kmax (or until the memory ceiling / CPU limit) -> RESULT line.
Levels are checkpointed in data/enumeration/<grammar>/, so a rerun resumes."""
import argparse
import json

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.runner.queue import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--grammar", required=True)
ap.add_argument("--kmax", type=int, default=31)
ap.add_argument("--max_gb", type=float, default=4.0)
a = ap.parse_args()
e = Enumerator(G.get(a.grammar), ROOT / "data" / "enumeration" / a.grammar, log=lambda s: print(s, flush=True))
k, state = e.run(a.kmax, a.max_gb)
print("RESULT " + json.dumps(dict(grammar=a.grammar, terminal_state=state, k_reached=k,
                                  U_cum=e.stats[-1]["U_cum"], levels=len(e.stats))), flush=True)
