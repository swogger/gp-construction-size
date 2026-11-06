"""Post hoc (not preregistered): why is EDL the exception? Share of GP-style random trees (ramped half-and-half,
the GP's own initialiser and depth limit) that are defined (finite) on the definition grid, per operator.
-> paper/results/edl_mechanism.md"""
import random
from pathlib import Path

import numpy as np

from reach import grammar as G
from reach.engines.treegp import MAX_DEPTH, _PATCHED, _patch
from reach.family import FAMILY
from reach.galib.generator import generate_ramped_population
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
N_POP, POP = 50, 100


def main():
    saved = {n: getattr(random, n) for n in _PATCHED}
    L = ["# Post hoc: random-tree definedness by operator", "",
         f"{N_POP} GP initial populations of {POP} per operator (ramped half-and-half, depth limit {MAX_DEPTH}); "
         "a tree is *defined* if it is finite at all 100 grid points, *nowhere defined* if finite at none.", "",
         "| Operator | defined everywhere | nowhere defined | populations with no tree defined anywhere | median size |",
         "|---|---|---|---|---|"]
    V = {}
    try:
        for name in FAMILY:
            g, xs = G.get(name), G.grid_sample()
            every = nowhere = empty_pop = 0
            sizes = []
            for s in range(N_POP):
                _patch(random.Random(1000 + s))
                pop = generate_ramped_population(POP, MAX_DEPTH, g.operators, list(g.constants), list(g.variables),
                                                 seed_variable_leaves=False)
                fin = np.array([np.isfinite(G.evaluate(g, t, xs)) for t in pop])
                every += int(fin.all(axis=1).sum()); nowhere += int((~fin.any(axis=1)).sum())
                empty_pop += int(not fin.any())
                sizes += [G.count_nodes(t) for t in pop]
            n = N_POP * POP
            V[name[2:]] = every / n
            L.append(f"| {name} | {every / n:.3f} | {nowhere / n:.3f} | {empty_pop}/{N_POP} | {int(np.median(sizes))} |")
    finally:
        for k, f in saved.items():
            setattr(random, k, f)
    values.record("edl_mechanism", dict(defined=V, others_min=min(v for k, v in V.items() if k != "EDL"),
                                        others_max=max(v for k, v in V.items() if k != "EDL")))
    out = ROOT / "paper/results/edl_mechanism.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
