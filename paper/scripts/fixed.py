"""Run results on the fixed engine: the 125 EDL runs that crashed in the first study (68 at E0, 57 at 4E0/16E0) are
replaced by their reruns under the generation-0 fix (phase verify_fix; supplement S1 items 2 and 7). The registered
analyses (Table 1) keep the crashed runs as failures; the post hoc analyses use these results."""
from functools import lru_cache

from reach.runner.queue import Phase


@lru_cache(maxsize=None)
def _reruns():
    return {k: r for k, r in Phase("verify_fix").results().items() if r.get("source")}


def results(phase):
    """{key: record} for a phase, with every error run replaced by its verify_fix rerun."""
    vf = _reruns()
    out = {}
    for k, r in Phase(phase).results().items():
        if r.get("terminal_state") == "error":
            r = vf[f"{phase}|{k}"]
        out[k] = r
    return out
