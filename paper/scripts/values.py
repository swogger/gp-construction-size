"""Every number the paper quotes from an analysis: the scripts record them here, and assemble.py fills them into the
text, where they appear as placeholders {{script.key|filter|...:format}} (D037).

- record(script, values) writes paper/results/values/<script>.json (numbers, unformatted).
- render(text) replaces each placeholder by its formatted value. Filters: abs, pct (x100), sig2 (two significant
  figures, thousands separated), fold (2^(value/s) with the exchange rate s). A hyphen-minus in the output becomes
  the typographic minus. An unknown key is an error, so a value can never be silently missing.
- The lock (paper/values.lock.json) holds the rendered text of every placeholder as last accepted. If a value
  changes, assemble.py stops and lists the changes; they are accepted with `assemble.py --accept` after checking
  that the sentences around them still hold (claims.py)."""
import json
import math
import re
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "results/values"
LOCK = Path(__file__).resolve().parents[1] / "values.lock.json"
PH = re.compile(r"\{\{([^|:}]+)((?:\|\w+)*)(?::([^}]+))?\}\}")


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item"):
        v = v.item()
    return v


def record(script, values):
    DIR.mkdir(parents=True, exist_ok=True)
    flat = {}

    def walk(prefix, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        else:
            flat[prefix] = _plain(v)
    walk("", values)
    (DIR / f"{script}.json").write_text(json.dumps(flat, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def load():
    return {f"{p.stem}.{k}": v for p in sorted(DIR.glob("*.json")) for k, v in json.loads(p.read_text()).items()}


def sig2(x):
    if x == 0:
        return "0"
    d = -int(math.floor(math.log10(abs(x)))) + 1
    r = round(x, d)
    return f"{r:,.0f}" if abs(r) >= 10 else f"{r:g}"


def render(text, vals=None):
    vals = load() if vals is None else vals
    used = {}

    def one(m):
        key, filters, fmt = m.group(1), [f for f in m.group(2).split("|") if f], m.group(3)
        if key not in vals:
            raise KeyError(f"no recorded value for {{{{{key}}}}}")
        v = vals[key]
        for f in filters:
            if f == "abs":
                v = abs(v)
            elif f == "pct":
                v = 100 * v
            elif f == "fold":
                v = 2 ** (v / vals["exchange_rate.s"])
            elif f == "sig2":
                v = sig2(v)
            elif f == "sig3":
                d = -int(math.floor(math.log10(abs(v)))) + 2
                v = f"{round(v, d):,.0f}" if abs(round(v, d)) >= 100 else f"{round(v, d):g}"
            else:
                raise ValueError(f"unknown filter {f} in {m.group(0)}")
        out = format(v, fmt) if fmt and not isinstance(v, str) else str(v)
        out = out.replace("-", "−")
        used[m.group(0)] = out
        return out
    return PH.sub(one, text), used


def check_lock(used, accept=False):
    """Compare rendered placeholders with the lock; return the list of changes (and update the lock if accept)."""
    old = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    changes = [(k, old.get(k), v) for k, v in sorted(used.items()) if old.get(k) != v]
    if accept or not old:
        LOCK.write_text(json.dumps(dict(sorted(used.items())), indent=1, ensure_ascii=False) + "\n")
        return []
    return changes
