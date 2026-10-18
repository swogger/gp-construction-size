"""Post hoc (D041): validating the measurement on which the paper rests.
1. Construction identities at high precision: every verified calculator construction is evaluated with mpmath at 50
   digits at 200 random points of the study domain, against the function it is meant to compute (identity holds
   if the largest relative difference is below 1e-30).
2. Exact recoveries on a fresh grid: every GP or ES run scored as an exact recovery is re-evaluated on 1,000 random
   points of the study domain (not the 100-point grid) against its target (relative difference < 1e-8 everywhere).
3. Sensitivity of shortest sizes to the hashing precision: the seven one-variable grammars are re-enumerated to
   17 nodes with values rounded to 7, 9 and 11 decimal places; for every random target and calculator function of
   at most 17 nodes, the size at which it first appears is compared with the size at the paper's 9 places.
-> paper/results/validate_measure.md, values"""
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import mpmath as mp
import numpy as np

from reach import grammar as G
from reach.enumerate import Enumerator
from reach.family import CALC, CALC2, FAMILY
import fixed  # noqa: E402
import values  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
mp.mp.dps = 50
OPS = {"eml": lambda a, b: mp.exp(a) - mp.log(b), "edl": lambda a, b: mp.exp(a) / mp.log(b),
       "neml": lambda a, b: mp.log(a) - mp.exp(b), "lde": lambda a, b: mp.log(a) / mp.exp(b),
       "pli": lambda a, b: mp.exp(mp.log(mp.log(a)) / b), "plm": lambda a, b: mp.exp(-b * mp.log(mp.log(a)))}
TARGET = {"exp(x)": lambda x: mp.exp(x), "ln(x)": lambda x: mp.log(x), "-x": lambda x: -x, "1/x": lambda x: 1 / x,
          "x^2": lambda x: x ** 2, "sqrt(x)": mp.sqrt, "x+1": lambda x: x + 1, "x-1": lambda x: x - 1, "2x": lambda x: 2 * x,
          "x/2": lambda x: x / 2, "exp(-x)": lambda x: mp.exp(-x), "x^x": lambda x: x ** x, "exp(exp(x))": lambda x: mp.exp(mp.exp(x)),
          "ln(ln(x))": lambda x: mp.log(mp.log(x)), "x+y": lambda x, y: x + y, "x-y": lambda x, y: x - y, "x*y": lambda x, y: x * y,
          "x/y": lambda x, y: x / y, "x^y": lambda x, y: x ** y, "log_x(y)": lambda x, y: mp.log(y) / mp.log(x),
          "(x+y)/2": lambda x, y: (x + y) / 2, "sqrt(x^2+y^2)": lambda x, y: mp.sqrt(x * x + y * y)}
assert set(TARGET) == set(CALC) | set(CALC2)


_UN = {"x": lambda a: a, "e": mp.exp, "l": mp.log}
_BIN = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        "pow": lambda a, b: mp.exp(b * mp.log(a))}


def op_fn(name):
    """Family operators, or a screened form A(x) o B(y) named like 'esubl' (reach.grammar)."""
    if name in OPS:
        return OPS[name]
    a, o, b = _UN[name[0]], _BIN[name[1:-1]], _UN[name[-1]]
    return lambda x, y: o(a(x), b(y))


def mp_eval(node, env):
    if node.is_operator:
        return op_fn(node.value)(mp_eval(node.left, env), mp_eval(node.right, env))
    v = node.value
    if isinstance(v, str):
        return env[v]
    if v == float("-inf"):
        return mp.ninf
    if isinstance(v, float) and abs(v - float(mp.e)) < 1e-15:
        return +mp.e                                       # the grammar constant e, exactly
    return mp.mpf(v) if not isinstance(v, complex) else mp.mpc(v)


def identities():
    rng = np.random.default_rng(20261018)
    rows = []
    for op in FAMILY:
        for gname in (op, "F2-" + op[2:]):
            p = ROOT / f"data/family/minimal_{gname}.json"
            if not p.exists():
                continue
            g = G.get(gname)
            for fn, hit in json.loads(p.read_text())["table"].items():
                if fn not in TARGET or not hit or not hit.get("verified"):
                    continue
                tree, worst = G.parse_expr(hit["expr"], g), mp.mpf(0)
                for _ in range(200):
                    env = {v: mp.mpf(2) + mp.mpf(float(rng.random())) for v in g.variables}
                    c, t = mp_eval(tree, env), TARGET[fn](*[env[v] for v in g.variables])
                    worst = max(worst, abs(c - t) / max(1, abs(t)))
                rows.append((op, fn, hit["K"], float(worst)))
    return rows


def fresh_grid():
    targets = {}
    for d in (ROOT / "data/targets").iterdir():
        for f in d.glob("*.json"):
            if f.name.endswith("_log.json"):
                continue
            for t in json.loads(f.read_text()):
                if isinstance(t, dict) and "target_id" in t:
                    targets[t["target_id"]] = t
    rng = np.random.default_rng(20261018)
    x1 = rng.uniform(G.DOMAIN_LO, G.DOMAIN_HI, 1000).astype(np.complex128)
    x2 = {v: rng.uniform(G.DOMAIN_LO, G.DOMAIN_HI, 1000).astype(np.complex128) for v in ("x", "y")}
    out, cache = Counter(), {}
    phases = ["full_gp", "full_budget", "full_budget64", "full_es", "nox_gp", "heldout_gp", "pen_gp", "pen_nox", "pen_es",
              "hc_gp", "restart_seeds", "verify_fix"]
    for ph in phases:
        p = ROOT / f"data/rescored/{ph}.json"
        fl = set(json.loads(p.read_text())) if p.exists() else set()
        for k, r in fixed.results(ph).items():
            if not (r.get("exact_hash_match") == "true" or k in fl) or not r.get("best_expr"):
                continue
            g = G.get(r["grammar"])
            xs = x1 if len(g.variables) == 1 else {v: x2[v] for v in g.variables}
            tid = r["target_id"]
            if tid not in cache:
                cache[tid] = G.evaluate(g, G.parse_expr(targets[tid]["expression"], g), xs)
            t = cache[tid]
            c = G.evaluate(g, G.parse_expr(r["best_expr"], g), xs)
            ok = np.isfinite(t)
            with np.errstate(all="ignore"):
                rel = np.abs(c[ok] - t[ok]) / max(1.0, float(np.abs(t[ok]).max()))
            fin = np.isfinite(rel)
            if fin.all() and rel.max() < 1e-8:
                kind = "exact"
            elif fin.mean() >= 0.99 and rel[fin].max() < 1e-5:
                kind = "numerical"                        # equal up to rounding, or undefined at isolated points
            else:
                kind = "different"
            out[(ph, kind)] += 1
    return out


def precision(kmax=17):
    rows = defaultdict(dict)
    for op in FAMILY:
        g = G.get(op)
        xs = G.sample_grid(g)
        items = {f"calc:{fn}": G.evaluate(g, G.parse_expr(h["expr"], g), xs)
                 for fn, h in json.loads((ROOT / f"data/family/minimal_{op}.json").read_text())["table"].items()
                 if fn in CALC and h and h.get("verified") and h["K"] <= kmax}
        items.update({t["target_id"]: G.evaluate(g, G.parse_expr(t["expression"], g), xs)
                      for t in json.loads((ROOT / f"data/targets/{op}/targets_conf.json").read_text()) if t["K_construction"] <= kmax})
        for prec in (7, 9, 11):
            old, G.HASH_PRECISION = G.HASH_PRECISION, prec
            tmp = Path(tempfile.mkdtemp())
            try:
                e = Enumerator(g, tmp, log=lambda *_: None)
                e.run(kmax, max_gb=6)
                for name, vec in items.items():
                    rows[(op, name)][prec] = e.min_size(vec)
            finally:
                G.HASH_PRECISION = old
                shutil.rmtree(tmp, ignore_errors=True)
    return rows


def fp_dependent():
    """Targets whose double-precision values on the grid differ from their exact (50-digit) values (relative > 1e-6 at
    any grid point): the function the GP evaluator computes there depends on floating-point rounding (D043)."""
    out = {}
    for d in sorted((ROOT / "data/targets").iterdir()):
        f = d / "targets_conf.json"
        if not f.exists() or not (d.name.startswith("F-") or d.name.startswith("S-")):
            continue
        g = G.get(d.name)
        grid = G.sample_grid(g)
        for t in json.loads(f.read_text()):
            tr = G.parse_expr(t["expression"], g)
            dv = G.evaluate(g, tr, grid)
            worst = 0.0
            for x, v in zip(grid, dv):
                m = complex(mp_eval(tr, {"x": mp.mpf(x.real)}))
                worst = max(worst, abs(m - v) / max(1.0, abs(v)))
                if worst > 1e-6:
                    break
            out[t["target_id"]] = not worst < 1e-6
    return out


def robustness(flag):
    """Key estimates on all random targets and on the targets whose values do not depend on rounding."""
    from reach.analysis.family_full import _k50
    rs = lambda ph: set(json.loads((ROOT / f"data/rescored/{ph}.json").read_text())) if (ROOT / f"data/rescored/{ph}.json").exists() else set()
    def pts(ph, exp, ops, keep):
        fl = rs(ph)
        a = [(float(r["K"]), float(r.get("exact_hash_match") == "true" or k in fl)) for k, r in fixed.results(ph).items()
             if r.get("exp") == exp and r["grammar"] in ops and keep(r["target_id"])]
        return np.array([p[0] for p in a]), np.array([p[1] for p in a])
    six = [g for g in FAMILY if g != "F-EDL"]
    res = {}
    for lab, keep in (("all", lambda t: True), ("exact", lambda t: not flag.get(t, False))):
        k6 = _k50(*pts("full_gp", "conf", six, keep))
        res[lab] = dict(six=k6, edl=_k50(*pts("full_gp", "conf", ["F-EDL"], keep)) - k6,
                        crossover=k6 - _k50(*pts("nox_gp", "confB", six, keep)),
                        slope=float(np.polyfit(*pts("full_gp", "conf", FAMILY, keep), 1)[0]))
        pred = json.loads((ROOT / "data/heldout/predictions.json").read_text())
        y, p = [], []
        for r in fixed.results("heldout_gp").values():
            if r.get("exp") in ("conf", "calc", "calc2") and keep(r["target_id"]):
                y.append(float(r.get("exact_hash_match") == "true")); p.append(pred[r["target_id"]]["p"])
        from scipy.stats import rankdata
        y, rk = np.array(y), rankdata(p); n1 = y.sum(); n0 = len(y) - n1
        res[lab]["auc"] = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return res


def main():
    ids = identities()
    fg = fresh_grid()
    pr = precision()
    flag = fp_dependent()
    rob = robustness(flag)
    fam = {g: [t for t in flag if t.startswith(g + "_c_")] for g in FAMILY}
    share = {g[2:]: sum(flag[t] for t in v) / len(v) for g, v in fam.items()}
    held_flag = [t for t in flag if t.startswith("S-")]
    held = [r for r in ids if r[3] < 1e-30]
    same7 = [k for k, v in pr.items() if v.get(7) == v.get(9)]
    same11 = [k for k, v in pr.items() if v.get(11) == v.get(9)]
    diff = {k: v for k, v in pr.items() if len(set(v.values())) > 1}
    kinds = Counter()
    for (ph, kind), n in fg.items():
        kinds[kind] += n
    L = ["# Post hoc: validating the measurement", "",
         f"**Construction identities (mpmath, 50 digits, 200 random points of the study domain):** {len(held)} of {len(ids)} "
         "verified calculator constructions agree with the function they are meant to compute to better than 10⁻³⁰"
         + ("." if len(held) == len(ids) else "; exceptions: " + ", ".join(f"{o[2:]} {f} ({k} nodes; largest relative difference {w:.1e})"
                                                    for o, f, k, w in ids if w >= 1e-30)
            + ". That construction equals e⁻ˣ only in floating-point arithmetic: on (2, e) an intermediate value that is exactly real "
           "carries rounding noise in its imaginary part, whose sign selects the branch of the next logarithm."), "",
         f"**Exact recoveries on a fresh grid (1,000 random points of the study domain):** of {sum(kinds.values())} runs scored as "
         f"exact, {kinds['exact']} agree with the target to 10⁻⁸ everywhere, {kinds['numerical']} agree up to rounding (10⁻⁵) or are "
         f"undefined at isolated points, and {kinds['different']} differ from it: they matched on the grid only.", "",
         f"**Shortest sizes against the hashing precision (re-enumeration to 17 nodes, seven grammars, {len(pr)} functions):** "
         f"unchanged for {len(same7)} at 7 decimal places and {len(same11)} at 11 (against 9)."
         + ("" if not diff else " Changed: " + "; ".join(f"{o[2:]} {n}: " + ", ".join(f"{p}→{s}" for p, s in sorted(v.items())) for (o, n), v in list(diff.items())[:20]) + "."), ""]
    nf = sum(flag[t] for v in fam.values() for t in v); nt = sum(len(v) for v in fam.values())
    L += [f"**Targets whose function depends on floating-point rounding:** for {nf} of {nt} random targets of the first study "
          f"({', '.join(f'{k} {100 * x:.0f}%' for k, x in share.items())}) and {sum(flag[t] for t in held_flag)} of {len(held_flag)} "
          "held-out targets, the double-precision values on the grid differ from the exact (50-digit) values of the same tree "
          "(relative difference above 10⁻⁶ at some grid point): rounding noise decides a branch of the complex logarithm, or the "
          "evaluation is ill-conditioned. GP, enumeration and the exact-recovery check all use the same double-precision "
          "evaluator, so the study is consistent, but for these targets the function is defined by that evaluator, not by "
          "exact complex arithmetic. Key estimates without them:", "",
          "| Estimate | all targets | targets independent of rounding |", "|---|---|---|",
          f"| GP K₅₀, pooled six | {rob['all']['six']:.2f} | {rob['exact']['six']:.2f} |",
          f"| EDL − pooled six | {rob['all']['edl']:+.2f} | {rob['exact']['edl']:+.2f} |",
          f"| crossover (GP − GP without crossover, pooled six) | {rob['all']['crossover']:+.2f} | {rob['exact']['crossover']:+.2f} |",
          f"| recovery against size, linear slope per node (all seven) | {rob['all']['slope']:+.3f} | {rob['exact']['slope']:+.3f} |",
          f"| H6 AUC on the held-out grammars | {rob['all']['auc']:.3f} | {rob['exact']['auc']:.3f} |", ""]
    (ROOT / "paper/results/validate_measure.md").write_text("\n".join(L) + "\n")
    values.record("validate_measure", dict(id_held=len(held), id_total=len(ids), id_worst=max(r[3] for r in ids),
                                           fresh_total=sum(kinds.values()), fresh_exact=kinds["exact"], fresh_numerical=kinds["numerical"],
                                           fresh_different=kinds["different"],
                                           fp_n=nf, fp_total=nt, fp_share=share, fp_held=sum(flag[t] for t in held_flag),
                                           fp_held_total=len(held_flag), rob=rob,
                                           rob_maxchange=max(abs(rob["exact"][k] - rob["all"][k]) for k in ("six", "edl", "crossover")),
                                           prec_total=len(pr), prec_same7=len(same7), prec_same11=len(same11)))
    print("\n".join(L))


if __name__ == "__main__":
    main()
