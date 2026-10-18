"""Phase definitions. Each builder returns the task list that `plan` freezes into manifest.json.

A task is {key, argv, cwd, cpu_limit, weight, meta}: argv may use "{python}" for the queue's
interpreter, cwd is relative to the repository root, cpu_limit is in CPU seconds, and weight is
the relative expected cost (used only for the ETA).
"""
import importlib.util
import json

from reach import regression
from reach.runner.queue import ROOT

PHASES = {}


def phase(name, description, verify=None):
    def deco(fn):
        PHASES[name] = dict(build=fn, description=description, verify=verify)
        return fn
    return deco


def legacy_seeds():
    spec = importlib.util.spec_from_file_location("legacy_seeds", ROOT / "legacy" / "research" / "seeds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SEEDS


@phase("smoke", "Eight tiny self-test tasks covering every terminal state (about 10 s).")
def smoke():
    tasks = [dict(key=f"ok_{i}", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "ok"],
                  cpu_limit=10) for i in range(5)]
    tasks += [dict(key="timeout", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "timeout"],
                   cpu_limit=2, weight=10),
              dict(key="error", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "error"],
                   cpu_limit=10),
              dict(key="noresult", argv=["{python}", "-m", "reach.runner.selftest_task", "--mode", "noresult"],
                   cpu_limit=10)]
    return tasks


def _verify_against_p1(ph):
    rows = ph.results()
    bad = regression.compare(rows, regression.load_p1())
    print(f"[{ph.name}] {len(rows)} runs compared with P1_synth_main.csv on {', '.join(regression.FIELDS)}")
    if not rows:
        print("no results yet")
        return 1
    for key, f, ours, ref in bad[:30]:
        print(f"  MISMATCH {key} {f}: ours {ours!r} vs stored {ref!r}")
    print("all identical" if not bad else f"{len(bad)} mismatches")
    return 0 if not bad else 1


@phase("legacy_check",
       "Legacy EML engine on every K = 7 and K = 13 target, both regimes, first P1 seed (122 runs, "
       "about 25 CPU-minutes). Checks the queue end to end against stored P1 rows.",
       verify=_verify_against_p1)
def legacy_check():
    seed = legacy_seeds()[1]
    targets = json.loads((ROOT / "legacy/results/synthetic_dense_k/targets.json").read_text())
    tasks = []
    for t in targets:
        if t["K_construction"] not in (7, 13):
            continue
        for reg in ("A", "B"):
            tasks.append(dict(
                key=f"{t['target_id']}|{reg}|{seed}", cwd="legacy",
                argv=["{python}", "-u", "research/rerun/synth_task.py",
                      "--targets", "results/synthetic_dense_k/targets.json",
                      "--target_id", t["target_id"], "--regime", reg, "--seed", str(seed),
                      "--generations", "200", "--timeout", "600", "--no_size_penalty", "--cpu_clock"],
                cpu_limit=700, weight=4.0 if reg == "A" else 1.0,
                meta=dict(target_id=t["target_id"], regime=reg, seed=seed)))
    return tasks


@phase("m1_regression",
       "M1 acceptance: the new engine in legacy mode on every K = 7 and K = 13 EML target, both "
       "regimes, all 10 P1 seeds (1,220 runs, about 4 CPU-hours). Must match P1 exactly.",
       verify=_verify_against_p1)
def m1_regression():
    seeds = legacy_seeds()[1:11]
    targets = json.loads((ROOT / "legacy/results/synthetic_dense_k/targets.json").read_text())
    tasks = []
    for t in targets:
        if t["K_construction"] not in (7, 13):
            continue
        for reg in ("A", "B"):
            for s in seeds:
                tasks.append(dict(
                    key=f"{t['target_id']}|{reg}|{s}",
                    argv=["{python}", "-m", "reach.tasks.treegp_task", "--grammar", "EML",
                          "--targets", "legacy/results/synthetic_dense_k/targets.json",
                          "--target_id", t["target_id"], "--regime", reg, "--seed", str(s),
                          "--generations", "200", "--timeout", "600", "--legacy"],
                    cpu_limit=700, weight=4.0 if reg == "A" else 1.0,
                    meta=dict(target_id=t["target_id"], regime=reg, seed=s)))
    return tasks


ENUM_GRAMMARS = ["EML", "ARITH", "ELEM6", "HYB"] + [
    f"BOOL-{b}-n{n}" for n in (3, 4) for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


@phase("enum_pilot",
       "M4: enumerate every pilot grammar by size until the memory ceiling (8 GB of stored values for "
       "real grammars; peak about 12 GB) or 4 CPU-hours (18 tasks; checkpointed per level). Run with "
       "--workers 2 so peak memory stays within 32 GB. Gives lambda_G and the null.")
def enum_pilot():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "40", "--max_gb", "8" if not g.startswith("BOOL") else "2"],
                 cpu_limit=4 * 3600, weight=1.0, meta=dict(grammar=g)) for g in ENUM_GRAMMARS]


PILOT_GRAMMARS = ["EML", "ARITH", "ELEM6", "HYB"] + [
    f"BOOL-{b}-n4" for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


def _gp_task(exp, g, t, regime, seed, gens, extra=(), weight=1.0, tname="targets"):
    timeout = {200: 600, 800: 2400, 3200: 6000, 12800: 3600 * 6}[gens]
    path = f"data/targets/{g}/{tname}.json"
    tag = "+".join([exp] + [e.strip("-") for e in extra])
    return dict(key=f"{tag}|{t['target_id']}|{regime}|{gens}|{seed}",
                argv=["{python}", "-m", "reach.tasks.treegp_task", "--grammar", g, "--targets", path,
                      "--target_id", t["target_id"], "--regime", regime, "--seed", str(seed),
                      "--generations", str(gens), "--timeout", str(timeout), *extra],
                cpu_limit=timeout + 100, weight=weight,
                meta=dict(exp=exp, grammar=g, target_id=t["target_id"], K=t["K_construction"],
                          K_proven=t["K_proven"], regime=regime, generations=gens, seed=seed,
                          seed_leaves="on" if "--seed_leaves" in extra else "off"))


@phase("pilot_p1",
       "Pilot scout (clean engine, Regime A): every pilot grammar's targets x 3 seeds at E0 = 2e4; "
       "plus probes: E1 = 8e4 (EML, ELEM6; 3 levels), seeded leaf on (EML K11, ELEM6 K10), and a "
       "3,200-generation cost probe. About 3,250 runs, about 20 CPU-hours.")
def pilot_p1():
    from reach.seeds import SEEDS
    tasks = []
    T = {g: json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()) for g in PILOT_GRAMMARS}
    for g in PILOT_GRAMMARS:
        w = 1.0 if g.startswith("BOOL") else 4.0
        for t in T[g]:
            for s in SEEDS[:3]:
                tasks.append(_gp_task("scout", g, t, "A", s, 200, weight=w))
    for g, levels in (("EML", (11, 13, 15)), ("ELEM6", (8, 10, 12))):
        for t in T[g]:
            if t["K_construction"] in levels:
                for s in SEEDS[:3]:
                    tasks.append(_gp_task("e1", g, t, "A", s, 800, weight=16.0))
    for g, k in (("EML", 11), ("ELEM6", 10)):
        for t in T[g]:
            if t["K_construction"] == k:
                for s in SEEDS[:3]:
                    tasks.append(_gp_task("seedleaf", g, t, "A", s, 200, ("--seed_leaves", "on"), weight=4.0))
                tasks.append(_gp_task("e2", g, t, "A", SEEDS[0], 3200, weight=64.0))
    return tasks


@phase("pilot_p1b",
       "Pilot P1 rerun for the real grammars with targets_v2 (dead-subtree filter on every level, D020): "
       "scout x 3 seeds at E0, E1 probe, seeded leaf on, 3,200-generation cost probe. About 1,500 runs.")
def pilot_p1b():
    from reach.seeds import SEEDS
    tn = "targets_v2"
    T = {g: json.loads((ROOT / f"data/targets/{g}/{tn}.json").read_text()) for g in PILOT_GRAMMARS[:4]}
    tasks = [_gp_task("scout", g, t, "A", s, 200, weight=4.0, tname=tn)
             for g in T for t in T[g] for s in SEEDS[:3]]
    for g, levels in (("EML", (11, 13, 15)), ("ELEM6", (8, 10, 12))):
        tasks += [_gp_task("e1", g, t, "A", s, 800, weight=16.0, tname=tn)
                  for t in T[g] if t["K_construction"] in levels for s in SEEDS[:3]]
    for g, k in (("EML", 11), ("ELEM6", 10)):
        for t in T[g]:
            if t["K_construction"] == k:
                tasks += [_gp_task("seedleaf", g, t, "A", s, 200, ("--seed_leaves", "on"), weight=4.0, tname=tn)
                          for s in SEEDS[:3]]
                tasks.append(_gp_task("e2", g, t, "A", SEEDS[0], 3200, weight=64.0, tname=tn))
    return tasks


BOOL5 = [f"BOOL-{b}-n5" for b in ("NAND", "NOR", "AON", "AN", "AONX", "AX1", "RICH")]


@phase("enum_bool5", "Enumerate the 7 Boolean grammars with n = 5 inputs (1 CPU-hour and 3 GB each). "
       "n = 4 saturated at E0 (pilot P1). Run with --workers 2.")
def enum_bool5():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "60", "--max_gb", "3"],
                 cpu_limit=3600, meta=dict(grammar=g)) for g in BOOL5]


@phase("scout_bool5", "Scout of the n = 5 Boolean grammars at E0 (clean engine, Regime A, 3 seeds).")
def scout_bool5():
    from reach.seeds import SEEDS
    return [_gp_task("scout", g, t, "A", s, 200)
            for g in BOOL5 if (ROOT / f"data/targets/{g}/targets.json").exists()
            for t in json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()) for s in SEEDS[:3]]


SWEEP_LEVELS = {"EML": (9, 11, 13, 15, 17), "ARITH": (9, 11, 13, 15, 17),
                "HYB": (5, 7, 9, 11, 13), "ELEM6": (4, 5, 6, 7, 8)}


@phase("budget_sweep", "Scaling pilot: real grammars at 800 and 3,200 generations (4x and 16x E0) on the "
       "5 levels around each grammar's E0 reach; targets_v2, same 3 seeds as pilot_p1b (paired).")
def budget_sweep():
    from reach.seeds import SEEDS
    tasks = []
    for g, levels in SWEEP_LEVELS.items():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_v2.json").read_text()):
            if t["K_construction"] in levels:
                for gens, w in ((800, 4.0), (3200, 16.0)):
                    tasks += [_gp_task("sweep", g, t, "A", s, gens, weight=w, tname="targets_v2") for s in SEEDS[:3]]
    return tasks


@phase("budget_sweep_bool5", "Scaling pilot for the n = 5 Boolean grammars: 800 and 3,200 generations on the "
       "levels where E0 convergence (scout_bool5) is between 0.1 and 0.9 (at most 5 per grammar); same seeds.")
def budget_sweep_bool5():
    from collections import defaultdict
    from reach.runner.queue import read_results
    from reach.seeds import SEEDS
    rate = defaultdict(list)
    for r in read_results(ROOT / "data/runs/scout_bool5/results.jsonl").values():
        rate[(r["grammar"], r["K"])].append(r["converged"] == "true")
    tasks = []
    for g in BOOL5:
        ks = sorted(k for (gg, k), v in rate.items() if gg == g and 0.1 <= sum(v) / len(v) <= 0.9)
        mid = sorted(ks, key=lambda k: abs(sum(rate[(g, k)]) / len(rate[(g, k)]) - 0.5))[:5]
        for t in json.loads((ROOT / f"data/targets/{g}/targets.json").read_text()):
            if t["K_construction"] in mid:
                for gens, w in ((800, 4.0), (3200, 16.0)):
                    tasks += [_gp_task("sweep", g, t, "A", s, gens, weight=w) for s in SEEDS[:3]]
    return tasks


# ---------------------------------------------------------------- EML-family feasibility pilot (D021)
def _family():
    from reach.family import FAMILY
    return FAMILY


@phase("enum_family", "EML-family pilot step 1: enumerate the 7 family grammars (faithful semantics) until "
       "the 8 GB memory ceiling or 1 CPU-hour each. Run with --workers 2.")
def enum_family():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g,
                              "--kmax", "40", "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=g))
            for g in _family()]


@phase("family_prepare", "EML-family pilot step 2: minimal-construction tables, scout targets (targets_fam) "
       "and calculator targets (calc) per operator.")
def family_prepare():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_prepare", "--grammar", g],
                 cpu_limit=3600, meta=dict(grammar=g)) for g in _family()]


@phase("scout_family", "EML-family pilot step 3: GP at E0 (clean engine, Regime A): scout targets x 3 seeds "
       "and calculator targets x 20 seeds, per operator.")
def scout_family():
    from reach.seeds import SEEDS
    tasks = []
    for g in _family():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_fam.json").read_text()):
            tasks += [_gp_task("scout", g, t, "A", s, 200, tname="targets_fam") for s in SEEDS[:3]]
        for t in json.loads((ROOT / f"data/targets/{g}/calc.json").read_text()):
            tasks += [_gp_task("calc", g, t, "A", s, 200, tname="calc") for s in SEEDS[:20]]
    return tasks


@phase("family_predictors", "EML-family: target-level searchability predictors from the enumeration alone "
       "(nearest smaller function in nMSE, near-duplicate counts, subtree viability); targets with K <= 19. "
       "Run with --workers 2 (memory).")
def family_predictors():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_predictors", "--grammar", g, "--kmax", "19"],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


# ---------------------------------------------------------------- EML-family pilot, round 2 (D022)
@phase("family_prepare2", "EML-family round 2: targets_fam2 (sizes 3..19, near-duplicate filter). --workers 2.")
def family_prepare2():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_prepare2", "--grammar", g],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


@phase("scout_family2", "EML-family round 2: GP at E0 with stop-on-exact: targets_fam2 x 3 seeds and calculator "
       "targets x 20 seeds, per operator.")
def scout_family2():
    from reach.seeds import SEEDS
    tasks, x = [], ("--stop_on", "exact")
    for g in _family():
        for t in json.loads((ROOT / f"data/targets/{g}/targets_fam2.json").read_text()):
            tasks += [_gp_task("scout", g, t, "A", s, 200, x, tname="targets_fam2") for s in SEEDS[:3]]
        for t in json.loads((ROOT / f"data/targets/{g}/calc.json").read_text()):
            tasks += [_gp_task("calc", g, t, "A", s, 200, x, tname="calc") for s in SEEDS[:20]]
    return tasks


@phase("family_predictors2", "EML-family round 2: enumeration-only predictors (incl. locality) for targets_fam2.")
def family_predictors2():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.family_predictors", "--grammar", g,
                              "--targets", "targets_fam2", "--suffix", "_v2", "--kmax", "19"],
                 cpu_limit=7200, meta=dict(grammar=g)) for g in _family()]


# ---------------------------------------------------------------- full EML-family study (D023; confirmatory)
FAM2 = ["F2-EML", "F2-EDL", "F2-NEML", "F2-LDE0", "F2-LDE1", "F2-PLI", "F2-PLM"]
FULL_SEEDS = slice(20, 50)          # fresh seeds, disjoint from the pilots (SEEDS[0:20])


def _conf(g):
    return json.loads((ROOT / f"data/targets/{g}/targets_conf.json").read_text())


def _calc_sets(g1):
    g2 = "F2-" + g1[2:]
    out = [(g1, "calc", t) for t in json.loads((ROOT / f"data/targets/{g1}/calc.json").read_text())]
    p2 = ROOT / f"data/targets/{g2}/calc2.json"
    out += [(g2, "calc2", t) for t in json.loads(p2.read_text())] if p2.exists() else []
    return out


@phase("enum_family2v", "Full study step 1: enumerate the 7 two-variable family grammars (leaves x, y, c), 8 GB cap.")
def enum_family2v():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", g, "--kmax", "40",
                              "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=g)) for g in FAM2]


@phase("full_prepare", "Full study step 2: confirmatory targets (targets_conf) and two-variable tables + calc2.")
def full_prepare():
    return [dict(key=g, argv=["{python}", "-m", "reach.tasks.full_prepare", "--grammar", g], cpu_limit=7200,
                 meta=dict(grammar=g)) for g in _family() + FAM2]


@phase("full_gp", "Full study H1-H3 (confirmatory): GP-A at E0, stop on exact: targets_conf x 5 seeds; calculator "
       "targets (one and two variables) x 30 seeds. Fresh seeds.")
def full_gp():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact")
    tasks = []
    for g in _family():
        tasks += [_gp_task("conf", g, t, "A", sd, 200, x, tname="targets_conf") for t in _conf(g) for sd in s[:5]]
        tasks += [_gp_task(exp, gg, t, "A", sd, 200, x, tname=exp) for gg, exp, t in _calc_sets(g) for sd in s[:30]]
    return tasks


@phase("full_es", "Full study H5: (1+4)-ES at E0, stop on exact: targets_conf sizes 3..15, 15 per size x 3 seeds; "
       "calculator targets x 10 seeds.")
def full_es():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact", "--algorithm", "es")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("es", g, t, "A", sd, 200, x, weight=4.0, tname="targets_conf")
                  for K, ts in by.items() if K <= 15 for t in ts[:15] for sd in s[:3]]
        tasks += [_gp_task("es_" + exp, gg, t, "A", sd, 200, x, weight=4.0, tname=exp)
                  for gg, exp, t in _calc_sets(g) for sd in s[:10]]
    return tasks


@phase("full_budget", "Full study H4: GP-A at 4x and 16x E0, stop on exact: targets_conf sizes 9..17, 20 per size x 3 "
       "seeds; calculator targets x 10 seeds at 16x.")
def full_budget():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    x = ("--stop_on", "exact")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        for gens, w in ((800, 4.0), (3200, 16.0)):
            tasks += [_gp_task("budget", g, t, "A", sd, gens, x, weight=w, tname="targets_conf")
                      for K, ts in by.items() if 9 <= K <= 17 for t in ts[:20] for sd in s[:3]]
        tasks += [_gp_task("budget_" + exp, gg, t, "A", sd, 3200, x, weight=16.0, tname=exp)
                  for gg, exp, t in _calc_sets(g) for sd in s[:10]]
    return tasks


@phase("full_budget64", "Full study H4b: GP-A at 64x E0 (12,800 generations), stop on exact, on a subset: EML, LDE0, "
       "PLM; sizes 11..17; 10 targets per size x 2 seeds.")
def full_budget64():
    from reach.seeds import SEEDS
    s = SEEDS[FULL_SEEDS]
    tasks = []
    for g in ("F-EML", "F-LDE0", "F-PLM"):
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("budget64", g, t, "A", sd, 12800, ("--stop_on", "exact"), weight=64.0, tname="targets_conf")
                  for K, ts in by.items() if 11 <= K <= 17 for t in ts[:10] for sd in s[:2]]
    return tasks


@phase("full_pysr", "Full study, exploratory: PySR 2.7 (Julia; .venv-pysr) on the calculator targets (one and two "
       "variables) of 6 operators (-EML excluded: PySR rejects a -inf input column); 10 seeds at E0 and 16x E0.")
def full_pysr():
    return [dict(key=g, cwd=".", argv=[".venv-pysr/bin/python", "-m", "reach.tasks.pysr_check", "--grammar", g],
                 cpu_limit=6 * 3600, weight=30.0, meta=dict(grammar=g))
            for g in ("F-EML", "F-EDL", "F-LDE0", "F-LDE1", "F-PLI", "F-PLM")]


# ---------------------------------------------------------------- held-out extension (D027)
@phase("screen_ops", "Held-out extension A1: universality screen of 30 candidate forms x 4 constants (D027).")
def screen_ops():
    from reach.grammar import CAND_CONSTS, cand_forms
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c],
                 cpu_limit=1800, meta=dict(grammar=f"S-{f}-{c}")) for f in cand_forms() for c in CAND_CONSTS]


@phase("verify_fix", "Held-out extension A3: rerun 500 sampled non-error full_gp runs (must be bit-identical after the "
       "generation-0 elitism fix) and every error run of full_gp and full_budget (exploratory EDL sensitivity).")
def verify_fix():
    import random
    tasks = []
    for ph in ("full_gp", "full_budget"):
        man = json.loads((ROOT / f"data/runs/{ph}/manifest.json").read_text())["tasks"]
        res = {json.loads(l)["key"]: json.loads(l) for l in open(ROOT / f"data/runs/{ph}/results.jsonl")}
        err = [t for t in man if res[t["key"]]["terminal_state"] == "error"]
        pick = err + (random.Random(20261015).sample([t for t in man if res[t["key"]]["terminal_state"] != "error"], 500)
                      if ph == "full_gp" else [])
        tasks += [dict(t, key=f"{ph}|{t['key']}", meta=dict(t["meta"], source=ph)) for t in pick]
    return tasks


@phase("screen_ops2", "Held-out extension A1 (amended): screen with K1 <= 21, K2 <= 17 and the x - y / x / y criterion; "
       "the five known A(x) o B(y) universal operators included as positive controls (D027 amendment).")
def screen_ops2():
    from reach.grammar import CAND_CONSTS, cand_forms
    ctrl = [("esubl", "1"), ("edivl", "e"), ("lsube", "minf"), ("ldive", "0"), ("ldive", "1")]
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c,
                                         "--k1", "21", "--k2", "17", "--max_gb", "6"],
                 cpu_limit=3 * 3600, meta=dict(grammar=f"S-{f}-{c}", control=(f, c) in ctrl))
            for f, c in ctrl + [(f, c) for f in cand_forms() for c in CAND_CONSTS]]


@phase("screen_ops3", "Held-out extension A1 fallback pools (D027): Stachowiak's cos x - arccos y with the 4 constants, "
       "and the four known A(x) o B(y) forms with the constants not yet studied; same screen as screen_ops2.")
def screen_ops3():
    from reach.grammar import CAND_CONSTS
    used = {("esubl", "1"), ("edivl", "e"), ("lsube", "minf"), ("ldive", "0"), ("ldive", "1")}
    pairs = [("cosacos", c) for c in CAND_CONSTS] + [(f, c) for f in ("esubl", "edivl", "lsube", "ldive")
                                                       for c in CAND_CONSTS if (f, c) not in used]
    return [dict(key=f"S-{f}-{c}", argv=["{python}", "-m", "reach.tasks.screen_op", "--form", f, "--const", c,
                                         "--k1", "21", "--k2", "17", "--max_gb", "6"],
                 cpu_limit=3 * 3600, meta=dict(grammar=f"S-{f}-{c}")) for f, c in pairs]


HELDOUT = ["lsube-0", "esubl-e", "lsube-e", "lpowx-m1"]      # D027, chosen by the author after the screen


@phase("enum_heldout", "Held-out extension A2: enumerate the 4 held-out grammars, one and two variables (8 GB cap, "
       "as enum_family / enum_family2v). Run with --workers 2.")
def enum_heldout():
    return [dict(key=f"{p}-{h}", argv=["{python}", "-m", "reach.tasks.enumerate_task", "--grammar", f"{p}-{h}",
                                       "--kmax", "40", "--max_gb", "8"], cpu_limit=3600, meta=dict(grammar=f"{p}-{h}"))
            for h in HELDOUT for p in ("S", "S2")]


@phase("heldout_prepare", "Held-out extension A2: minimal tables, calculator targets and random minimal targets "
       "(fresh seed 20261115) for the held-out grammars.")
def heldout_prepare():
    return [dict(key=f"{p}-{h}", argv=["{python}", "-m", "reach.tasks.heldout_prepare", "--grammar", f"{p}-{h}"],
                 cpu_limit=7200, meta=dict(grammar=f"{p}-{h}")) for h in HELDOUT for p in ("S", "S2")]


HELD_SEEDS = slice(50, 80)           # fresh seeds for the held-out extension


def _held_calc_sets(h):
    out = [(f"S-{h}", "calc", t) for t in json.loads((ROOT / f"data/targets/S-{h}/calc.json").read_text())]
    p2 = ROOT / f"data/targets/S2-{h}/calc2.json"
    return out + ([(f"S2-{h}", "calc2", t) for t in json.loads(p2.read_text())] if p2.exists() else [])


@phase("heldout_gp", "Held-out extension H6/H7 (confirmatory after the OSF addendum): GP-A at E0, stop on exact; "
       "targets_conf x 5 seeds; calculator targets (one and two variables) x 30 seeds. Mirrors full_gp.")
def heldout_gp():
    from reach.seeds import SEEDS
    s, x, tasks = SEEDS[HELD_SEEDS], ("--stop_on", "exact"), []
    for h in HELDOUT:
        tasks += [_gp_task("conf", f"S-{h}", t, "A", sd, 200, x, tname="targets_conf") for t in _conf(f"S-{h}") for sd in s[:5]]
        tasks += [_gp_task(exp, g, t, "A", sd, 200, x, tname=exp) for g, exp, t in _held_calc_sets(h) for sd in s[:30]]
    return tasks


@phase("nox_gp", "Held-out extension H8 (confirmatory after the OSF addendum): GP without crossover (regime B: same "
       "population, selection and budget; every offspring gets subtree mutation) at E0, stop on exact, on the same "
       "targets_conf and seeds as full_gp's scout runs (paired with H1).")
def nox_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact")
    return [_gp_task("confB", g, t, "B", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


# D034 (2026-10-16), post hoc sensitivity analyses, specified and committed before any run (not preregistered):
# (1) the penalty for undefined values scaled to the target (pen_*): same targets and seeds as full_gp, nox_gp and
#     full_es (random targets), every operator, so EDL is compared with the pooled six under a fitness that never
#     ranks an undefined tree above a constant; (2) headless-chicken crossover (hc_gp), the standard control for
#     the crossover effect, on full_gp's random targets and first two seeds, with EML-GP's penalty as in full_gp.
@phase("pen_gp", "D034 sensitivity: GP-A at E0, stop on exact, penalty scaled to the target; full_gp's targets_conf "
       "and seeds (5 per target).")
def pen_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--penalty", "scaled")
    return [_gp_task("confP", g, t, "A", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


@phase("pen_nox", "D034 sensitivity: GP without crossover (regime B) at E0, stop on exact, penalty scaled to the "
       "target; nox_gp's targets and seeds.")
def pen_nox():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--penalty", "scaled")
    return [_gp_task("confBP", g, t, "B", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:5]]


@phase("pen_es", "D034 sensitivity: (1+4)-ES at E0, stop on exact, penalty scaled to the target; full_es's random "
       "targets and seeds (sizes 3..15, 15 per size x 3 seeds).")
def pen_es():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--algorithm", "es", "--penalty", "scaled")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("esP", g, t, "A", sd, 200, x, weight=4.0, tname="targets_conf")
                  for K, ts in by.items() if K <= 15 for t in ts[:15] for sd in s[:3]]
    return tasks


@phase("hc_gp", "D034 control: GP-A with headless-chicken crossover (each parent crossed with a fresh random tree) at "
       "E0, stop on exact, EML-GP's penalty; full_gp's targets_conf and first two seeds.")
def hc_gp():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact", "--crossover", "headless")
    return [_gp_task("confH", g, t, "A", sd, 200, x, tname="targets_conf") for g in _family() for t in _conf(g)
            for sd in s[:2]]


# D038 (2026-10-16), post hoc, specified and committed before any run (not preregistered), answering the critical
# referee report: (1) more seeds at E0 on the budget runs' set A targets, so that independent restarts can be
# compared with single long runs at equal total budget up to 16 E0; (2) a random-sampling baseline.
@phase("restart_seeds", "D038: GP-A at E0, stop on exact, on full_budget's set A targets (sizes 9..17, 20 per size) "
       "with 15 further seeds (the 6th to 20th), for restarts against single long runs.")
def restart_seeds():
    from reach.seeds import SEEDS
    s, x = SEEDS[FULL_SEEDS], ("--stop_on", "exact")
    tasks = []
    for g in _family():
        by = {}
        for t in _conf(g):
            by.setdefault(t["K_construction"], []).append(t)
        tasks += [_gp_task("conf", g, t, "A", sd, 200, x, tname="targets_conf")
                  for K, ts in by.items() if 9 <= K <= 17 for t in ts[:20] for sd in s[5:20]]
    return tasks


@phase("rand_sample", "D038: random-sampling baseline: 4 x 250,000 trees per operator drawn as GP's initial "
       "population; exact hits of every random target counted.")
def rand_sample():
    return [dict(key=f"rand|{g}|{i}", argv=["{python}", "-m", "reach.tasks.random_sample", "--grammar", g,
                                             "--targets", f"data/targets/{g}/targets_conf.json", "--n", "250000",
                                             "--seed", str(20261016 + i)],
                 cpu_limit=3600, meta=dict(exp="rand", grammar=g, part=i)) for g in _family() for i in range(4)]
