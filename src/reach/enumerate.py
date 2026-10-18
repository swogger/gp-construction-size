"""Bottom-up enumeration of distinct functions by size (M4; decisions D009, D012).

Level K holds every function whose smallest tree has exactly K nodes. A level is built from
smaller levels only (unary op on level K-1; binary op on levels i, j with i + j = K - 1), and
a candidate is kept only if its semantic key has not been seen at any smaller size, so the
first size at which a function appears is its minimal size (on the sample, for real grammars).

Real grammars: values on EML-GP's 100-point grid on (2, 3), rounded to 9 decimals (absolute,
as EML-GP's func_hash), -0.0 and NaN canonicalised, then hashed to 64 bits. Boolean grammars:
the key is the packed truth table itself (exact).

Per level the output records N_G(K) (syntactic), C(K) (candidates evaluated: the cost of
realisable bottom-up enumeration, D009), U_G(K) (new distinct functions) and cumulative counts.
Each level is checkpointed (keys plus parent pointers; values are rebuilt on resume), so an
interrupted or CPU-limited run resumes at the next level.
"""
import json
import time
from pathlib import Path

import numpy as np

from reach import grammar as G

CHUNK = 200_000                 # candidates evaluated at once (real: 100 complex values each)
_MIX = np.uint64(0x9E3779B97F4A7C15)


def _coeffs(n):
    return np.random.default_rng(20261012).integers(1, 2 ** 63, size=n, dtype=np.uint64) | np.uint64(1)


def real_keys(v, coeffs):
    """64-bit semantic key per row of a complex (n, P) array."""
    with np.errstate(all="ignore"):
        x = np.concatenate([np.round(v.real, G.HASH_PRECISION), np.round(v.imag, G.HASH_PRECISION)], axis=1) + 0.0
    x[np.isnan(x)] = np.nan
    u = x.view(np.uint64)
    u = (u ^ (u >> np.uint64(31))) * _MIX
    u ^= u >> np.uint64(29)
    return (u * coeffs).sum(axis=1, dtype=np.uint64)


class Enumerator:
    def __init__(self, g, out_dir, log=print):
        self.g, self.out, self.log = g, Path(out_dir), log
        self.out.mkdir(parents=True, exist_ok=True)
        self.real = g.kind == "real"
        self.xs = G.sample_grid(g) if self.real else None
        self.coeffs = _coeffs(2 * G.N_POINTS) if self.real else None
        self.mask = np.uint64(g.mask) if not self.real else None
        self.unary = [o for o in g.ops if o.arity == 1]
        self.binary = [o for o in g.ops if o.arity == 2]
        self.opi = {o.name: i for i, o in enumerate(g.ops)}
        self.vals, self.keys, self.parents = {}, {}, {}
        self.known = np.empty(0, dtype=np.uint64)          # sorted keys of all levels so far
        self.stats = []
        self.syn = G.syntactic_counts(g, 200)

    # ------------------------------------------------------------ values
    def _apply(self, op, a, b=None):
        with np.errstate(all="ignore"):
            if self.real:
                return op.fn(a) if b is None else op.fn(a, b)
            return op.fn(self.mask, a) if b is None else op.fn(self.mask, a, b)

    def _key(self, v):
        return real_keys(v, self.coeffs) if self.real else v.astype(np.uint64)

    def _leaves(self):
        if self.real:
            xs = self.xs if isinstance(self.xs, dict) else {self.g.variables[0]: self.xs}
            shape = next(iter(xs.values())).shape
            leaves = [xs[v] for v in self.g.variables] + [np.full(shape, complex(c)) for c in self.g.constants]
            return np.stack(leaves), [("leaf", v) for v in list(self.g.variables) + list(self.g.constants)]
        vm = G.var_masks(self.g.n_inputs)
        vals = [vm[v] for v in self.g.variables] + [self.g.mask for _ in self.g.constants]
        return np.array(vals, dtype=np.uint64), [("leaf", v) for v in list(self.g.variables) + list(self.g.constants)]

    # ------------------------------------------------------------ one level
    def build_level(self, k, progress_every=30.0, store_vals=True):
        """store_vals=False keeps keys and parents only: for the last level, which nothing builds on."""
        t0 = time.process_time()
        new_vals, new_keys, new_par, new_ok = [], [], [], []
        n_cand = 0
        jobs = []                                            # (op, i, j): j = None for unary
        for op in self.unary:
            if k - 1 in self.vals:
                jobs.append((op, k - 1, None))
        for i in range(1, k - 1):
            j = k - 1 - i
            if i in self.vals and j in self.vals:
                for op in self.binary:
                    jobs.append((op, i, j))
        total = sum(len(self.vals[i]) * (1 if j is None else len(self.vals[j])) for _, i, j in jobs)
        last = time.time()
        for op, i, j in jobs:
            A = self.vals[i]
            if j is None:
                blocks = [(s, None) for s in range(0, len(A), CHUNK)]
            else:
                B = self.vals[j]
                cols = max(1, min(len(B), CHUNK))              # chunk both operands
                rows = max(1, CHUNK // max(cols, 1))
                blocks = [(s, c) for s in range(0, len(A), rows) for c in range(0, len(B), cols)]
            for s, c in blocks:
                if j is None:
                    a_idx = np.arange(s, min(s + CHUNK, len(A)))
                    v = self._apply(op, A[a_idx])
                    b_idx = np.full(len(a_idx), -1)
                else:
                    a_rng = np.arange(s, min(s + rows, len(A)))
                    b_rng = np.arange(c, min(c + cols, len(B)))
                    if self.real:
                        v = self._apply(op, A[a_rng][:, None, :], B[b_rng][None, :, :]).reshape(-1, A.shape[1])
                    else:
                        v = self._apply(op, A[a_rng][:, None], B[b_rng][None, :]).reshape(-1)
                    a_idx = np.repeat(a_rng, len(b_rng))
                    b_idx = np.tile(b_rng, len(a_rng))
                n_cand += len(a_idx)
                keys = self._key(v)
                keys_u, first = np.unique(keys, return_index=True)
                pos = np.searchsorted(self.known, keys_u)
                pos[pos >= len(self.known)] = 0
                fresh = (self.known[pos] != keys_u) if len(self.known) else np.ones(len(keys_u), bool)
                sel = first[fresh]
                if len(sel):
                    if store_vals:
                        new_vals.append(v[sel])
                    new_ok.append(self._viable(v[sel]))
                    new_keys.append(keys[sel])
                    new_par.append(np.stack([np.full(len(sel), self.opi[op.name]), np.full(len(sel), i),
                                             a_idx[sel], np.full(len(sel), -1 if j is None else j),
                                             b_idx[sel]], axis=1))
                if time.time() - last > progress_every:
                    el = time.process_time() - t0
                    frac = n_cand / max(total, 1)
                    self.log(f"  K={k}: {100 * frac:5.1f}% of {total:,} candidates, "
                             f"cpu {el:,.0f}s, level ETA {el / frac - el if frac else 0:,.0f}s")
                    last = time.time()
        if new_keys:
            keys = np.concatenate(new_keys)
            keys_u, first = np.unique(keys, return_index=True)      # dedup across chunks
            first.sort()
            vals = np.concatenate(new_vals)[first] if store_vals else None
            ok = np.concatenate(new_ok)[first]
            par = np.concatenate(new_par)[first].astype(np.int64)
            keys = keys[first]
        else:
            vals = np.empty((0, G.N_POINTS), np.complex128) if self.real else np.empty(0, np.uint64)
            keys, par, ok = np.empty(0, np.uint64), np.empty((0, 5), np.int64), np.empty(0, bool)
        self._add_level(k, vals, keys, par)
        self.stats.append(self._level_stats(k, n_cand, keys, ok, time.process_time() - t0, store_vals))
        self._save_level(k)

    def _n_candidates(self, k):
        n = len(self.unary) * len(self.vals.get(k - 1, ()))
        return n + len(self.binary) * sum(len(self.vals.get(i, ())) * len(self.vals.get(k - 1 - i, ()))
                                          for i in range(1, k - 1))

    def _add_level(self, k, vals, keys, par):
        if vals is not None:
            self.vals[k] = vals
        self.keys[k], self.parents[k] = keys, par
        self.known = np.union1d(self.known, keys)

    def _viable(self, v):
        """Real: finite on the whole grid and non-constant. Bool: every function."""
        if not self.real:
            return np.ones(len(v), bool)
        with np.errstate(all="ignore"):
            return np.isfinite(v).all(axis=1) & (np.abs(v - v.mean(axis=1, keepdims=True)).max(axis=1) > 1e-6)

    def _level_stats(self, k, n_cand, keys, ok, cpu, stored=True):
        cum = sum(len(self.keys[i]) for i in self.keys if i <= k)
        return dict(K=k, N_syntactic=self.syn[k], candidates=int(n_cand), U=int(len(keys)), U_cum=int(cum),
                    U_finite_nonconst=int(ok.sum()), cpu_s=round(cpu, 2), vals_stored=stored)

    # ------------------------------------------------------------ persistence
    def _save_level(self, k):
        np.savez(self.out / f"level_{k:03d}.npz", keys=self.keys[k], parents=self.parents[k])
        (self.out / "counts.json").write_text(json.dumps(dict(
            grammar=self.g.name, levels=self.stats, hash_precision=G.HASH_PRECISION,
            sample="grid (2,3) 100 points" if self.real else "full truth table"), indent=1))

    def load(self):
        """Rebuild saved levels from parent pointers; returns the last complete K."""
        cj = self.out / "counts.json"
        if not cj.exists():
            return 0
        self.stats = json.loads(cj.read_text())["levels"]
        last = 0
        for st in self.stats:
            k = st["K"]
            z = np.load(self.out / f"level_{k:03d}.npz")
            keys, par = z["keys"], z["parents"]
            if not st.get("vals_stored", True):
                self._add_level(k, None, keys, par)
                return -k
            if k == 1:
                vals, _ = self._leaves()
            else:
                vals = np.empty((len(par), G.N_POINTS), np.complex128) if self.real else np.empty(len(par), np.uint64)
                for oi, i, j in np.unique(par[:, [0, 1, 3]], axis=0):
                    r = np.where((par[:, 0] == oi) & (par[:, 1] == i) & (par[:, 3] == j))[0]
                    op, a = self.g.ops[oi], self.vals[i][par[r, 2]]
                    vals[r] = self._apply(op, a) if j < 0 else self._apply(op, a, self.vals[j][par[r, 4]])
            self._add_level(k, vals, keys, par)
            last = k
        return last

    def run(self, kmax, max_gb=12.0):
        start = self.load()
        if start < 0:
            self.log(f"{self.g.name}: already finished at K={-start} (memory ceiling)")
            return -start, "budget_exhausted"
        if start:
            self.log(f"{self.g.name}: resumed after K={start}")
        if start == 0:
            vals, _ = self._leaves()
            keys = self._key(vals)
            self._add_level(1, vals, keys, np.full((len(keys), 5), -1, np.int64))
            self.stats.append(self._level_stats(1, len(keys), keys, self._viable(vals), 0.0))
            self._save_level(1)
            start = 1
        for k in range(start + 1, kmax + 1):
            size_gb = sum(v.nbytes for v in self.vals.values()) / 1e9
            if self.real:                         # upper bound: every candidate is new
                size_gb += self._n_candidates(k) * 16 * G.N_POINTS / 1e9
            if size_gb > max_gb:
                self.log(f"{self.g.name}: memory ceiling (predicted {size_gb:.1f} GB > {max_gb} GB): "
                         f"K={k} is the last level, keys only")
                self.build_level(k, store_vals=False)
                self._log_level()
                return k, "budget_exhausted"
            if not self.real and len(self.known) == (1 << (1 << self.g.n_inputs)):
                self.log(f"{self.g.name}: all Boolean functions found by K={k - 1}")
                return k - 1, "completed"
            self.build_level(k)
            self._log_level()
        return kmax, "completed"

    def _log_level(self):
        st = self.stats[-1]
        self.log(f"{self.g.name} K={st['K']}: N={st['N_syntactic']:,} candidates={st['candidates']:,} "
                 f"U={st['U']:,} U_cum={st['U_cum']:,} cpu={st['cpu_s']:,.1f}s")

    # ------------------------------------------------------------ lookup
    def min_size(self, vec):
        """Smallest K at which this function (value vector or truth table) appears, or None."""
        key = self._key(vec[None, :] if self.real else np.array([vec], dtype=np.uint64))[0]
        for k in sorted(self.keys):
            if np.any(self.keys[k] == key):
                return k
        return None

    def expr(self, k, idx):
        """Representative expression of function idx at level k (from parent pointers)."""
        op_i, i, a, j, b = self.parents[k][idx]
        if k == 1:
            return G.expr_str(G.leaf(self._leaves()[1][idx][1]))
        name = self.g.ops[op_i].name
        if j < 0:
            return f"{name}({self.expr(i, a)})"
        return f"{name}({self.expr(i, a)}, {self.expr(j, b)})"
