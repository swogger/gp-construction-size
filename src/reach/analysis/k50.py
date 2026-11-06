"""Logistic reach K50 with a target (cluster) bootstrap: port of legacy analyze_p1.logit / k50."""
from collections import defaultdict

import numpy as np


def logit(x, y, iters=300):
    """IRLS fit of P(y) = 1 / (1 + exp(-(b0 + b1 x))). Returns (b0, b1)."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-9, None)
        nb = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ (X @ b + (y - p) / W))
        if np.max(np.abs(nb - b)) < 1e-12:
            return nb
        b = nb
    return b


def k50(size, y, cluster, boot=1000, seed=20260926):
    """Midpoint -b0/b1 on `size` (any unit), 95% CI from resampling clusters (targets).
    Returns (estimate, lo, hi, slope); NaN when the fit is degenerate."""
    size, y = np.asarray(size, float), np.asarray(y, float)
    nan4 = (float("nan"),) * 4
    if y.min() == y.max() or size.min() == size.max():
        return nan4
    try:
        b = logit(size, y)
    except np.linalg.LinAlgError:
        return nan4
    est = -b[0] / b[1]
    by = defaultdict(list)
    for i, c in enumerate(cluster):
        by[c].append(i)
    ids, rng, bs = list(by), np.random.default_rng(seed), []
    for _ in range(boot):
        idx = np.concatenate([by[ids[j]] for j in rng.integers(0, len(ids), len(ids))])
        if y[idx].min() == y[idx].max() or size[idx].min() == size[idx].max():
            continue
        try:
            bb = logit(size[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        if bb[1] < 0:
            bs.append(-bb[0] / bb[1])
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (float("nan"), float("nan"))
    return float(est), float(lo), float(hi), float(b[1])
