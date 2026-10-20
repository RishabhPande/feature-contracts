"""Added latency and memory per request (Table 6). Contracts with random
parameters, relation groups of 20 features, one core.

python3 latency.py [--out results]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from fcontracts.contracts import FeatureContract


def fit_random(d, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    groups = [list(range(i, min(i + 20, d))) for i in range(0, d, 20)]
    c = FeatureContract(groups, 1e-2, calibration="grid")
    c.fit(X[: n // 2], X[n // 2:])
    return c, rng


class Compiled:
    """All relation groups stacked: one gather, one elementwise transform and
    one sparse block-diagonal product per request."""

    def __init__(self, c):
        from scipy.sparse import block_diag
        self.c = c
        if c.rel:
            self.idx = np.concatenate([r["g"] for r in c.rel])
            cat = lambda k: np.concatenate([r["tf"][k] for r in c.rel])
            self.lo, self.hi, self.med, self.iqr, self.mu, self.sd = (cat(k) for k in range(6))
            self.P = block_diag([r["P"] / r["diag"][None, :] for r in c.rel], format="csr").T.tocsr()
            self.scale = np.concatenate([r["sd"] for r in c.rel])
        else:
            self.idx = None

    def __call__(self, x, R, lr1, lr0, thr):
        c = self.c
        viol = np.fmax(c.lo - x, x - c.hi) > c.kappa * c.s
        if self.idx is not None:
            z = (np.arcsinh((np.clip(x[self.idx], self.lo, self.hi) - self.med) / self.iqr) - self.mu) / self.sd
            res = (self.P @ z) / self.scale
            viol[self.idx] |= np.abs(res) > c.t
        if R is not None:
            R += 1.0
            R *= np.where(viol, lr1, lr0)
            R[R >= thr] = 0.0
        return viol


def single_request(c, x, R, lr1, lr0, thr):
    """One request: domain + relations, then a Shiryaev-Roberts update per feature."""
    ex = np.fmax(c.lo - x, x - c.hi) / c.s
    dom = ex > c.kappa
    viol = dom.copy()
    for r in c.rel:
        g = r["g"]
        env_lo, env_hi, med, iqr, mu, sd = r["tf"]
        z = (np.arcsinh((np.clip(x[g], env_lo, env_hi) - med) / iqr) - mu) / sd
        res = (z @ r["P"]) / r["diag"] / r["sd"]
        viol[g] |= np.abs(res) > c.t
    if R is not None:
        R[:] = (1.0 + R) * np.where(viol, lr1, lr0)
        R[R >= thr] = 0.0
    return viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=3000)
    a = ap.parse_args()
    out = {}
    for d in [100, 1000, 10000]:
        c, rng = fit_random(d)
        Xq = rng.normal(size=(a.reps, d))
        p0 = np.full(d, 1e-4); p1 = np.maximum(0.005, 10 * p0)
        lr1, lr0, thr = p1 / p0, (1 - p1) / (1 - p0), d * 1e7
        res = {}
        comp = Compiled(c)
        for i in range(5):     # check the fast path gives the same violations
            assert np.array_equal(comp(Xq[i], None, lr1, lr0, thr), single_request(c, Xq[i], None, lr1, lr0, thr))
        for label, paging in (("check", False), ("check_plus_paging", True)):
            R = np.zeros(d) if paging else None
            ts = []
            for i in range(a.reps):
                t = time.perf_counter()
                comp(Xq[i], R, lr1, lr0, thr)
                ts.append(time.perf_counter() - t)
            ts = np.array(ts[100:]) * 1e6
            res[label] = {"p50_us": float(np.percentile(ts, 50)), "p99_us": float(np.percentile(ts, 99))}
        t = time.perf_counter()
        c.check(Xq)
        res["vectorized_us_per_request"] = (time.perf_counter() - t) / a.reps * 1e6
        res["contract_kb"] = len(pickle.dumps({k: v for k, v in c.__dict__.items() if k != "rates"})) / 1024
        out[d] = res
        print(d, json.dumps(res), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "latency.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
