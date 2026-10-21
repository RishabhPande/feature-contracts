"""Exact single-score conformal calibration versus the paper's two-step grid rule
(post-confirmation analysis; the frozen configuration is unchanged).

For each dataset and calibration rule it reports the realized false-flag rate on
(a) the time-ordered test stream and (b) an exchangeable control in which all rows
are randomly permuted before the same split, where the finite-sample bound applies,
plus mean recall and top-1 localization over the core fault cells.

python3 conformal_eval.py --dataset baf --out results_v3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats


def cells_for(X_test, X_ref, d, W, seeds, prevalence):
    ref = reference_stats(X_ref)
    out = []
    for seed in seeds:
        srng = np.random.default_rng(20_000 + seed)
        start = int(srng.integers(0, len(X_test) - W + 1))
        win = X_test[start:start + W]
        for ci, (fault, prev, br) in enumerate(itertools.product(CONFIG["paper_faults"], prevalence, CONFIG["breadth"])):
            rng = np.random.default_rng([seed, ci, 7])
            feats = rng.choice(d, size=min(br, d), replace=False)
            Xw, ch = inject(win, fault, feats, prev, rng, ref)
            rows = np.flatnonzero(ch.any(1))
            if len(rows) >= CONFIG["min_changed"]:
                out.append((fault, prev, br, seed, Xw, rows, ch))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    ds = a.dataset
    X, y, groups, cols = load(ds)
    alpha = CONFIG["alpha"][ds]
    sf, sc, st = splits(len(X))
    d = X.shape[1]
    W = min(CONFIG["window"][ds], len(X) - sc.stop)
    res = {"alpha": alpha}
    perm = np.random.default_rng(123).permutation(len(X))
    for order, XX in (("time", X), ("exchangeable", X[perm])):
        X_fit, X_cal, X_test = XX[sf], XX[sc], XX[st]
        cells = cells_for(X_test, np.vstack([X_fit, X_cal]), d, W, a.seeds, [0.1, 1.0]) if order == "time" else []
        for calib in ("grid", "conformal1"):
            c = FeatureContract(groups, alpha, calibration=calib).fit(X_fit, X_cal)
            ff = float(c.check(X_test).flag.mean())
            n = len(X_test)
            r = {"ff_per_m": ff * 1e6, "flags": int(round(ff * n)), "n_test": n,
                 "kappa": float(c.kappa), "tau": float(c.t) if np.isfinite(c.t) else None}
            if cells:
                rec, loc = [], []
                for fault, prev, br, seed, Xw, rows, ch in cells:
                    o = c.check(Xw)
                    rec.append(o.flag[rows].mean())
                    top = o.top[rows, 0]
                    loc.append(np.mean([(t >= 0 and ch[rr, t]) for rr, t in zip(rows, top)]))
                r["recall"] = float(np.mean(rec)); r["localized"] = float(np.mean(loc))
                r["recall_runs"] = [float(v) for v in rec]
            res[f"{order}|{calib}"] = r
            print(ds, order, calib, {k: v for k, v in r.items() if k != "recall_runs"}, flush=True)
    with open(os.path.join(a.out, f"{ds}_conformal.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
