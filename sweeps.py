"""Figure 3 (recall vs false flags as the budget varies) and Table 9 (severity
tier: hard-violation flags and false pages on the clean stream).

python3 sweeps.py --dataset default [--out results]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts import baselines as B
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats
from fcontracts.paging import run_paging

ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prevalence", type=float, nargs="*", default=[0.01, 0.10, 1.0])
    a = ap.parse_args()
    ds = a.dataset
    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    X_fit, X_cal, X_test = X[sf], X[sc], X[st]
    d = X.shape[1]
    ref = reference_stats(np.vstack([X_fit, X_cal]))
    W = min(CONFIG["window"][ds], len(X_test))

    # windows reused across budgets
    cells = []
    for seed in a.seeds:
        srng = np.random.default_rng(20_000 + seed)
        start = int(srng.integers(0, len(X_test) - W + 1))
        win = X_test[start:start + W]
        for ci, (fault, prev, br) in enumerate(itertools.product(
                CONFIG["paper_faults"], a.prevalence, CONFIG["breadth"])):
            rng = np.random.default_rng([seed, ci, 7])
            feats = rng.choice(d, size=min(br, d), replace=False)
            Xw, ch = inject(win, fault, feats, prev, rng, ref)
            rows = np.flatnonzero(ch.any(1))
            if len(rows) >= CONFIG["min_changed"]:
                cells.append((fault, prev, br, seed, Xw, rows))

    out = []
    fixed = {"minmax": B.MinMax().fit(X_fit, X_cal), "autoprofile": B.AutoProfile().fit(X_fit, X_cal)}
    for name, det in fixed.items():
        ff = det.check(X_test).flag.mean() * 1e5
        for fault, prev, br, seed, Xw, rows in cells:
            out.append(dict(method=name, alpha=np.nan, ff=ff, fault=fault, prevalence=prev, breadth=br,
                            seed=seed, recall=det.check(Xw).flag[rows].mean()))
    for alpha in ALPHAS:
        dets = {"contracts_domain": FeatureContract(groups, alpha, relations=False).fit(X_fit, X_cal),
                "contracts": FeatureContract(groups, alpha).fit(X_fit, X_cal),
                "robust_z": B.RobustZ().fit(X_fit, X_cal, alpha),
                "conformance": B.Conformance().fit(X_fit, X_cal, alpha),
                "pca_recon": B.PCARecon().fit(X_fit, X_cal, alpha)}
        for name, det in dets.items():
            ff = det.check(X_test).flag.mean() * 1e5
            for fault, prev, br, seed, Xw, rows in cells:
                out.append(dict(method=name, alpha=alpha, ff=ff, fault=fault, prevalence=prev, breadth=br,
                                seed=seed, recall=det.check(Xw).flag[rows].mean()))
        print(f"[{ds}] alpha={alpha} done", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(a.out, f"{ds}_budget_sweep.csv"), index=False)

    # Table 9: severity tier on the clean test stream
    alpha = CONFIG["alpha"][ds]
    tier = {}
    for h in [None, 0.5, 1.0, 2.0, 5.0]:
        c = FeatureContract(groups, alpha, tier="all" if h is None else "hard",
                            h=1.0 if h is None else h).fit(X_fit, X_cal)
        r = c.check(X_test)
        if h is None:
            fl = r.flag.mean() * 1e5
            pages = run_paging(r.viol_rows, r.viol_cols, d, c.p0_all)
        else:
            fl = len(np.unique(r.hard_rows)) / len(X_test) * 1e5
            pages = run_paging(r.hard_rows, r.hard_cols, d, c.p0_hard)
        tier["all" if h is None else f"h={h}"] = {"flags_per_100k": fl,
                                                   "pages_per_100k": len(pages) / len(X_test) * 1e5}
    with open(os.path.join(a.out, f"{ds}_tier.json"), "w") as fh:
        json.dump(tier, fh, indent=2)
    print(json.dumps(tier, indent=1))


if __name__ == "__main__":
    main()
