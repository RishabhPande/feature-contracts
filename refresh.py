"""Exploratory (designed after the IEEE-CIS confirmation failed): periodic
contract refresh on the most recent clean traffic.

For each refresh period, contracts are refit every `period` test requests on
the latest `ref_len` requests (80% fit, 20% calibration), as a team would do
with an approved schedule. Reports false flags and false pages on the clean
test stream, false flags by segment, and recall on injected faults.

python3 refresh.py --dataset ieee [--out results]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats
from fcontracts.paging import run_paging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee")
    ap.add_argument("--out", default="results")
    ap.add_argument("--periods", type=int, nargs="*", default=[0, 50000, 20000])
    ap.add_argument("--ref-len", type=int, default=100000)
    a = ap.parse_args()
    ds = a.dataset
    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    n_ref = st.start
    d = X.shape[1]
    alpha = CONFIG["alpha"][ds]
    out = {}
    for period in a.periods:
        flags = np.zeros(len(X) - n_ref, bool)
        hr, hc, vr, vc = [], [], [], []
        seg_rates, contracts = [], []
        starts = [n_ref] if period == 0 else list(range(n_ref, len(X), period))
        for i, s in enumerate(starts):
            e = len(X) if i == len(starts) - 1 else starts[i + 1]
            if period == 0:
                Xf, Xc = X[sf], X[sc]
            else:
                lo = max(0, s - a.ref_len)
                cut = lo + int(0.8 * (s - lo))
                Xf, Xc = X[lo:cut], X[cut:s]
            c = FeatureContract(groups, alpha).fit(Xf, Xc)
            r = c.check(X[s:e])
            flags[s - n_ref:e - n_ref] = r.flag
            hr.append(r.hard_rows + s - n_ref); hc.append(r.hard_cols)
            vr.append(r.viol_rows + s - n_ref); vc.append(r.viol_cols)
            contracts.append((s, e, c))
        p0h = np.max([c.p0_hard for _, _, c in contracts], 0)
        pages = run_paging(np.concatenate(hr), np.concatenate(hc), d, p0h)
        n = len(flags)
        seg = np.array_split(np.arange(n), 6)
        res = {"false_flags_per_100k": float(flags.mean() * 1e5),
               "false_pages_hard_per_100k": len(pages) / n * 1e5,
               "false_flags_by_sixth": [float(flags[s_].mean() * 1e5) for s_ in seg]}
        # recall on injected faults with the contract active at the window
        ref = reference_stats(np.vstack([X[sf], X[sc]]))
        rec = {}
        for fault in CONFIG["paper_faults"]:
            vals = []
            for seed in range(3):
                rng = np.random.default_rng([seed, 99, hash(fault) % 997])
                s0 = int(rng.integers(n_ref, len(X) - 10000))
                win = X[s0:s0 + 10000]
                c = [c for s, e, c in contracts if s <= s0 < e][0]
                feats = rng.choice(d, size=1, replace=False)
                Xw, ch = inject(win, fault, feats, 0.10, rng, ref)
                rows = np.flatnonzero(ch.any(1))
                if len(rows) >= 5:
                    vals.append(c.check(Xw).flag[rows].mean())
            rec[fault] = float(np.mean(vals)) if vals else None
        res["recall_prev10_breadth1"] = rec
        res["mean_recall"] = float(np.mean([v for v in rec.values() if v is not None]))
        out[f"period_{period}"] = res
        print(period, json.dumps(res), flush=True)
    with open(os.path.join(a.out, f"{ds}_refresh.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
