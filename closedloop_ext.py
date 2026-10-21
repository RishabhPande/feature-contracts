"""Closed-loop extensions (post-confirmation analysis).

A fault starts at ONSET and CLEARS after DURATION requests, so the policy must both
switch in and switch back. Responses, all driven by the same contract pages (severity
tier plus rate alarms), are compared on the whole test stream:

  unprotected        primary model, no response
  mask               primary model with violated features masked (clean threshold)
  fallback           first page switches to a model trained without the paged features;
                     stays switched to the end of the stream (the paper's policy)
  fallback_recover   as fallback, but switches back once no paged feature has violated for
                     RECOVER requests; later pages switch in again
  impute_recover     rule-based: paged features are replaced by their reference median,
                     primary model and threshold kept (same switching as fallback_recover)
  review_recover     conservative rule: while switched, requests violating a paged feature
                     are sent to review (counted as alerts); the rest use the primary model
  oracle             fallback without the truly corrupted features, exactly while the fault lasts

The fault 'shared' corrupts the most important feature (magnitude) and the second
most important (lagged stale values that stay in range), mimicking a fallback that shares
an upstream source with the primary model.

Thresholds: 'fpr1' sets each model's threshold at a 1% false positive rate on the
calibration slice (as in the paper); 'costopt' minimizes expected cost (false alarm 1,
missed fraud 50) on the calibration slice.

Per run: recall, FPR, precision, cost per 1,000 requests, number of switches, false
switches (switch-ins while no fault is active) per million clean requests, share of the
stream spent switched, delay to switch in and to switch back after the fault clears.

python3 closedloop_ext.py --dataset cc --out results_v3
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats
from fcontracts.paging import run_paging

RECOVER = 1000


def fit_model(X, y):
    return HistGradientBoostingClassifier(random_state=0).fit(np.clip(X, -1e30, 1e30), y)


def score(m, X):
    return m.predict_proba(np.clip(X, -1e30, 1e30))[:, 1]


def threshold(m, X, y, mode):
    s = score(m, X)
    if mode == "fpr1":
        return float(np.quantile(s[y == 0], 0.99))
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.9999, 400)))
    cost = [((s > c) & (y == 0)).sum() + 50 * ((s <= c) & (y == 1)).sum() for c in cand]
    return float(cand[int(np.argmin(cost))])


def metrics(pred, y):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); n = len(y)
    return dict(recall=tp / max(1, tp + fn), fpr=fp / max(1, (y == 0).sum()), precision=tp / max(1, tp + fp),
                cost50=1000 * (fp + 50 * fn) / n)


def segments(pages, vtimes, n, recover):
    """Switching policy with recovery. pages: sorted (t, j). vtimes: dict j -> sorted violation times.
    Returns list of (start, end, frozenset(paged))."""
    segs = []
    i = 0
    pages = [p for p in pages if p[0] < n]
    while i < len(pages):
        t0 = pages[i][0]
        P = {j for t, j in pages if t == t0}
        start = t0
        cur = t0
        while True:
            V = np.unique(np.concatenate([vtimes.get(j, np.zeros(0, int)) for j in P])) if P else np.zeros(0, int)
            V = V[V >= cur]
            prev = cur; end = None
            for v in V:
                if v - prev > recover:
                    end = prev + recover; break
                prev = v
            if end is None:
                end = prev + recover
            # a page on a new feature before recovery extends the set
            new = [(t, j) for t, j in pages if cur < t < end and j not in P]
            if new:
                tn = new[0][0]
                segs.append((start, tn, frozenset(P)))
                P |= {j for t, j in new if t == tn}
                start = cur = tn
                continue
            segs.append((start, min(end, n), frozenset(P)))
            break
        while i < len(pages) and pages[i][0] < end:
            i += 1
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--faults", nargs="*", default=["none", "magnitude", "scale", "encoding", "nan_fill", "stale_lag", "shared"])
    ap.add_argument("--choices", nargs="*", default=["top1", "top5"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--onset", type=int, default=5000)
    ap.add_argument("--duration", type=int, default=20000)
    ap.add_argument("--imp-sample", type=int, default=30000)
    a = ap.parse_args()
    ds = a.dataset
    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    Xf, yf, Xc, yc, Xt, yt = X[sf], y[sf], X[sc], y[sc], X[st], y[st]
    n, d = Xt.shape
    onset, dur = a.onset, min(a.duration, n - a.onset - 2 * RECOVER)
    con = FeatureContract(groups, CONFIG["alpha"][ds]).fit(Xf, Xc)
    ref = reference_stats(np.vstack([Xf, Xc]))
    med = np.nanmedian(np.vstack([Xf, Xc]), 0)
    prim = fit_model(Xf, yf)
    sub = np.random.default_rng(0).choice(len(Xc), size=min(a.imp_sample, len(Xc)), replace=False)
    imp = permutation_importance(prim, np.clip(Xc[sub], -1e30, 1e30), yc[sub], scoring="roc_auc",
                                 n_repeats=2, random_state=0).importances_mean
    order = np.argsort(-imp)
    cache = {}

    def fallback(P, mode):
        k = (tuple(sorted(P)), mode)
        if k not in cache:
            keep = np.array([j for j in range(d) if j not in P])
            m = cache.get(("model", k[0])) or fit_model(Xf[:, keep], yf)
            cache[("model", k[0])] = m
            cache[k] = (keep, m, threshold(m, Xc[:, keep], yc, mode))
        return cache[k]

    thr = {m: threshold(prim, Xc, yc, m) for m in ("fpr1", "costopt")}
    rows = []
    combos = [c for c in itertools.product(a.faults, a.choices, a.seeds) if not (c[0] == "none" and (c[1] != a.choices[0] or c[2] != a.seeds[0]))]
    fault_mask = np.zeros(n, bool); fault_mask[onset:onset + dur] = True
    for fault, choice, seed in combos:
        rng = np.random.default_rng([seed, hash(fault) % 997, hash(choice) % 991, 11])
        feats = order[:1] if choice == "top1" else order[:5]
        Xcor = Xt.copy()
        changed_feats = set()
        if fault == "shared":
            Xw, _ = inject(Xt[onset:onset + dur], "magnitude", order[:1], 1.0, rng, ref, pool=Xc)
            Xw, _ = inject(Xw, "stale_lag", order[1:2], 1.0, rng, ref, pool=Xc)
            Xcor[onset:onset + dur] = Xw
            changed_feats = {int(order[0]), int(order[1])}
            if choice != "top1":
                continue
        elif fault != "none":
            Xw, ch = inject(Xt[onset:onset + dur], fault, feats, 1.0, rng, ref, pool=Xc)
            Xcor[onset:onset + dur] = Xw
            changed_feats = {int(j) for j in np.flatnonzero(ch.any(0))}
        res = con.check(Xcor)
        alarms = con.rates.alarms(Xcor)
        pages = sorted(run_paging(res.hard_rows, res.hard_cols, d, con.p0_hard) + [(t, j) for t, j, _ in alarms])
        order_v = np.lexsort((res.viol_rows, res.viol_cols))
        vr, vc = res.viol_rows[order_v], res.viol_cols[order_v]
        b = np.searchsorted(vc, np.arange(d + 1))
        vtimes = {j: vr[b[j]:b[j + 1]] for j in range(d) if b[j + 1] > b[j]}
        # add rate-alarm windows as violation evidence for recovery (alarm row = window end)
        for t, j, _ in alarms:
            w = CONFIG["rate"]["window"]
            vtimes[j] = np.unique(np.r_[vtimes.get(j, np.zeros(0, int)), np.arange(max(0, t - w + 1), t + 1)])
        segs_rec = segments(pages, vtimes, n, RECOVER)
        segs_stay = []
        if pages:
            t0 = pages[0][0]
            segs_stay = [(t0, n, frozenset(j for t, j in pages if t == t0))]
        Xm = Xcor.copy()
        Xm[res.viol_rows, res.viol_cols] = np.nan
        s_unp = score(prim, Xcor); s_mask = score(prim, Xm); del Xm
        base = dict(dataset=ds, fault=fault, choice=choice, seed=seed, n_fault=int(dur if fault != "none" else 0))
        corr = changed_feats
        for mode in ("fpr1", "costopt"):
            arms = {"unprotected": s_unp > thr[mode], "mask": s_mask > thr[mode]}
            if corr:
                keep, om, othr = fallback(corr, mode)
                p = s_unp > thr[mode]
                p[fault_mask] = score(om, Xcor[fault_mask][:, keep]) > othr
                arms["oracle"] = p
            for pol, segs in (("fallback", segs_stay), ("fallback_recover", segs_rec)):
                p = s_unp > thr[mode]
                for s0, e0, P in segs:
                    keep, fm, fthr = fallback(P, mode)
                    p[s0:e0] = score(fm, Xcor[s0:e0][:, keep]) > fthr
                arms[pol] = p
            pi = s_unp > thr[mode]; pr = s_unp > thr[mode]
            for s0, e0, P in segs_rec:
                Xi = Xcor[s0:e0].copy(); Pl = sorted(P)
                Xi[:, Pl] = med[Pl]
                pi[s0:e0] = score(prim, Xi) > thr[mode]
                vio = np.zeros(e0 - s0, bool)
                for j in P:
                    tj = vtimes.get(j, np.zeros(0, int)); tj = tj[(tj >= s0) & (tj < e0)]
                    vio[tj - s0] = True
                pr[s0:e0] |= vio
            arms["impute_recover"] = pi; arms["review_recover"] = pr
            for arm, pred in arms.items():
                info = {}
                segs = {"fallback": segs_stay, "fallback_recover": segs_rec, "impute_recover": segs_rec,
                        "review_recover": segs_rec}.get(arm)
                if segs is not None:
                    sw = np.zeros(n, bool)
                    for s0, e0, _ in segs:
                        sw[s0:e0] = True
                    ins = [s0 for s0, _, _ in segs if s0 == 0 or not sw[s0 - 1]]
                    false_in = [s0 for s0 in ins if not fault_mask[s0]]
                    clean_n = int((~fault_mask).sum()) if corr else n
                    first_in = [s0 for s0 in ins if fault_mask[s0]]
                    ends = [e0 for _, e0, _ in segs if e0 >= onset + dur]
                    info = dict(switches=len(ins), false_switches_per_m=len(false_in) / clean_n * 1e6,
                                time_switched=float(sw.mean()),
                                time_switched_clean=float(sw[~fault_mask].mean()),
                                switch_in_delay=(first_in[0] - onset) if (first_in and corr) else np.nan,
                                recovery_delay=(min(ends) - (onset + dur)) if (ends and corr and sw[onset + dur - 1]) else np.nan,
                                correct=int(bool(corr) and any(P & corr for s0, _, P in segs if fault_mask[s0])))
                rows.append(dict(base, threshold=mode, arm=arm, **metrics(pred, yt), **info))
        print(f"[{ds}] {fault} {choice} {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{ds}_closedloop_ext.csv"), index=False)


if __name__ == "__main__":
    main()
