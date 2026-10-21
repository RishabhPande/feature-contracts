"""Closed-loop response (v3): does correct localization lead to the correct
operational response, and what does it buy in decision quality?

A fault starts 5,000 requests into the test stream and lasts to its end. The
contract checks every request; the first page (at any time, so false pages
before the fault also trigger a response, as they would in production) switches
scoring to a fallback model trained without the paged features. Arms, scored on
the faulted period:
  clean             primary model on clean data (reference)
  unprotected       primary model on corrupted data
  mask              primary model, violated features masked, clean threshold
  contract          unprotected until the first page, then fallback(paged features)
  contract_mask     mask until the first page, then fallback(paged features)
  oracle            fallback without the truly corrupted features, from onset

python3 closedloop.py --dataset default [--out results_v3] [--paging hard|burst]
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
from fcontracts.paging import run_paging, burst_aware_pages

ONSET = 5000


def fit_model(X, y):
    return HistGradientBoostingClassifier(random_state=0).fit(np.clip(X, -1e30, 1e30), y)


def score(m, X):
    return m.predict_proba(np.clip(X, -1e30, 1e30))[:, 1]


def thr_at(m, X, y, fpr=0.01):
    s = score(m, X)
    return float(np.quantile(s[y == 0], 1 - fpr))


def rates(pred, y):
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


def metrics(pred, y):
    """Recall, FPR, precision and expected cost per 1,000 requests for missed-fraud
    to false-alarm cost ratios of 10 and 50 (a false alarm costs 1)."""
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); n = len(y)
    return dict(recall=tp / max(1, tp + fn), fpr=fp / max(1, (y == 0).sum()),
                precision=tp / max(1, tp + fp),
                cost10=1000 * (fp + 10 * fn) / n, cost50=1000 * (fp + 50 * fn) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--faults", nargs="*", default=["none", "magnitude", "scale", "encoding", "default_fill", "nan_fill", "stale_lag", "misjoin_ref"])
    ap.add_argument("--prevalence", type=float, nargs="*", default=[1.0, 0.1])
    ap.add_argument("--choices", nargs="*", default=["top1", "top5", "unimportant1"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--imp-sample", type=int, default=30000)
    a = ap.parse_args()
    ds = a.dataset
    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    Xf, yf, Xc, yc, Xt, yt = X[sf], y[sf], X[sc], y[sc], X[st], y[st]
    d = X.shape[1]
    con = FeatureContract(groups, CONFIG["alpha"][ds]).fit(Xf, Xc)
    ref = reference_stats(np.vstack([Xf, Xc]))
    prim = fit_model(Xf, yf)
    thr = thr_at(prim, Xc, yc)
    rng0 = np.random.default_rng(0)
    sub = rng0.choice(len(Xc), size=min(a.imp_sample, len(Xc)), replace=False)
    imp = permutation_importance(prim, np.clip(Xc[sub], -1e30, 1e30), yc[sub], scoring="roc_auc",
                                 n_repeats=2, random_state=0).importances_mean
    order = np.argsort(-imp)
    pd.DataFrame({"feature": cols, "importance": imp}).to_csv(os.path.join(a.out, f"{ds}_importance.csv"), index=False)
    cache = {}

    def fallback(feats):
        k = tuple(sorted(set(int(f) for f in feats)))
        if k not in cache:
            keep = np.array([j for j in range(d) if j not in k])
            m = fit_model(Xf[:, keep], yf)
            cache[k] = (keep, m, thr_at(m, Xc[:, keep], yc))
        return cache[k]

    ev = slice(ONSET, len(Xt))
    y_ev = yt[ev]
    clean_m = metrics(score(prim, Xt[ev]) > thr, y_ev)
    rows = []
    combos = [c for c in itertools.product(a.faults, a.prevalence, a.choices, a.seeds)
              if not (c[0] == "none" and (c[1] != a.prevalence[0] or c[2] != a.choices[0]))]
    for fault, prev, choice, seed in combos:
        rng = np.random.default_rng([seed, int(prev * 100), hash(fault) % 997, hash(choice) % 991])
        if choice == "top1":
            feats = order[:1]
        elif choice == "top5":
            feats = order[:5]
        else:
            feats = rng.choice(order[min(20, d - 1):], size=1, replace=False)
        Xcor = Xt.copy()
        if fault == "none":
            feats = np.array([], int); changed = np.zeros((len(Xt) - ONSET, d), bool)
        else:
            Xw, changed = inject(Xt[ev], fault, feats, prev, rng, ref, pool=Xc)
            Xcor[ev] = Xw
        res = con.check(Xcor)
        alarms = con.rates.alarms(Xcor)
        out = {}
        for pname in ("hard", "burst"):
            if pname == "hard":
                pages = run_paging(res.hard_rows, res.hard_cols, d, con.p0_hard)
            else:
                pages = burst_aware_pages(res.viol_rows, res.viol_cols, d, con.p0_onset, con.max_run,
                                          min_run=CONFIG["burst"]["min_run"], run_mult=CONFIG["burst"]["run_mult"])
            pages = sorted(pages + [(t, j) for t, j, _ in alarms])
            out[pname] = pages
        # sporadic masking of violated features (evaluation period only, to save memory)
        sel = res.viol_rows >= ONSET
        Xm = Xcor[ONSET:].copy()
        Xm[res.viol_rows[sel] - ONSET, res.viol_cols[sel]] = np.nan
        s_unp = score(prim, Xcor[ONSET:])
        s_mask = score(prim, Xm)
        del Xm
        import gc; gc.collect()
        base = dict(dataset=ds, fault=fault, prevalence=prev, choice=choice, seed=seed,
                    features="|".join(cols[j] for j in feats), n_changed=int(changed.any(1).sum()))
        rows.append(dict(base, arm="clean", **clean_m))
        rows.append(dict(base, arm="unprotected", **metrics(s_unp > thr, y_ev)))
        rows.append(dict(base, arm="mask", **metrics(s_mask > thr, y_ev)))
        if len(feats):
            keep, om, othr = fallback(feats)
            rows.append(dict(base, arm="oracle", **metrics(score(om, Xcor[ev][:, keep]) > othr, y_ev)))
        for pname, pages in out.items():
            if pages:
                T = pages[0][0]
                paged = {j for t, j in pages if t == T}
            else:
                T, paged = None, set()
            corr = set(int(f) for f in feats)
            info = dict(first_page=T, response_delay=(T - ONSET) if (T is not None and T >= ONSET) else np.nan,
                        pre_onset_response=int(T is not None and T < ONSET),
                        paged="|".join(cols[j] for j in sorted(paged)),
                        correct_localization=int(bool(paged & corr)),
                        complete_localization=int(corr <= paged),
                        wrong_response=int(bool(paged) and not (paged & corr)),
                        pages_in_period=sum(1 for t, _ in pages if t >= ONSET))
            for arm, sc_before in (("contract", s_unp), ("contract_mask", s_mask)):
                pred = (sc_before > thr).copy()          # evaluation period only
                if T is not None and T < len(Xt):
                    keep, fm, fthr = fallback(paged)
                    s0 = max(T, ONSET)
                    pred[s0 - ONSET:] = score(fm, Xcor[s0:][:, keep]) > fthr
                rows.append(dict(base, **info, arm=f"{arm}|{pname}", **metrics(pred, y_ev)))
        print(f"[{ds}] {fault} {prev} {choice} {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{ds}_closedloop.csv"), index=False)


if __name__ == "__main__":
    main()
