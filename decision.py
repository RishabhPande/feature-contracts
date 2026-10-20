"""Decision impact (Sections 6.6 and 6.8): clean, corrupted, masked, masked with a
mask-specific threshold (new in v2), and fallback scoring.

python3 decision.py --dataset default [--models hgb lr mlp] [--out results]
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats


class Model:
    """Wraps a classifier; models that cannot take NaN get median imputation."""

    def __init__(self, kind, seed=0):
        self.kind = kind
        if kind == "hgb":
            self.m = HistGradientBoostingClassifier(random_state=seed)
        elif kind == "lr":
            self.m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        elif kind == "mlp":
            self.m = make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=200,
                                                                    early_stopping=True, random_state=seed))

    def fit(self, X, y):
        self.med = np.nanmedian(X, 0)
        self.m.fit(self._prep(X), y)
        return self

    def _prep(self, X):
        if self.kind == "hgb":
            return np.clip(X, -1e300, 1e300)
        X = np.where(np.isnan(X), self.med, X)
        return np.clip(X, -1e12, 1e12)

    def score(self, X):
        return self.m.predict_proba(self._prep(X))[:, 1]


def thr_at_fpr(scores, y, fpr):
    return float(np.quantile(scores[y == 0], 1 - fpr))


def rates(pred, y):
    return float(pred[y == 1].mean()), float(pred[y == 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=["hgb", "lr", "mlp"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = CONFIG["decision"]
    ds = a.dataset
    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    X_fit, y_fit, X_cal, y_cal, X_test, y_test = X[sf], y[sf], X[sc], y[sc], X[st], y[st]
    con = FeatureContract(groups, CONFIG["alpha"][ds]).fit(X_fit, X_cal)
    ref = reference_stats(np.vstack([X_fit, X_cal]))
    rows = []
    for kind in a.models:
        mdl = Model(kind).fit(X_fit, y_fit)
        thr = thr_at_fpr(mdl.score(X_cal), y_cal, cfg["fpr"])
        imp = permutation_importance(mdl.m, mdl._prep(X_cal), y_cal, scoring="roc_auc",
                                     n_repeats=3, random_state=0).importances_mean
        order = np.argsort(-imp)
        clean_rec, clean_fpr = rates(mdl.score(X_test) > thr, y_test)
        fallbacks, mask_thr = {}, {}

        def fallback(feats):
            k = tuple(sorted(feats))
            if k not in fallbacks:
                keep = [j for j in range(X.shape[1]) if j not in k]
                fm = Model(kind).fit(X_fit[:, keep], y_fit)
                fallbacks[k] = (keep, fm, thr_at_fpr(fm.score(X_cal[:, keep]), y_cal, cfg["fpr"]))
            return fallbacks[k]

        def masked_threshold(pattern):
            if pattern not in mask_thr:
                Xm = X_cal.copy(); Xm[:, list(pattern)] = np.nan
                mask_thr[pattern] = thr_at_fpr(mdl.score(Xm), y_cal, cfg["fpr"])
            return mask_thr[pattern]

        for fault, prev, br, seed in itertools.product(cfg["faults"], cfg["prevalence"], cfg["breadth"], cfg["seeds"]):
            feats = order[:br]
            rng = np.random.default_rng([seed, br, int(prev * 100), hash(fault) % 1000])
            Xc, changed = inject(X_test, fault, feats, prev, rng, ref)
            base = dict(dataset=ds, model=kind, fault=fault, prevalence=prev, breadth=br, seed=seed,
                        features="|".join(cols[j] for j in feats))
            rows.append(dict(base, arm="clean", recall=clean_rec, fpr=clean_fpr))
            rows.append(dict(base, arm="corrupted", **dict(zip(("recall", "fpr"), rates(mdl.score(Xc) > thr, y_test)))))
            # mask what the contract flags
            res = con.check(Xc)
            Xm = Xc.copy()
            Xm[res.viol_rows, res.viol_cols] = np.nan
            sm = mdl.score(Xm)
            rows.append(dict(base, arm="masked", **dict(zip(("recall", "fpr"), rates(sm > thr, y_test)))))
            # mask-specific thresholds per mask pattern
            pat = {}
            for r_, c_ in zip(res.viol_rows, res.viol_cols):
                pat.setdefault(r_, []).append(c_)
            t_req = np.full(len(Xc), thr)
            groups_ = {}
            for r_, cs in pat.items():
                groups_.setdefault(tuple(sorted(set(cs))), []).append(r_)
            for p, rr in groups_.items():
                t_req[rr] = masked_threshold(p)
            rows.append(dict(base, arm="masked_rethreshold", **dict(zip(("recall", "fpr"), rates(sm > t_req, y_test)))))
            keep, fm, fthr = fallback(feats)
            rows.append(dict(base, arm="fallback", **dict(zip(("recall", "fpr"), rates(fm.score(Xc[:, keep]) > fthr, y_test)))))
            rows.append(dict(base, arm="flagged_share", recall=float(res.flag[changed.any(1)].mean()), fpr=np.nan))
        print(f"[{ds}] {kind} done", flush=True)
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{ds}_decision.csv"), index=False)


if __name__ == "__main__":
    main()
