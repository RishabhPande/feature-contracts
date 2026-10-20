"""Drop-in helpers for the feature-contracts reproduction package.

1. conformal_margin(): order-statistic (split-conformal) choice of the envelope
   margin kappa, replacing the grid search in Algorithm 1, line 5.
   Guarantee: if the calibration requests and a new clean request are
   exchangeable (fit slice held fixed), P(new request has a domain violation)
   <= alpha.

2. conformal_threshold(): the same for any per-request score (e.g. the max
   standardized residual), for a single joint threshold.

3. calibration_conditional_bound(): a high-probability upper bound on the
   realized false-flag rate of the chosen threshold (binomial / Clopper-Pearson).

4. fit_dropout_model(): a primary model trained with random feature masking,
   for the extra decision-impact arm (masked scoring with a model that has
   seen missing values during training).

Integrate by calling these from run_experiments.py; nothing here reads data.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. Envelope margin
# ---------------------------------------------------------------------------
def required_margin(X_cal, lo, hi, s, support=None, discrete=None):
    """Smallest kappa for which each calibration request has no domain violation.

    X_cal: (n, d) array, NaN = missing (never a domain violation).
    lo, hi: per-feature min and max on D_fit. s: per-feature Q99 - Q01 on D_fit
    (zeros replaced by a small positive value by the caller or here).
    support / discrete: optional per-feature value sets for discrete features;
    an unseen category gives an infinite required margin.
    Returns an (n,) array kappa*_r.
    """
    X = np.asarray(X_cal, dtype=float)
    s = np.where(np.asarray(s, float) > 0, s, 1e-12)
    below = (lo - X) / s
    above = (X - hi) / s
    k = np.fmax(below, above)                    # NaN-safe max
    k = np.where(np.isnan(k), 0.0, k)
    k = np.maximum(k, 0.0)
    if discrete is not None and support is not None:
        for j in np.flatnonzero(discrete):
            col = X[:, j]
            unseen = ~np.isnan(col) & ~np.isin(col, list(support[j]))
            k[unseen, j] = np.inf
            # discrete features use the value set, not the envelope
            k[~unseen, j] = 0.0
    return k.max(axis=1)


def conformal_threshold(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    A new exchangeable clean request exceeds the returned value with
    probability at most alpha (strict '>' flags a violation). Returns inf if
    the calibration slice is too small to resolve alpha, i.e. n < 1/alpha - 1.
    """
    scores = np.sort(np.asarray(scores, float))
    n = scores.size
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(scores[k - 1])


def conformal_margin(X_cal, lo, hi, s, alpha, support=None, discrete=None,
                     grid=None):
    """Envelope margin with a finite-sample guarantee.

    If grid is given (e.g. the paper's {0, 0.1, ..., 55}), returns the smallest
    grid value >= the conformal margin, which keeps the guarantee.
    """
    k_star = required_margin(X_cal, lo, hi, s, support, discrete)
    kappa = conformal_threshold(k_star, alpha)
    if grid is not None and math.isfinite(kappa):
        ok = [g for g in sorted(grid) if g >= kappa]
        kappa = ok[0] if ok else math.inf
    return kappa


# ---------------------------------------------------------------------------
# 2. Joint score for domain + relation checks (one threshold, one guarantee)
# ---------------------------------------------------------------------------
def joint_score(k_star, max_abs_residual, kappa0, t0):
    """Combine the two checks into one score so a single conformal threshold
    controls the request-level rate. kappa0 and t0 set the relative scale and
    must be fixed on D_fit (e.g. their 99.9th percentiles there), not on D_cal.
    A request is flagged when joint_score > c, i.e. k* > c*kappa0 or
    |r| > c*t0.
    """
    return np.maximum(np.asarray(k_star) / kappa0,
                      np.asarray(max_abs_residual) / t0)


# ---------------------------------------------------------------------------
# 3. High-probability bound on the realized false-flag rate
# ---------------------------------------------------------------------------
def calibration_conditional_bound(n_cal, n_exceed, delta=0.05):
    """Upper (1-delta) Clopper-Pearson bound on the clean flag rate, given that
    n_exceed of n_cal held-out clean requests exceed the threshold.

    Use a second held-out slice (not the one that chose the threshold) or the
    clean test stream, so the count is independent of the choice.
    """
    if n_exceed >= n_cal:
        return 1.0
    return float(stats.beta.ppf(1 - delta, n_exceed + 1, n_cal - n_exceed))


# ---------------------------------------------------------------------------
# 4. Missingness-aware primary model
# ---------------------------------------------------------------------------
def fit_dropout_model(X_fit, y_fit, p_drop=0.1, n_copies=2, top_features=None,
                      p_top=0.3, random_state=0, **hgb_kwargs):
    """HistGradientBoostingClassifier trained on augmented copies of the data
    in which features are set to NaN at random, so masked scoring at serving
    time is in-distribution.

    p_drop: masking rate for every feature. top_features / p_top: optionally
    mask the most important features more often, since those are the ones the
    paper masks on every request. The original rows are kept.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(random_state)
    X_fit = np.asarray(X_fit, float)
    y_fit = np.asarray(y_fit)
    blocks, labels = [X_fit], [y_fit]
    for _ in range(n_copies):
        Xc = X_fit.copy()
        m = rng.random(Xc.shape) < p_drop
        if top_features is not None:
            m[:, top_features] |= rng.random((Xc.shape[0], len(top_features))) < p_top
        Xc[m] = np.nan
        blocks.append(Xc)
        labels.append(y_fit)
    model = HistGradientBoostingClassifier(random_state=random_state, **hgb_kwargs)
    model.fit(np.vstack(blocks), np.concatenate(labels))
    return model


def masked_thresholds(model, X_cal, y_cal, feature_sets, fpr=0.01):
    """Decision threshold per mask pattern, set at the target false positive
    rate on calibration data with those features set to NaN.

    In a synthetic check, masking with the clean threshold collapsed recall,
    while masking with a mask-specific threshold matched a fallback model.
    feature_sets: list of tuples of feature indices (e.g. each top-k feature
    alone and the top-5 together). Returns {tuple: threshold}.
    """
    X_cal = np.asarray(X_cal, float)
    neg = np.asarray(y_cal) == 0
    out = {}
    for fs in feature_sets:
        Xm = X_cal.copy()
        Xm[:, list(fs)] = np.nan
        p = model.predict_proba(Xm)[:, 1]
        out[tuple(fs)] = float(np.quantile(p[neg], 1 - fpr))
    return out


# ---------------------------------------------------------------------------
# Self-check on synthetic data (python3 conformal_and_dropout.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    d, alpha = 50, 1e-2
    X_fit = rng.standard_t(4, size=(20000, d))
    lo, hi = X_fit.min(0), X_fit.max(0)
    s = np.quantile(X_fit, 0.99, 0) - np.quantile(X_fit, 0.01, 0)
    rates = []
    for rep in range(200):
        X_cal = rng.standard_t(4, size=(2000, d))
        kappa = conformal_margin(X_cal, lo, hi, s, alpha)
        X_new = rng.standard_t(4, size=(5000, d))
        viol = ((X_new < lo - kappa * s) | (X_new > hi + kappa * s)).any(1)
        rates.append(viol.mean())
    print(f"target alpha={alpha}; mean realized flag rate over 200 draws="
          f"{np.mean(rates):.4f}; 95th pct={np.quantile(rates, .95):.4f}")
    print("Clopper-Pearson 95% bound for 3 exceedances in 2,000:",
          round(calibration_conditional_bound(2000, 3), 5))

    # dropout model sanity check
    n = 20000
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=.5, size=n) > 1.5).astype(int)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    base = HistGradientBoostingClassifier(random_state=0).fit(X[:15000], y[:15000])
    drop = fit_dropout_model(X[:15000], y[:15000], top_features=[0])
    Xt = X[15000:].copy(); Xt[:, 0] = np.nan          # mask the top feature
    print("AUC with top feature masked: plain HGB "
          f"{roc_auc_score(y[15000:], base.predict_proba(Xt)[:, 1]):.3f}, "
          f"dropout-trained {roc_auc_score(y[15000:], drop.predict_proba(Xt)[:, 1]):.3f}")

    # threshold check: clean threshold vs mask-specific threshold
    Xc, yc, Xe, ye = X[10000:15000], y[10000:15000], X[15000:], y[15000:]
    pc = base.predict_proba(Xc)[:, 1]
    thr_clean = np.quantile(pc[yc == 0], 0.99)
    thr_mask = masked_thresholds(base, Xc, yc, [(0,)])[(0,)]
    pm = base.predict_proba(np.where(np.arange(5) == 0, np.nan, Xe))[:, 1]
    for nm, t in [("clean threshold", thr_clean), ("mask threshold", thr_mask)]:
        print(f"masked scoring, {nm}: recall {(pm > t)[ye == 1].mean():.3f}, "
              f"FPR {(pm > t)[ye == 0].mean():.3f}")
