"""Baselines (Section 5.3). Each detector has fit(X_fit, X_cal, alpha) and
check(X) -> StreamResult, plus `localizes` and `pooled` (pages on one stream).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .contracts import StreamResult, top3_from_evidence


def calibrated_threshold(scores_cal, alpha):
    """Smallest threshold with share(score > thr) <= alpha on the calibration slice."""
    s = np.sort(scores_cal)
    n = len(s)
    k = int(np.floor(alpha * n))          # number allowed above
    return s[n - k - 1] if k < n else -np.inf


class Schema:
    localizes, pooled, name = False, True, "schema"

    def fit(self, X_fit, X_cal, alpha=None):
        return self

    def check(self, X):
        r = StreamResult(X.shape[0])
        return r.finish()        # types and nullability always pass on numeric data


class MinMax:
    localizes, pooled, name = True, True, "minmax"

    def fit(self, X_fit, X_cal, alpha=None):
        X_ref = np.vstack([X_fit, X_cal])
        self.lo, self.hi = np.nanmin(X_ref, 0), np.nanmax(X_ref, 0)
        self.s = np.nan_to_num(np.nanquantile(X_ref, .99, 0) - np.nanquantile(X_ref, .01, 0), nan=1.0)
        self.s = np.where(self.s > 0, self.s, 1.0)
        return self

    def _viol(self, X):
        ex = np.fmax(self.lo - X, X - self.hi) / self.s
        ex = np.where(np.isnan(ex), 0.0, ex)
        return ex > 0, np.log10(1 + np.maximum(ex, 0))

    def check(self, X, chunk=20000):
        r = StreamResult(X.shape[0])
        for a in range(0, X.shape[0], chunk):
            v, ev = self._viol(X[a:a + chunk])
            r.flag[a:a + chunk] = v.any(1)
            r.top[a:a + chunk] = top3_from_evidence(ev, v)
            rr, cc = np.nonzero(v)
            r.viol_rows.append(rr + a); r.viol_cols.append(cc)
        r.finish()
        r.hard_rows, r.hard_cols = r.viol_rows, r.viol_cols
        return r


class AutoProfile(MinMax):
    """Nullability, value sets (<= 50 integer values) and ranges, like a
    data-validation library's profiler."""
    name = "autoprofile"

    def fit(self, X_fit, X_cal, alpha=None):
        super().fit(X_fit, X_cal)
        X_ref = np.vstack([X_fit, X_cal])
        self.nullable = np.isnan(X_ref).any(0)
        self.sets = {}
        for j in range(X_ref.shape[1]):
            c = X_ref[:, j]; c = c[~np.isnan(c)]
            if c.size and np.all(np.mod(c, 1) == 0):
                u = np.unique(c)
                if u.size <= 50:
                    self.sets[j] = u
        return self

    def _viol(self, X):
        v, ev = super()._viol(X)
        v = v | (np.isnan(X) & ~self.nullable[None, :])
        for j, u in self.sets.items():
            bad = ~np.isnan(X[:, j]) & ~np.isin(X[:, j], u)
            v[:, j] |= bad
            ev[:, j] = np.where(bad, np.maximum(ev[:, j], 1.0), ev[:, j])
        return v, ev


class _Scored:
    """Calibrated request-level score; optional per-feature contributions."""
    localizes, pooled = False, True

    def check(self, X, chunk=20000):
        r = StreamResult(X.shape[0])
        for a in range(0, X.shape[0], chunk):
            Xc = X[a:a + chunk]
            s, contrib = self.score(Xc)
            f = s > self.thr
            r.flag[a:a + chunk] = f
            if contrib is not None:
                r.top[a:a + chunk] = top3_from_evidence(contrib, np.repeat(f[:, None], contrib.shape[1], 1))
            rr = np.flatnonzero(f)
            r.viol_rows.append(rr + a); r.viol_cols.append(np.zeros_like(rr))
        r.finish()
        r.hard_rows, r.hard_cols = r.viol_rows, r.viol_cols
        return r

    def calibrate(self, X_cal, alpha):
        self.thr = calibrated_threshold(self.score(X_cal)[0], alpha)
        return self


class RobustZ(_Scored):
    localizes, name = True, "robust_z"

    def fit(self, X_fit, X_cal, alpha):
        self.med = np.nanmedian(X_fit, 0)
        mad = np.nanmedian(np.abs(X_fit - self.med), 0) * 1.4826
        sd = np.nanstd(X_fit, 0)
        self.mad = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
        return self.calibrate(X_cal, alpha)

    def score(self, X):
        z = np.nan_to_num(np.abs(X - self.med) / self.mad)
        return z.max(1), z


class IForest(_Scored):
    name = "iforest"

    def __init__(self, n_estimators=100, max_samples=256, seed=0):
        self.kw = dict(n_estimators=n_estimators, max_samples=max_samples, random_state=seed)

    def fit(self, X_fit, X_cal, alpha):
        self.med = np.nanmedian(X_fit, 0)
        self.m = IsolationForest(**self.kw).fit(np.where(np.isnan(X_fit), self.med, X_fit))
        return self.calibrate(X_cal, alpha)

    def score(self, X):
        Xf = np.where(np.isnan(X), self.med, X)
        Xf = np.clip(Xf, -1e300, 1e300)
        return -self.m.score_samples(Xf), None


class _Std:
    def _std_fit(self, X):
        self.mu = np.nanmean(X, 0)
        sd = np.nanstd(X, 0)
        self.sd = np.where(sd > 0, sd, 1.0)

    def _std(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z), -1e6, 1e6)


class PCARecon(_Scored, _Std):
    localizes, name = True, "pca_recon"

    def fit(self, X_fit, X_cal, alpha):
        self._std_fit(X_fit)
        self.p = PCA(n_components=0.9, svd_solver="full").fit(self._std(X_fit))
        return self.calibrate(X_cal, alpha)

    def score(self, X):
        Z = self._std(X)
        E = (Z - self.p.inverse_transform(self.p.transform(Z))) ** 2
        return E.sum(1), E


class Conformance(_Scored, _Std):
    """Fariha et al. (2021) style: lower-variance half of principal components;
    score = largest normalized deviation; responsibility by intervention."""
    localizes, name = True, "conformance"

    def fit(self, X_fit, X_cal, alpha):
        self._std_fit(X_fit)
        Z = self._std(X_fit)
        p = PCA(svd_solver="full").fit(Z)
        k = Z.shape[1] // 2
        self.W = p.components_[-max(k, 1):]          # low-variance directions
        proj = (Z - p.mean_) @ self.W.T
        self.c = p.mean_
        self.psd = np.where(proj.std(0) > 0, proj.std(0), 1.0)
        return self.calibrate(X_cal, alpha)

    def _s(self, Z):
        return np.abs(((Z - self.c) @ self.W.T) / self.psd).max(1)

    def score(self, X):
        Z = self._std(X)
        Wn = self.W.T / self.psd                     # (d, k)
        proj = (Z - self.c) @ Wn                     # (n, k)
        s = np.abs(proj).max(1)
        contrib = np.empty_like(Z)
        for j in range(Z.shape[1]):            # intervention: reset feature j to its mean (z = 0)
            pj = proj - np.outer(Z[:, j], Wn[j])
            contrib[:, j] = s - np.abs(pj).max(1)
        return s, contrib


class PSI:
    """Population stability index per feature on fixed windows; pages directly."""
    name, localizes, pooled = "psi", True, False

    def __init__(self, window=1000, bins=10, level=0.25):
        self.w, self.bins, self.level = window, bins, level

    def fit(self, X_fit, X_cal, alpha=None):
        X_ref = np.vstack([X_fit, X_cal])
        self.edges, self.base = [], []
        for c in X_ref.T:
            c = c[~np.isnan(c)]
            e = np.unique(np.quantile(c, np.linspace(0, 1, self.bins + 1)[1:-1])) if c.size else np.array([])
            self.edges.append(e)
            self.base.append(self._hist(c, e))
        return self

    @staticmethod
    def _hist(c, e):
        h = np.bincount(np.searchsorted(e, c[~np.isnan(c)], side="right"), minlength=len(e) + 1).astype(float)
        h = (h + 0.5) / (h.sum() + 0.5 * len(h))
        return h

    def pages(self, X):
        out = []
        for a in range(0, X.shape[0] - self.w + 1, self.w):
            W = X[a:a + self.w]
            for j, (e, b) in enumerate(zip(self.edges, self.base)):
                q = self._hist(W[:, j], e)
                if np.sum((q - b) * np.log(q / b)) > self.level:
                    out.append((a + self.w - 1, j))
        return out
