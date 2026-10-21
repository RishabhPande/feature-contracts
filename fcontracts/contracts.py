"""Feature contracts: learning, calibration and online checking (Sections 4.1-4.5).

All checks are vectorized over a batch of requests and produce exactly what a
per-request check would (no information flows between requests). Outputs are
kept compact (top-3 evidence features and sparse violation pairs) so that
streams with hundreds of features fit in memory.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

from .config import CONFIG, KAPPA_GRID, T_GRID, RATE_MARGINS

UNSEEN_EXCESS = 10.0     # exceedance assigned to an unseen category (evidence only)


def wilson_upper(k, n, z=1.96):
    if n == 0:
        return 1.0
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + half) / den)


def conformal_threshold(scores, alpha):
    """ceil((n+1)(1-alpha))-th smallest score; inf if n is too small."""
    s = np.sort(np.asarray(scores, float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return math.inf if k > len(s) else float(s[k - 1])


class StreamResult:
    """Compact result of checking a stream of n requests."""

    def __init__(self, n):
        self.n = n
        self.flag = np.zeros(n, bool)
        self.top = np.full((n, 3), -1, np.int32)
        self.viol_rows, self.viol_cols = [], []
        self.hard_rows, self.hard_cols = [], []

    def finish(self):
        cat = lambda a: np.concatenate(a) if a else np.zeros(0, np.int64)
        self.viol_rows, self.viol_cols = cat(self.viol_rows), cat(self.viol_cols)
        self.hard_rows, self.hard_cols = cat(self.hard_rows), cat(self.hard_cols)
        return self


def top3_from_evidence(ev, viol):
    """Indices of up to three violated features with the highest evidence."""
    e = np.where(viol, ev, -np.inf)
    k = min(3, e.shape[1])
    idx = np.argpartition(-e, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(e, idx, 1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, 1)
    vals = np.take_along_axis(vals, order, 1)
    idx = np.where(np.isfinite(vals), idx, -1)
    out = np.full((e.shape[0], 3), -1, np.int32)
    out[:, :k] = idx
    return out


class FeatureContract:
    def __init__(self, groups, alpha, relations=True, calibration=None,
                 tier="hard", h=None, drift_screen=False):
        self.groups = [list(g) for g in groups if len(g) >= 2] if relations else []
        self.alpha = alpha
        self.relations = relations
        self.calibration = calibration or CONFIG["calibration"]
        self.tier = tier
        self.h = CONFIG["paging"]["severity_h"] if h is None else h
        self.rmult = CONFIG["paging"]["residual_hard_mult"]
        self.drift_screen = drift_screen
        self.version = None

    # ------------------------------------------------------------------ learn
    def fit(self, X_fit, X_cal):
        X_ref = np.vstack([X_fit, X_cal])
        d = X_fit.shape[1]
        self.d = d
        self.discrete = np.zeros(d, bool)
        self.support = [None] * d
        for j in range(d):
            col = X_ref[:, j]
            col = col[~np.isnan(col)]
            if col.size and np.all(np.mod(col, 1) == 0):
                u = np.unique(col)
                if u.size <= CONFIG["discrete_max_values"]:
                    self.discrete[j] = True
                    self.support[j] = u
        with np.errstate(all="ignore"):
            self.lo = np.nanmin(X_fit, 0)
            self.hi = np.nanmax(X_fit, 0)
            s = np.nanquantile(X_fit, 0.99, 0) - np.nanquantile(X_fit, 0.01, 0)
        rng_ = self.hi - self.lo
        s = np.where(s > 0, s, np.where(rng_ > 0, rng_, np.maximum(np.abs(self.hi), 1.0) * 1e-3))
        self.s = np.nan_to_num(s, nan=1.0)
        self.lo = np.nan_to_num(self.lo, nan=0.0)
        self.hi = np.nan_to_num(self.hi, nan=0.0)
        # nullability (v3): a feature never missing in the reference may not be missing
        self.non_nullable = ~np.isnan(X_ref).any(0)
        # drift screen (v3): exempt features that trend with time in the reference
        self.exempt = np.zeros(d, bool)
        self.trend = np.zeros(d)
        if self.drift_screen:
            from scipy.stats import spearmanr
            m = CONFIG["drift_screen"]
            idx = np.linspace(0, len(X_ref) - 1, min(m["sample"], len(X_ref))).astype(int)
            for j in range(d):
                col = X_ref[idx, j]
                ok = ~np.isnan(col)
                if ok.sum() > 100 and np.nanstd(col[ok]) > 0:
                    rho = spearmanr(idx[ok], col[ok]).statistic
                    self.trend[j] = 0.0 if np.isnan(rho) else rho
            self.exempt = np.abs(self.trend) >= m["spearman_abs"]

        # domain margin
        ex_cal, unseen_cal = self._exceed(X_cal)
        if self.calibration == "conformal1":
            return self._fit_single_score(X_fit, X_cal, ex_cal, unseen_cal)
        if self.calibration == "conformal":
            req = np.where(self.discrete[None, :], np.where(unseen_cal, np.inf, 0.0), ex_cal).max(1)
            k = conformal_threshold(req, self.alpha / 2)
            ok = [g for g in KAPPA_GRID if g >= k]
            self.kappa = ok[0] if ok else KAPPA_GRID[-1]
        else:
            self.kappa = KAPPA_GRID[-1]
            for kap in KAPPA_GRID:
                v = self._domain_viol(ex_cal, unseen_cal, kap).any(1)
                if v.mean() <= self.alpha / 2:
                    self.kappa = kap
                    break
        self.env_lo = self.lo - self.kappa * self.s
        self.env_hi = self.hi + self.kappa * self.s

        # relations
        self.rel = []
        for g in self.groups:
            Z = self._transform(X_fit, g, fit=True)
            lw = LedoitWolf().fit(Z)
            P = np.linalg.pinv(lw.covariance_)
            diag = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            R = (Z @ P) / diag
            sd = R.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            self.rel.append({"g": np.array(g), "P": P, "diag": diag, "sd": sd,
                             "tf": self._tf_last})
        dom_cal = self._domain_viol(ex_cal, unseen_cal, self.kappa).any(1)
        if self.rel:
            rmax = self._max_abs_resid(X_cal)
            if self.calibration == "conformal":
                tt = conformal_threshold(rmax, self.alpha / 2)
                ok = [t for t in T_GRID if t >= tt]
                self.t = ok[0] if ok else T_GRID[-1]
            else:
                self.t = T_GRID[-1]
                for t in T_GRID:
                    if (dom_cal | (rmax > t)).mean() <= self.alpha:
                        self.t = t
                        break
        else:
            self.t = math.inf

        # realized calibration-slice rates, p0 for paging
        res = self.check(X_cal)
        n = X_cal.shape[0]
        z = CONFIG["paging"]["wilson_z"]
        wil = lambda cnt: np.array([max(wilson_upper(c, n, z), CONFIG["paging"]["p0_floor"]) for c in cnt])
        self.p0_hard = wil(np.bincount(res.hard_cols, minlength=d))
        self.p0_all = wil(np.bincount(res.viol_cols, minlength=d))
        self.p0 = self.p0_hard if self.tier == "hard" else self.p0_all
        from .paging import onset_stats
        self.p0_onset, self.max_run = onset_stats(res.viol_rows, res.viol_cols, d, n)
        self.cal_flag_rate = res.flag.mean()

        # rate checks
        self.rates = RateCheck().fit(X_fit, X_cal)
        self.version = f"v{abs(hash((self.kappa, self.t, d))) % 10**8:08d}"
        return self

    # ---------------------------------------------- exact single-score option
    def _fit_single_score(self, X_fit, X_cal, ex_cal, unseen_cal):
        """Exact split-conformal calibration of the full rule (post-confirmation option).

        Score s(x) = max(domain exceedance in spreads, max residual / b), with +inf for an
        unseen category or a nullability violation. The relation transform clips to the fit
        envelope (kappa = 0) and b is set on the fit slice, so the score is fixed before the
        calibration slice is used; one conformal quantile q at level alpha then gives
        P(flag) <= alpha for an exchangeable clean request. kappa = q and tau = b q.
        """
        d = self.d
        # value sets and nullability from the fit slice only, so calibration scores are exchangeable
        # with new requests (the grid rule learns them on fit + calibration)
        self.discrete = np.zeros(d, bool)
        self.support = [None] * d
        for j in range(d):
            col = X_fit[:, j]
            col = col[~np.isnan(col)]
            if col.size and np.all(np.mod(col, 1) == 0):
                u = np.unique(col)
                if u.size <= CONFIG["discrete_max_values"]:
                    self.discrete[j] = True
                    self.support[j] = u
        self.non_nullable = ~np.isnan(X_fit).any(0)
        ex_cal, unseen_cal = self._exceed(X_cal)
        self.kappa = 0.0
        self.env_lo, self.env_hi = self.lo.copy(), self.hi.copy()
        self.rel = []
        for g in self.groups:
            Z = self._transform(X_fit, g, fit=True)
            lw = LedoitWolf().fit(Z)
            P = np.linalg.pinv(lw.covariance_)
            diag = np.sqrt(np.clip(np.diag(P), 1e-12, None))
            R = (Z @ P) / diag
            sd = R.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            self.rel.append({"g": np.array(g), "P": P, "diag": diag, "sd": sd, "tf": self._tf_last})
        if self.rel:
            rfit = self._max_abs_resid(X_fit)
            self.b = float(max(np.quantile(rfit, 1 - self.alpha / 2), 1e-6))
        else:
            self.b = math.inf
        sc = self.score(X_cal)
        self.q = conformal_threshold(sc, self.alpha)
        self.cal_scores = np.sort(sc[np.isfinite(sc)])
        self.kappa = self.q if math.isfinite(self.q) else KAPPA_GRID[-1]
        self.t = self.b * self.q if (self.rel and math.isfinite(self.q)) else math.inf
        # domain envelope for checking; relation clipping keeps the fit envelope stored in tf
        self.env_lo = self.lo - self.kappa * self.s
        self.env_hi = self.hi + self.kappa * self.s
        res = self.check(X_cal)
        n = X_cal.shape[0]
        z = CONFIG["paging"]["wilson_z"]
        wil = lambda cnt: np.array([max(wilson_upper(c, n, z), CONFIG["paging"]["p0_floor"]) for c in cnt])
        self.p0_hard = wil(np.bincount(res.hard_cols, minlength=d))
        self.p0_all = wil(np.bincount(res.viol_cols, minlength=d))
        self.p0 = self.p0_hard if self.tier == "hard" else self.p0_all
        from .paging import onset_stats
        self.p0_onset, self.max_run = onset_stats(res.viol_rows, res.viol_cols, d, n)
        self.cal_flag_rate = res.flag.mean()
        self.rates = RateCheck().fit(X_fit, X_cal)
        self.version = f"c1{abs(hash((self.q, d))) % 10**8:08d}"
        return self

    def score(self, X, chunk=20000):
        """Per-request single score (requires the conformal1 fit or a fitted b)."""
        out = np.zeros(X.shape[0])
        for a in range(0, X.shape[0], chunk):
            Xc = X[a:a + chunk]
            ex, unseen = self._exceed(Xc)
            dom = np.where(self.discrete[None, :], 0.0, ex).max(1)
            dom = np.where(unseen.any(1), np.inf, dom)
            if self.rel and math.isfinite(self.b):
                dom = np.maximum(dom, self._max_abs_resid(Xc) / self.b)
            out[a:a + chunk] = dom
        return out

    def _exceed(self, X):
        below = (self.lo - X) / self.s
        above = (X - self.hi) / self.s
        ex = np.fmax(below, above)
        ex = np.where(np.isnan(ex), 0.0, np.maximum(ex, 0.0))
        unseen = np.zeros(X.shape, bool)
        if getattr(self, "exempt", None) is not None and self.exempt.any():
            ex[:, self.exempt] = 0.0
        for j in np.flatnonzero(self.discrete & ~getattr(self, "exempt", np.zeros(self.d, bool))):
            col = X[:, j]
            unseen[:, j] = ~np.isnan(col) & ~np.isin(col, self.support[j])
        nn = getattr(self, "non_nullable", None)
        if nn is not None and nn.any():
            unseen[:, nn] |= np.isnan(X[:, nn])      # missing where never missing
        return ex, unseen

    def _domain_viol(self, ex, unseen, kappa):
        return np.where(self.discrete[None, :], unseen, ex > kappa) | unseen

    def _transform(self, X, g, fit=False):
        Xg = X[:, g]
        if fit:
            env_lo, env_hi = self.env_lo[g], self.env_hi[g]
            C = np.clip(Xg, env_lo, env_hi)
            med = np.nanmedian(C, 0)
            iqr = np.nanquantile(C, 0.75, 0) - np.nanquantile(C, 0.25, 0)
            iqr = np.where(iqr > 0, iqr, self.s[g])
            A = np.arcsinh((C - med) / iqr)
            mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
            sd = np.where(sd > 0, sd, 1.0)
            self._tf_last = (env_lo, env_hi, med, iqr, mu, sd)
        else:
            raise RuntimeError
        Z = (A - mu) / sd
        return np.nan_to_num(Z, nan=0.0)

    def _resid(self, X, r):
        env_lo, env_hi, med, iqr, mu, sd = r["tf"]
        C = np.clip(X[:, r["g"]], env_lo, env_hi)
        Z = np.nan_to_num((np.arcsinh((C - med) / iqr) - mu) / sd, nan=0.0)
        return (Z @ r["P"]) / r["diag"] / r["sd"]

    def _max_abs_resid(self, X):
        m = np.zeros(X.shape[0])
        for r in self.rel:
            m = np.maximum(m, np.abs(self._resid(X, r)).max(1))
        return m

    # ------------------------------------------------------------------ check
    def check(self, X, chunk=20000, use_relations=True):
        n, d = X.shape
        out = StreamResult(n)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            Xc = X[a:b]
            ex, unseen = self._exceed(Xc)
            dom = self._domain_viol(ex, unseen, self.kappa)
            excess = np.where(unseen, UNSEEN_EXCESS,
                              np.where(self.discrete[None, :], 0.0, np.maximum(ex - self.kappa, 0.0)))
            hard = unseen | (dom & (excess >= self.h))
            absr = np.zeros(Xc.shape)
            if use_relations:
                for r in self.rel:
                    absr[:, r["g"]] = np.maximum(absr[:, r["g"]], np.abs(self._resid(Xc, r)))
                relv = absr > self.t
                viol = dom | relv
                hard = hard | (absr > self.rmult * self.t)
            else:
                viol = dom
            ev = np.log10(1.0 + excess) + (absr / self.t if math.isfinite(self.t) else 0.0)
            out.flag[a:b] = viol.any(1)
            out.top[a:b] = top3_from_evidence(ev, viol)
            rr, cc = np.nonzero(viol)
            out.viol_rows.append(rr + a); out.viol_cols.append(cc)
            rr, cc = np.nonzero(hard)
            out.hard_rows.append(rr + a); out.hard_cols.append(cc)
        return out.finish()


class RateCheck:
    """Missing-value and consecutive-repeat rates per micro-batch (Section 4.1)."""

    def __init__(self):
        c = CONFIG["rate"]
        self.w, self.tail, self.margins = c["window"], c["tail"], c["margins"]

    def _counts(self, X):
        n, d = X.shape
        nw = n // self.w
        if nw == 0:
            return np.zeros((0, d)), np.zeros((0, d))
        Xw = X[: nw * self.w]
        miss = np.isnan(Xw)
        rep = np.zeros_like(miss)
        rep[1:] = (Xw[1:] == Xw[:-1])
        rep[0::self.w] = False
        m = miss.reshape(nw, self.w, d).sum(1)
        r = rep.reshape(nw, self.w, d).sum(1)
        return m, r

    def fit(self, X_fit, X_cal):
        mf, rf = self._counts(X_fit)
        mc, rc = self._counts(X_cal)
        self.thr = {}
        for name, F, C in (("missing", mf, mc), ("repeat", rf, rc)):
            if F.shape[0] == 0:
                self.thr[name] = np.full(X_fit.shape[1], np.inf)
                continue
            p = np.maximum(F.sum(0) / (F.shape[0] * self.w), 1e-6)
            base = stats.binom.ppf(1 - self.tail, self.w, p)
            block = F.max(0)
            thr = np.full(F.shape[1], np.inf)
            for j in range(F.shape[1]):
                for m in self.margins:
                    t = max(base[j], m * block[j])
                    if C.shape[0] == 0 or not (C[:, j] > t).any():
                        thr[j] = t
                        break
            self.thr[name] = thr
        return self

    def alarms(self, X):
        """List of (end_row, feature, kind) for windows that exceed a threshold."""
        m, r = self._counts(X)
        out = []
        for name, M in (("missing", m), ("repeat", r)):
            ww, jj = np.nonzero(M > self.thr[name][None, :])
            out += [((w + 1) * self.w - 1, j, name) for w, j in zip(ww, jj)]
        return out
