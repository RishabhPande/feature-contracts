"""Corruption operators (Table 1) and fault windows.

inject(X_window, fault, features, prevalence, rng, ref) returns a corrupted
copy and a boolean (n, d) mask of values that actually changed.
"""
from __future__ import annotations

import numpy as np

from .config import CONFIG


def reference_stats(X_ref):
    with np.errstate(all="ignore"):
        return {
            "q90": np.nanquantile(X_ref, CONFIG["clip_quantile"], 0),
            "sd": np.nanstd(X_ref, 0),
            "lo": np.nanmin(X_ref, 0),
            "hi": np.nanmax(X_ref, 0),
            "binary": np.array([set(np.unique(c[~np.isnan(c)])) <= {0.0, 1.0}
                                for c in X_ref.T]),
        }


def inject(Xw, fault, features, prevalence, rng, ref, scale=None, clip_q=None,
           noise_sd=None):
    X = Xw.copy()
    n = X.shape[0]
    k = max(1, int(round(prevalence * n)))
    rows = np.sort(rng.choice(n, size=k, replace=False))
    for j in features:
        col = X[:, j]
        v = col[rows]
        if fault == "magnitude":
            v = rng.uniform(1e8, 5e9, size=k)
        elif fault == "scale":
            v = v * (scale or CONFIG["scale_factor"])
        elif fault == "encoding":
            v = v * 255.0 if ref["binary"][j] else -v
        elif fault == "default_fill":
            v = np.zeros(k)
        elif fault == "nan_fill":
            v = np.full(k, np.nan)
        elif fault == "stale":
            v = np.full(k, col[0])          # frozen at the value when the fault began
        elif fault == "misjoin":
            v = rng.permutation(v)
        elif fault == "clipping":
            q = ref["q90"][j] if clip_q is None else clip_q[j]
            v = np.minimum(v, q)
        elif fault == "inconsistency":
            sd = ref["sd"][j] * (noise_sd or CONFIG["noise_sd"])
            v = np.clip(v + rng.normal(scale=sd, size=k), ref["lo"][j], ref["hi"][j])
        else:
            raise ValueError(fault)
        col[rows] = v
    changed = ~((X == Xw) | (np.isnan(X) & np.isnan(Xw)))
    return X, changed
