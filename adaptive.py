"""Drift-aware calibration by online quantile tracking (post-confirmation analysis).

The per-request single score s_t (conformal1 calibration) is compared with a threshold
q_t that is updated online, in log1p units, by
    q_{t+1} = q_t + eta * (1[s_t > q_t] - alpha)
(adaptive conformal inference / quantile tracking). For any score sequence the long-run
flag rate converges to alpha, so the target is met under drift; the price is that a
sustained fault also raises q_t and is gradually absorbed. A guarded variant pauses the
update while the flag rate in the last 500 requests exceeds 20 times the rate in the
preceding 5,000 (an abrupt change); gradual drift does not trigger the pause.
Hard rules (unseen category, nullability) always flag and are not adapted.

Reports, on the clean time-ordered test stream: realized false flags overall and by
sixth, violation-test pages (burst-aware) and rate alarms; on injected faults: recall
over the window and in its first and last quarter (absorption), and event detection.

python3 adaptive.py --dataset ieee --out results_v3
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os

import numpy as np

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits
from fcontracts.inject import inject, reference_stats
from fcontracts.paging import burst_aware_pages

VARIANTS = {"fixed": dict(eta=0.0, guard=False), "aci_0.01": dict(eta=0.01, guard=False),
            "aci_0.05": dict(eta=0.05, guard=False), "aci_0.05_guard": dict(eta=0.05, guard=True)}


def track(u, q0, alpha, eta, guard, state=None):
    """u: log1p scores (inf allowed). Returns per-request threshold q_t and flags."""
    n = len(u)
    q = np.empty(n); fl = np.zeros(n, bool)
    qt = q0 if state is None else state["q"]
    hist = [] if state is None else list(state["hist"])
    recent = 0 if state is None else state["recent"]
    long_ = 0 if state is None else state["long"]
    buf = np.zeros(5500, bool) if state is None else state["buf"].copy()
    pos = 0 if state is None else state["pos"]
    for t in range(n):
        q[t] = qt
        f = bool(u[t] > qt)
        fl[t] = f
        # sliding counts: last 500 and the 5,000 before them
        old500 = int(buf[(pos - 500) % 5500]) if pos >= 500 else 0
        old5500 = int(buf[pos % 5500]) if pos >= 5500 else 0
        recent += int(f) - old500
        long_ += old500 - old5500
        buf[pos % 5500] = f
        pos += 1
        paused = guard and recent / 500 > 20 * max(long_ / 5000, alpha)
        if eta > 0 and math.isfinite(u[t]) and not paused:
            qt = max(0.0, qt + eta * (float(f) - alpha))
    st = {"q": qt, "hist": hist, "recent": recent, "long": long_, "buf": buf, "pos": pos}
    return q, fl, st


def feature_viol(c, X, qvec, chunk=20000):
    """Sparse (row, feature) violations of the single-score rule with per-request threshold."""
    rows, cols = [], []
    for a in range(0, len(X), chunk):
        Xc = X[a:a + chunk]
        ex, unseen = c._exceed(Xc)
        sj = np.where(c.discrete[None, :], 0.0, ex)
        if c.rel and math.isfinite(c.b):
            absr = np.zeros(Xc.shape)
            for r in c.rel:
                absr[:, r["g"]] = np.maximum(absr[:, r["g"]], np.abs(c._resid(Xc, r)))
            sj = np.maximum(sj, absr / c.b)
        v = (np.log1p(sj) > qvec[a:a + chunk][:, None]) | unseen
        rr, cc = np.nonzero(v)
        rows.append(rr + a); cols.append(cc)
    return np.concatenate(rows), np.concatenate(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--discrete-max", type=int, default=None,
                    help="value sets only for integer features with at most this many values (default: frozen config)")
    a = ap.parse_args()
    ds = a.dataset
    tag = ""
    if a.discrete_max is not None:
        CONFIG["discrete_max_values"] = a.discrete_max
        tag = f"_dm{a.discrete_max}"
    X, y, groups, cols = load(ds)
    alpha = CONFIG["alpha"][ds]
    sf, sc, st = splits(len(X))
    X_fit, X_cal, X_test = X[sf], X[sc], X[st]
    d = X.shape[1]
    c = FeatureContract(groups, alpha, calibration="conformal1").fit(X_fit, X_cal)
    q0 = math.log1p(c.q) if math.isfinite(c.q) else 50.0
    u_test = np.log1p(c.score(X_test))
    n = len(X_test)
    rate_alarms = c.rates.alarms(X_test)
    res = {"alpha": alpha, "n_test": n, "q0": q0, "rate_alarms": len(rate_alarms),
           "rate_alarms_per_m": len(rate_alarms) / n * 1e6, "variants": {}}
    states = {}
    for name, v in VARIANTS.items():
        q, fl, _ = track(u_test, q0, alpha, v["eta"], v["guard"])
        states[name] = q
        rows, cols_ = feature_viol(c, X_test, q)
        pages = burst_aware_pages(rows, cols_, d, c.p0_onset, c.max_run)
        six = np.array_split(np.arange(n), 6)
        res["variants"][name] = {
            "ff_per_m": float(fl.mean() * 1e6),
            "ff_by_sixth_per_m": [float(fl[i].mean() * 1e6) for i in six],
            "violation_pages": len(pages), "violation_pages_per_m": len(pages) / n * 1e6,
            "pages_by_sixth": [int(sum(1 for t, _ in pages if i[0] <= t <= i[-1])) for i in six],
            "q_end": float(q[-1]), "q_max": float(q.max())}
        print(ds, name, {k: v for k, v in res["variants"][name].items() if "sixth" not in k}, flush=True)

    # injected faults: state at window start comes from the clean run
    ref = reference_stats(np.vstack([X_fit, X_cal]))
    W = min(CONFIG["window"][ds], n)
    runs = []
    for seed in a.seeds:
        srng = np.random.default_rng(20_000 + seed)
        start = int(srng.integers(0, n - W + 1))
        win = X_test[start:start + W]
        pre = {}
        for name, v in VARIANTS.items():
            _, _, stt = track(u_test[:start], q0, alpha, v["eta"], v["guard"])
            pre[name] = stt
        for ci, (fault, prev, br) in enumerate(itertools.product(CONFIG["paper_faults"], [0.1, 1.0], CONFIG["breadth"])):
            rng = np.random.default_rng([seed, ci, 7])
            feats = rng.choice(d, size=min(br, d), replace=False)
            Xw, ch = inject(win, fault, feats, prev, rng, ref)
            crow = np.flatnonzero(ch.any(1))
            if len(crow) < CONFIG["min_changed"]:
                continue
            uw = np.log1p(c.score(Xw))
            for name, v in VARIANTS.items():
                stt = {k: (val.copy() if isinstance(val, np.ndarray) else val) for k, val in pre[name].items()}
                q, fl, _ = track(uw, None, alpha, v["eta"], v["guard"], state=stt)
                q1 = crow[crow < W // 4]; q4 = crow[crow >= 3 * W // 4]
                rows, cols_ = feature_viol(c, Xw, q)
                pages = burst_aware_pages(rows, cols_, d, c.p0_onset, c.max_run)
                det = any(ch[:, j].any() and j in feats for _, j in pages)
                runs.append(dict(variant=name, fault=fault, prevalence=prev, breadth=br, seed=seed,
                                 recall=float(fl[crow].mean()),
                                 recall_q1=float(fl[q1].mean()) if len(q1) else np.nan,
                                 recall_q4=float(fl[q4].mean()) if len(q4) else np.nan,
                                 detected=bool(det)))
    import pandas as pd
    df = pd.DataFrame(runs)
    df.to_csv(os.path.join(a.out, f"{ds}_adaptive_runs{tag}.csv"), index=False)
    for name in VARIANTS:
        g = df[df.variant == name]
        res["variants"][name].update({
            "recall": float(g.recall.mean()), "recall_q1": float(g.recall_q1.mean()), "recall_q4": float(g.recall_q4.mean()),
            "recall_systemic_q1": float(g[g.prevalence == 1.0].recall_q1.mean()),
            "recall_systemic_q4": float(g[g.prevalence == 1.0].recall_q4.mean()),
            "event_detection": float(g.detected.mean())})
        print(ds, name, "recall", res["variants"][name]["recall"], "q1/q4 systemic",
              res["variants"][name]["recall_systemic_q1"], res["variants"][name]["recall_systemic_q4"], flush=True)
    with open(os.path.join(a.out, f"{ds}_adaptive{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
