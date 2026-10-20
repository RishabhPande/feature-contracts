"""Fault-injection benchmark (Sections 5-6).

python3 run_experiments.py --dataset cc [--seeds 0 1 2 3 4] [--out results]

Writes results/<ds>_runs.csv (one row per run x method), results/<ds>_clean.json
(calibration, false flags and false pages on the clean test stream).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from fcontracts.config import CONFIG
from fcontracts.data import load, splits
from fcontracts.contracts import FeatureContract
from fcontracts import baselines as B
from fcontracts.inject import inject, reference_stats
from fcontracts.paging import run_paging, pooled_p0

PER = 1e5


class Res:
    """Plain container: flag, top, (viol rows, cols), (hard rows, cols)."""

    def __init__(self, flag, top, vr, vc, hr, hc):
        self.flag, self.top, self.vr, self.vc, self.hr, self.hc = flag, top, vr, vc, hr, hc

    @classmethod
    def of(cls, r):
        return cls(r.flag, r.top, r.viol_rows, r.viol_cols, r.hard_rows, r.hard_cols)

    def concat(self, other, offset):
        return Res(np.concatenate([self.flag, other.flag]), np.vstack([self.top, other.top]),
                   np.concatenate([self.vr, other.vr + offset]), np.concatenate([self.vc, other.vc]),
                   np.concatenate([self.hr, other.hr + offset]), np.concatenate([self.hc, other.hc]))


def build_detectors(X_fit, X_cal, groups, alpha):
    det = {
        "schema": B.Schema().fit(X_fit, X_cal),
        "minmax": B.MinMax().fit(X_fit, X_cal),
        "autoprofile": B.AutoProfile().fit(X_fit, X_cal),
        "robust_z": B.RobustZ().fit(X_fit, X_cal, alpha),
        "iforest": B.IForest().fit(X_fit, X_cal, alpha),
        "pca_recon": B.PCARecon().fit(X_fit, X_cal, alpha),
        "conformance": B.Conformance().fit(X_fit, X_cal, alpha),
        "contracts_domain": FeatureContract(groups, alpha, relations=False).fit(X_fit, X_cal),
        "contracts": FeatureContract(groups, alpha, relations=True).fit(X_fit, X_cal),
    }
    psi = B.PSI(**CONFIG["psi"]).fit(X_fit, X_cal)
    return det, psi


def paging_variants(name, det, res, d, cal_flag):
    """Yield (variant, pages) for one detector's result on a stream."""
    pc = CONFIG["paging"]
    if name == "contracts":
        c = det
        yield "contracts|all", run_paging(res.vr, res.vc, d, c.p0_all)
        yield "contracts|hard", run_paging(res.hr, res.hc, d, c.p0_hard)
        yield "contracts|cusum", run_paging(res.vr, res.vc, d, c.p0_all, test="cusum")
        for A in pc["extra_A"]:
            yield f"contracts|hard_A{int(A)}", run_paging(res.hr, res.hc, d, c.p0_hard, A=A)
            yield f"contracts|all_A{int(A)}", run_paging(res.vr, res.vc, d, c.p0_all, A=A)
        yield "contracts|every", sorted(set(zip(res.vr.tolist(), res.vc.tolist())))
    elif name in ("autoprofile", "minmax", "robust_z", "iforest", "pca_recon", "conformance"):
        yield f"{name}|pooled", run_paging(res.vr, res.vc, 1, pooled_p0(cal_flag[name]), pooled=True)


def with_rates(pages, alarms):
    return sorted(pages + [(t, j) for t, j, _ in alarms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["seeds"])
    ap.add_argument("--faults", nargs="*", default=CONFIG["faults"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = a.dataset
    t0 = time.time()

    X, y, groups, cols = load(ds)
    sf, sc, st = splits(len(X))
    X_fit, X_cal, X_test = X[sf], X[sc], X[st]
    d = X.shape[1]
    alpha = CONFIG["alpha"][ds]
    W, burn = CONFIG["window"][ds], CONFIG["burn_in"]
    det, psi = build_detectors(X_fit, X_cal, groups, alpha)
    ref = reference_stats(np.vstack([X_fit, X_cal]))
    con = det["contracts"]
    print(f"[{ds}] n={len(X)} d={d} alpha={alpha} kappa={con.kappa} t={con.t} "
          f"groups>=2: {len(con.rel)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- clean test stream: false flags and false pages
    cal_flag = {k: v.check(X_cal).flag for k, v in det.items()}
    clean = {"dataset": ds, "n_test": len(X_test), "d": d, "alpha": alpha,
             "kappa": con.kappa, "t": con.t, "kappa_domain": det["contracts_domain"].kappa,
             "cal_flag_rate": {k: float(v.mean()) for k, v in cal_flag.items()},
             "false_flags_per_100k": {}, "false_pages_per_100k": {}, "flag_bursts": {}}
    clean_alarms = con.rates.alarms(X_test)
    for k, dt in det.items():
        r = Res.of(dt.check(X_test))
        clean["false_flags_per_100k"][k] = float(r.flag.mean() * PER)
        f = r.flag.astype(int)
        clean["flag_bursts"][k] = int(((f[1:] == 1) & (f[:-1] == 0)).sum() + f[0])
        for var, pages in paging_variants(k, dt, r, d, cal_flag):
            if var.startswith("contracts|") and not var.endswith("every"):
                pages = with_rates(pages, clean_alarms)
            clean["false_pages_per_100k"][var] = len(pages) / len(X_test) * PER
    clean["false_pages_per_100k"]["contracts|rates"] = len(clean_alarms) / len(X_test) * PER
    clean["false_pages_per_100k"]["psi"] = len(psi.pages(X_test)) / len(X_test) * PER
    # which feature drives contract false flags
    rc = con.check(X_test)
    vc = np.bincount(rc.viol_cols, minlength=d)
    clean["top_false_flag_features"] = [(cols[j], int(vc[j])) for j in np.argsort(-vc)[:5] if vc[j]]
    with open(os.path.join(a.out, f"{ds}_clean.json"), "w") as fh:
        json.dump(clean, fh, indent=2)
    print(f"[{ds}] clean stream done ({time.time()-t0:.0f}s): "
          f"contracts ff={clean['false_flags_per_100k']['contracts']:.1f}", flush=True)

    # ---------------- fault runs
    rows = []
    n_test = len(X_test)
    W = min(W, n_test - burn)
    for seed in a.seeds:
        srng = np.random.default_rng(10_000 + seed)
        start = int(srng.integers(burn, n_test - W + 1))
        seg = X_test[start - burn:start + W]
        pre, win = seg[:burn], seg[burn:]
        pre_res = {k: Res.of(dt.check(pre)) for k, dt in det.items()}
        ctl_res = {k: pre_res[k].concat(Res.of(dt.check(win)), burn) for k, dt in det.items()}
        ctl_alarms = con.rates.alarms(seg)
        ctl_pages = {}
        for k, dt in det.items():
            for var, pages in paging_variants(k, dt, ctl_res[k], d, cal_flag):
                if var.startswith("contracts|") and not var.endswith("every"):
                    pages = with_rates(pages, ctl_alarms)
                ctl_pages[var] = pages
        ctl_pages["contracts|rates"] = [(t, j) for t, j, _ in ctl_alarms]
        ctl_pages["psi"] = psi.pages(seg)

        for ci, (fault, prev, br) in enumerate(itertools.product(a.faults, CONFIG["prevalence"], CONFIG["breadth"])):
            rng = np.random.default_rng([seed, ci])
            feats = rng.choice(d, size=min(br, d), replace=False)
            Xw, changed = inject(win, fault, feats, prev, rng, ref)
            crow = np.flatnonzero(changed.any(1))
            if len(crow) < CONFIG["min_changed"]:
                continue
            onset = burn + int(crow[0])
            segc = np.vstack([pre, Xw])
            alarms = con.rates.alarms(segc)
            base = dict(dataset=ds, fault=fault, prevalence=prev, breadth=br, seed=seed,
                        n_corrupted=len(crow), features="|".join(map(str, feats)))
            changed_full = np.zeros((len(seg), d), bool); changed_full[burn:] = changed
            corrupted_feats = set(feats.tolist())
            for k, dt in det.items():
                r = pre_res[k].concat(Res.of(dt.check(Xw)), burn)
                fl = r.flag[burn + crow]
                top = r.top[burn + crow]
                t1 = top[:, 0]
                ok1 = (t1 >= 0) & changed[crow, np.maximum(t1, 0)]
                ok3 = np.zeros(len(crow), bool)
                for m in range(3):
                    tm = top[:, m]
                    ok3 |= (tm >= 0) & changed[crow, np.maximum(tm, 0)]
                rec = dict(base, method=k, recall=fl.mean(),
                           flagged=int(fl.sum()), loc1=int((fl & ok1).sum()), loc3=int((fl & ok3).sum()))
                rows.append(rec)
                for var, pages in paging_variants(k, dt, r, d, cal_flag):
                    if var.startswith("contracts|") and not var.endswith("every"):
                        pages = with_rates(pages, alarms)
                    rows.append(dict(base, method=var, **page_stats(pages, ctl_pages[var], onset,
                                                                    len(seg), corrupted_feats, k != "contracts")))
            rows.append(dict(base, method="contracts|rates", **page_stats(
                [(t, j) for t, j, _ in alarms], ctl_pages["contracts|rates"], onset, len(seg), corrupted_feats, False)))
            rows.append(dict(base, method="psi", **page_stats(
                psi.pages(segc), ctl_pages["psi"], onset, len(seg), corrupted_feats, False)))
        print(f"[{ds}] seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{ds}_runs.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{ds}_runs.csv"), index=False)
    print(f"[{ds}] finished in {time.time()-t0:.0f}s", flush=True)


def page_stats(pages, ctl, onset, end, feats, pooled):
    ctl_by = {}
    for t, j in ctl:
        ctl_by.setdefault(j, []).append(t)
    det_t = None
    for t, j in pages:
        if t < onset:
            continue
        if not pooled and j not in feats:
            continue
        if any(onset <= tc <= t for tc in ctl_by.get(j, [])):
            continue
        det_t = t
        break
    n_win = sum(1 for t, _ in pages if t >= onset)
    return dict(detected=int(det_t is not None), delay=(det_t - onset) if det_t is not None else np.nan,
                pages_in_window=n_win)


if __name__ == "__main__":
    main()
