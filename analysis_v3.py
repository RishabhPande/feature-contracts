"""v3 tables: matched-rate detection, localization, ablation, operating points with
block-bootstrap intervals, supplementary faults, closed loop, sensitivity,
temporal windows. Writes results_v3/summary_v3.json and prints markdown.

python3 analysis_v3.py [--out results_v3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

CORE = ["magnitude", "scale", "encoding", "default_fill", "stale", "misjoin", "clipping", "inconsistency"]
SUPP = ["stale_lag", "misjoin_ref", "nan_fill"]
DS = ["cc", "ccv", "default", "ieee", "baf"]


def boot_mean(x, B=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return [np.nan] * 3
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def block_ci_rate(event_times, n, block=1000, B=2000, seed=0):
    """Rate per 100k with a block bootstrap over consecutive request blocks."""
    nb = n // block
    counts = np.bincount(np.minimum(np.asarray(event_times, int) // block, nb - 1), minlength=nb)[:nb]
    rng = np.random.default_rng(seed)
    est = counts.sum() / (nb * block) * 1e5
    bs = [counts[rng.integers(0, nb, nb)].sum() / (nb * block) * 1e5 for _ in range(B)]
    return [float(est), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v3")
    a = ap.parse_args()
    O = a.out
    S = {}
    for ds in DS:
        f = os.path.join(O, f"{ds}_runs.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f)
        c = json.load(open(os.path.join(O, f"{ds}_clean.json")))
        flags = np.load(os.path.join(O, f"{ds}_clean_flags.npz"))
        pages = json.load(open(os.path.join(O, f"{ds}_clean_pages.json")))
        n = c["n_test"]
        key = ["fault", "prevalence", "breadth", "seed"]
        core = r[r.fault.isin(CORE)]
        det = core[~core.method.str.contains(r"\||psi", regex=True)].copy()
        det["loc_paired"] = det.loc1 / det.n_corrupted
        det["loc_unpaired"] = det.loc1 / det.flagged.replace(0, np.nan)
        out = {"alpha": c["alpha"], "n_test": n, "exempt": c.get("exempt_features", [])}
        # operating points with block-bootstrap intervals (per 100k; x10 for per million)
        op = {}
        for m in ["minmax", "autoprofile", "conformance", "contracts_domain", "contracts", "contracts_ds"]:
            op[m] = {"flags": block_ci_rate(np.flatnonzero(flags[m]), n)}
        for v in ["contracts|all", "contracts|hard", "contracts|burst", "contracts|pooled", "contracts_ds|hard",
                  "contracts_ds|burst", "autoprofile|pooled", "autoprofile|perfeature", "conformance|pooled", "psi"]:
            if v in pages:
                op[v] = {"pages": block_ci_rate(pages[v], n)}
            elif v in c["false_pages_per_100k"]:
                op[v] = {"pages": [c["false_pages_per_100k"][v], np.nan, np.nan]}
        out["operating"] = op
        # detection and localization per method (core faults)
        meth = {}
        for m in ["minmax", "autoprofile", "robust_z", "iforest", "pca_recon", "conformance",
                  "contracts_domain", "contracts", "contracts_ds"]:
            s = det[det.method == m]
            meth[m] = {"recall": boot_mean(s.recall), "loc_paired": boot_mean(s.loc_paired),
                       "loc_unpaired": float(s.loc_unpaired.mean()), "ff": c["false_flags_per_100k"][m]}
        out["methods"] = meth
        wide = det.pivot_table(index=key, columns="method", values=["recall", "loc_paired"])
        diffs = {}
        for lab, col, x, y in [("recall_contracts_minus_autoprofile", "recall", "contracts", "autoprofile"),
                               ("recall_relations_gain", "recall", "contracts", "contracts_domain"),
                               ("loc_contracts_minus_minmax", "loc_paired", "contracts", "minmax"),
                               ("loc_contracts_minus_autoprofile", "loc_paired", "contracts", "autoprofile"),
                               ("loc_contracts_minus_conformance", "loc_paired", "contracts", "conformance"),
                               ("recall_drift_screen_cost", "recall", "contracts_ds", "contracts")]:
            diffs[lab] = boot_mean(wide[(col, x)] - wide[(col, y)])
        out["diffs"] = diffs
        # paging ablation: event detection with CIs, median delay (core faults)
        pg = core[core.method.str.contains(r"\||psi", regex=True)]
        pv = {}
        for v in sorted(pg.method.unique()):
            s = pg[pg.method == v]
            pv[v] = {"detected": boot_mean(s.detected), "delay_median": float(s.delay.median()),
                     "delay_iqr": [float(s.delay.quantile(.25)), float(s.delay.quantile(.75))]}
        out["paging"] = pv
        ev = pg.pivot_table(index=key, columns="method", values="detected")
        pd_ = {}
        for lab, x, y in [("perfeature_minus_pooled_contracts", "contracts|all", "contracts|pooled"),
                          ("contracts_minus_autoprofile_pooled", "contracts|all", "autoprofile|pooled"),
                          ("contracts_minus_autoprofile_perfeature", "contracts|all", "autoprofile|perfeature"),
                          ("burst_minus_hard", "contracts|burst", "contracts|hard")]:
            if x in ev and y in ev:
                pd_[lab] = boot_mean(ev[x] - ev[y])
        out["paging_diffs"] = pd_
        # supplementary faults
        sup = r[r.fault.isin(SUPP)]
        out["supplementary"] = {f: {m: float(sup[(sup.fault == f) & (sup.method == m)].recall.mean())
                                    for m in ["contracts", "autoprofile", "minmax"]} for f in SUPP}
        out["supplementary_detection"] = {f: {v: float(sup[(sup.fault == f) & (sup.method == v)].detected.mean())
                                              for v in ["contracts|hard", "contracts|burst", "autoprofile|pooled"]} for f in SUPP}
        out["fault_recall"] = {f: {m: float(core[(core.fault == f) & (core.method == m)].recall.mean())
                                   for m in ["contracts", "autoprofile"]} for f in CORE}
        # closed loop
        fcl = os.path.join(O, f"{ds}_closedloop.csv")
        if os.path.exists(fcl):
            cl = pd.read_csv(fcl)
            g = {}
            for scen, q in (("systemic", cl[(cl.prevalence == 1.0) & (cl.fault != "none")]),
                            ("sporadic", cl[(cl.prevalence == 0.1) & (cl.fault != "none")]),
                            ("none", cl[cl.fault == "none"])):
                if not len(q):
                    continue
                arms = {}
                for arm in sorted(q.arm.unique()):
                    s = q[q.arm == arm]
                    arms[arm] = {k: float(s[k].mean()) for k in ["recall", "fpr", "precision", "cost10", "cost50"]}
                    if "correct_localization" in s and s.correct_localization.notna().any():
                        arms[arm].update({k: float(s[k].mean()) for k in
                                          ["correct_localization", "complete_localization", "wrong_response",
                                           "pre_onset_response"]})
                        arms[arm]["response_delay_median"] = float(s.response_delay.median())
                g[scen] = arms
            out["closedloop"] = g
        fse = os.path.join(O, f"{ds}_sensitivity.csv")
        if os.path.exists(fse):
            se = pd.read_csv(fse)
            out["sensitivity"] = (se.groupby(["variant", "pattern", "duration"])
                                  .agg(detected=("detected", "mean"), delay=("delay", "median")).round(3)
                                  .reset_index().to_dict("records"))
        fte = os.path.join(O, f"{ds}_temporal.json")
        if os.path.exists(fte):
            out["temporal"] = json.load(open(fte))
        S[ds] = out
    with open(os.path.join(O, "summary_v3.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    print("datasets:", list(S))


if __name__ == "__main__":
    main()
