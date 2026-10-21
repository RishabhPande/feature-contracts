"""LaTeX tables for the post-confirmation analyses (exact conformal calibration,
adaptive calibration, closed-loop extensions, correlated faults, extra seeds,
prevalence/breadth sensitivity, operational summary).

python3 make_tables_ext.py --v3 results_v3 --out ../eswa/tables
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from make_tables import table, per_m, fci

DS = [("cc", "Credit card"), ("ccv", "Credit card + vel."), ("default", "Default-credit"),
      ("ieee", "IEEE-CIS"), ("baf", "Bank Acc.\\ Fraud")]
CORE = ["magnitude", "scale", "encoding", "default_fill", "stale", "misjoin", "clipping", "inconsistency"]


def j(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fm(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def boot(v, B=2000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v):
        return [np.nan] * 3
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [v.mean(), *np.percentile(bs, [2.5, 97.5])]


def poisson_ci(k, n):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
    return lo / n * 1e6, hi / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="results_v3")
    ap.add_argument("--out", default="../eswa/tables")
    a = ap.parse_args()
    V = a.v3

    # 1. exact single-score conformal calibration
    rows = []
    for k, n in DS:
        r = j(os.path.join(V, f"{k}_conformal.json"))
        if not r:
            continue
        al = r["alpha"] * 1e6
        for cal, lab in (("grid", "Two-step grid (paper)"), ("conformal1", "Single-score conformal")):
            t, e = r[f"time|{cal}"], r[f"exchangeable|{cal}"]
            rows.append([n if cal == "grid" else "", lab, f"{al:,.0f}", fm(e["ff_per_m"]), fm(t["ff_per_m"]),
                         f"{t['recall']:.3f}", f"{t['localized']:.3f}"])
    table(os.path.join(a.out, "t_conformal.tex"),
          ["Dataset", "Calibration", "Target / M", "False flags / M, exchangeable control", "False flags / M, time order",
           "Recall", "Localized share"], rows,
          "Exact single-score conformal calibration against the two-step grid rule (post-confirmation analysis). "
          "Exchangeable control: all rows randomly permuted before the same split, so calibration and test requests are "
          "exchangeable and the finite-sample bound applies. Recall and localized share: core faults at 10\\% and 100\\% "
          "prevalence, three seeds, time order.", "tab:conformal", wide=True)

    # 2. adaptive calibration
    rows = []
    lab = {"fixed": "Fixed conformal threshold", "aci_0.01": "Adaptive, $\\eta = 0.01$",
           "aci_0.05": "Adaptive, $\\eta = 0.05$", "aci_0.05_guard": "Adaptive, $\\eta = 0.05$, abrupt-change guard"}
    for k, n in DS:
        r = j(os.path.join(V, f"{k}_adaptive.json"))
        if not r:
            continue
        al = r["alpha"] * 1e6
        for vname, vl in lab.items():
            v = r["variants"][vname]
            rows.append([f"{n} ({al:,.0f})" if vname == "fixed" else "", vl, fm(v["ff_per_m"]),
                         fm(max(v["ff_by_sixth_per_m"])), fm(v["violation_pages_per_m"]),
                         f"{v['recall_systemic_q1']:.2f} / {v['recall_systemic_q4']:.2f}", f"{v['event_detection']:.2f}"])
        r2 = j(os.path.join(V, f"{k}_adaptive_dm10.json"))
        if r2:
            for vname in ("fixed", "aci_0.05"):
                v = r2["variants"][vname]
                rows.append(["", lab[vname] + ", value sets $\\le 10$ values", fm(v["ff_per_m"]), fm(max(v["ff_by_sixth_per_m"])),
                             fm(v["violation_pages_per_m"]), f"{v['recall_systemic_q1']:.2f} / {v['recall_systemic_q4']:.2f}",
                             f"{v['event_detection']:.2f}"])
    table(os.path.join(a.out, "t_adaptive.tex"),
          ["Dataset (target / M)", "Threshold", "False flags / M", "Worst sixth / M", "Violation pages / M",
           "Systemic recall, first / last quarter", "Event detection"], rows,
          "Drift-aware calibration by online quantile tracking (post-confirmation analysis). False flags and pages on the "
          "clean time-ordered test stream (pages: burst-aware test, excluding rate alarms). Systemic recall: share of "
          "corrupted requests flagged in the first and last quarter of a 100\\%-prevalence fault window; a drop shows the "
          "adaptive threshold absorbing a sustained fault. Value sets $\\le 10$ values (IEEE-CIS, exploratory): integer features "
          "with 11 to 50 values get envelopes instead of value sets, so that adaptation can act on them.", "tab:adaptive", wide=True)

    # 3. closed-loop extensions (one table per threshold rule)
    arms = [("unprotected", "No response"), ("mask", "Mask"), ("fallback", "Fallback, no recovery (paper)"),
            ("fallback_recover", "Fallback with recovery"), ("impute_recover", "Rule: impute median"),
            ("review_recover", "Rule: send violations to review"), ("oracle", "Oracle fallback")]
    for mode, ml, fname, label in (("fpr1", "1\\% FPR", "t_clext.tex", "tab:clext"),
                                   ("costopt", "cost-optimal", "t_clext_cost.tex", "tab:clextcost")):
        rows = []
        for k, n in DS:
            fp = os.path.join(V, f"{k}_closedloop_ext.csv")
            if not os.path.exists(fp):
                continue
            c = pd.read_csv(fp)
            q = c[(c.threshold == mode)]
            f = q[(q.fault != "none")]
            nf = q[q.fault == "none"]
            for arm, al in arms:
                g = f[f.arm == arm]
                if not len(g):
                    continue
                h = nf[nf.arm == arm]
                fs = g.false_switches_per_m.mean() if g.false_switches_per_m.notna().any() else np.nan
                nfcost = h.cost50.mean() - nf[nf.arm == "unprotected"].cost50.mean() if len(h) else np.nan
                ts = g.time_switched.mean() if g.time_switched.notna().any() else np.nan
                rd = g.recovery_delay
                rds = "never" if arm == "fallback" else ("--" if rd.isna().all() else f"{rd.median():,.0f}")
                sd = g.switch_in_delay.median() if g.switch_in_delay.notna().any() else np.nan
                rows.append([n if arm == "unprotected" else "", al, f"{g.recall.mean():.3f}", f"{g.cost50.mean():.1f}",
                             "--" if np.isnan(nfcost) else f"{nfcost:+.1f}", fm(fs) if not np.isnan(fs) else "--",
                             "--" if np.isnan(ts) else f"{ts:.2f}", "--" if np.isnan(sd) else f"{sd:,.0f}", rds])
            rows.append("MID")
        if rows and rows[-1] == "MID":
            rows = rows[:-1]
        cap = ("Closed loop with faults that clear, " + ml + " thresholds (post-confirmation analysis). Faults last 20,000 requests "
               "(5,000 on default-credit) and are then removed; all responses are driven by the same pages. Recall and cost over the "
               "whole test stream, averaged over faulted runs; extra cost, no fault: cost difference against no response in the run "
               "without a fault; false switches: switch-ins while no fault is active, per million clean requests; delays in requests "
               "(medians). Recovery: switch back once no paged feature has violated for 1,000 requests.")
        if mode == "costopt":
            cap = ("As Table~\\ref{tab:clext}, with each model's threshold chosen to minimize expected cost (false alarm 1, missed "
                   "fraud 50) on the calibration slice.")
        table(os.path.join(a.out, fname),
              ["Dataset", "Response", "Recall", "Cost / 1,000", "Extra cost, no fault", "False switches / M",
               "Share switched", "Switch-in delay", "Recovery delay"], rows, cap, label, wide=True, size="\\footnotesize")

    # 4. correlated faults
    rows = []
    gl = {"group_scale": "Group scale", "group_stale_lag": "Group lagged cache", "group_shift": "Group offset (+1 s.d.)"}
    for k, n in DS:
        fp = os.path.join(V, f"{k}_runs_group.csv")
        if not os.path.exists(fp):
            continue
        r = pd.read_csv(fp)
        for f, fl in gl.items():
            g = r[r.fault == f]
            rc = g[g.method == "contracts"].recall.mean(); rd = g[g.method == "contracts_domain"].recall.mean()
            ra = g[g.method == "autoprofile"].recall.mean()
            ec = g[g.method == "contracts|burst"].detected.mean(); ea = g[g.method == "autoprofile|perfeature"].detected.mean()
            rows.append([n if f == "group_scale" else "", fl, f"{rc:.2f} / {rd:.2f} / {ra:.2f}", f"{ec:.2f} / {ea:.2f}"])
    table(os.path.join(a.out, "t_group.tex"),
          ["Dataset", "Correlated fault", "Recall: contracts / domain only / auto-profiled", "Event detection: contracts / auto-profiled"],
          rows, "Correlated faults (post-confirmation analysis): one to five features from one lineage group corrupted on the "
          "same requests, so relations within the group are largely preserved. Five seeds, prevalence 1\\%, 10\\%, 100\\%.",
          "tab:group", wide=True)

    # 5. prevalence x breadth sensitivity
    rows = []
    for k, n in DS:
        r = pd.read_csv(os.path.join(V, f"{k}_runs.csv")); r = r[r.fault.isin(CORE)]
        cells = []
        for prev in (0.01, 0.1, 1.0):
            for br in (1, 5):
                g = r[(r.prevalence == prev) & (r.breadth == br)]
                cells.append(f"{g[g.method == 'contracts'].recall.mean():.2f} / {g[g.method == 'contracts|burst'].detected.mean():.2f}")
        rows.append([n] + cells)
    table(os.path.join(a.out, "t_prevbreadth.tex"),
          ["Dataset", "1\\%, 1 feat.", "1\\%, 5 feat.", "10\\%, 1 feat.", "10\\%, 5 feat.", "100\\%, 1 feat.", "100\\%, 5 feat."],
          rows, "Sensitivity to fault prevalence and the number of corrupted features: contracts' per-request recall / event "
          "detection (burst-aware paging with rate checks), core faults, five seeds.", "tab:prevbreadth", wide=True)

    # 6. extra seeds: event detection with 15 seeds, bootstrap over runs
    rows = []
    for k, n in DS:
        f1 = os.path.join(V, f"{k}_runs.csv"); f2 = os.path.join(V, f"{k}_runs_seeds2.csv")
        if not os.path.exists(f2):
            continue
        r = pd.concat([pd.read_csv(f1), pd.read_csv(f2)]); r = r[r.fault.isin(CORE)]
        cells = []
        for m in ("contracts|burst", "contracts|hard", "autoprofile|perfeature", "autoprofile|pooled"):
            g = r[r.method == m]
            b = boot(g.detected.values)
            cells.append(f"{b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]")
        p = r.pivot_table(index=["fault", "prevalence", "breadth", "seed"], columns="method", values="detected")
        dd = boot((p["contracts|burst"] - p["autoprofile|perfeature"]).values)
        cells.append(fci(dd, 2))
        rows.append([n, str(r.seed.nunique())] + cells)
    table(os.path.join(a.out, "t_seeds.tex"),
          ["Dataset", "Seeds", "Contracts, burst-aware", "Contracts, tier", "Auto-profiled, per feature", "Auto-profiled, pooled",
           "Burst-aware $-$ auto-profiled (paired)"], rows,
          "Event detection on core faults with fifteen fault seeds (the five of the main analysis plus ten new ones), with 95\\% "
          "bootstrap intervals over runs.", "tab:seeds", wide=True)

    # 7. rare false pages with exact Poisson intervals
    S = j(os.path.join(V, "summary_v3.json"))
    rows = []
    for k, n in DS:
        cj = j(os.path.join(V, f"{k}_clean.json"))
        if not cj:
            continue
        nt = cj["n_test"]; fp = cj["false_pages_per_100k"]
        cells = []
        for v in ("contracts|rates", "contracts|burst"):
            cnt = int(round(fp[v] * nt / 1e5)) if v == "contracts|rates" else int(round((fp[v] - fp["contracts|rates"]) * nt / 1e5))
            lo, hi = poisson_ci(cnt, nt)
            cells.append(f"{cnt:,} ({fm(lo)}--{fm(hi)})")
        rows.append([n, f"{nt:,}"] + cells)
    table(os.path.join(a.out, "t_rarepages.tex"),
          ["Dataset", "Clean test requests", "Rate-check alarms (95\\% interval / M)", "Burst-aware violation pages (95\\% interval / M)"],
          rows, "False pages on the clean test stream as counts with exact Poisson 95\\% intervals per million requests. Counts "
          "are rare and often clustered in one window, so intervals that assume independent events understate uncertainty.",
          "tab:rarepages")

    # 8. operational summary
    O = j(os.path.join(V, "overhead.json")) or {}
    rows = []
    for k, n in DS:
        s = S[k]; cl = s.get("closedloop", {}).get("systemic", {})
        loc_c = s["methods"]["contracts"]["loc_paired"][0]; loc_a = s["methods"]["autoprofile"]["loc_paired"][0]
        pc = s["paging"]["contracts|burst"]; pa = s["paging"]["autoprofile|perfeature"]
        cj = j(os.path.join(V, f"{k}_clean.json")); fpg = s["operating"]["contracts|burst"]["pages"][0] * 10
        if cl:
            cc = cl["contract_mask|hard"]
            gain = cl["unprotected"]["cost50"] - cc["cost50"]
            clc = f"{cc['correct_localization']:.2f}"
            gains = f"{gain:+.1f}"
        else:
            clc, gains = "--", "--"
        ov = O.get(k, {})
        rows.append([n, f"{loc_c:.2f} / {loc_a:.2f}", f"{pc['detected'][0]:.2f} / {pa['detected'][0]:.2f}",
                     f"{pc['delay_median']:.0f} / {pa['delay_median']:.0f}", fm(fpg), clc, gains,
                     f"{ov['batch_us_per_request']:.0f} / {ov['fit_seconds']:.0f}" if ov else "--"])
    table(os.path.join(a.out, "t_opsummary.tex"),
          ["Dataset", "Localized share (C / AP)", "Event detection (C / AP)", "Median delay (C / AP)", "False pages / M (C)",
           "Systemic: first page correct", "Systemic: cost saved / 1,000", "Check $\\mu$s / request; fit s"], rows,
          "Operational outcomes in one place. C: contracts (burst-aware paging with rate checks); AP: auto-profiled rule with "
          "per-feature paging. Cost saved: unprotected minus contract-driven response under systemic faults (positive is better). "
          "Check time: reference Python implementation on one core, amortized over a batch of 20,000 requests; fit: seconds to learn "
          "and calibrate a contract.",
          "tab:opsummary", wide=True)
    print("extension tables written to", a.out)


if __name__ == "__main__":
    main()
