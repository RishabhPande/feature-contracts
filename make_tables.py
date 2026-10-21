"""Generate every LaTeX results table in the manuscript from result files.

python3 make_tables.py --v3 results_v3 --v2 results --out ../eswa/tables
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DS = [("cc", "Credit card"), ("ccv", "Credit card + vel."), ("default", "Default-credit"),
      ("ieee", "IEEE-CIS"), ("baf", "Bank Acc.\\ Fraud")]
NAMES = {"minmax": "Min-max rule", "autoprofile": "Auto-profiled", "robust_z": "Robust z-score",
         "iforest": "Isolation forest", "pca_recon": "PCA reconstruction", "conformance": "Conformance",
         "contracts_domain": "Contracts, domain only", "contracts": "Feature contracts",
         "contracts_ds": "Contracts + drift screen"}


def f3(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fci(v, dig=3):
    m, lo, hi = v
    if np.isnan(m):
        return "--"
    fmt = lambda x: f"{x:+.{dig}f}"
    return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]"


def per_m(x):
    """per-100k -> per-million, compact."""
    if x is None or np.isnan(x):
        return "--"
    v = x * 10
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def table(path, cols, rows, caption, label, wide=False, size="\\small"):
    env = "table"  # single-column manuscript
    spec = "@{}l" + "c" * (len(cols) - 1) + "@{}"
    lines = [f"\\begin{{{env}}}[H]", "\\centering" + size, f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\resizebox{\\textwidth}{!}{" if wide else "\\resizebox{\\textwidth}{!}{",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", f"\\end{{{env}}}"]
    open(path, "w").write("\n".join(lines) + "\n")


def matched_recall(sweep_csv, target_ff):
    """Recall of contracts interpolated (log false flags) at a target false-flag rate."""
    s = pd.read_csv(sweep_csv)
    g = s[s.method == "contracts"].groupby("alpha").agg(r=("recall", "mean"), ff=("ff", "first")).reset_index()
    g = g.sort_values("ff")
    x = np.log10(np.maximum(g.ff.values, 0.1)); y = g.r.values
    t = np.log10(max(target_ff, 0.1))
    if t < x.min() or t > x.max():
        return np.nan
    return float(np.interp(t, x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="results_v3")
    ap.add_argument("--v2", default="results")
    ap.add_argument("--out", default="../eswa/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(os.path.join(a.v3, "summary_v3.json")))
    have = [(k, n) for k, n in DS if k in S]
    hdr = ["Method"] + [n for _, n in have]

    # T: recall and realized false flags per million
    rows = []
    for m in ["minmax", "autoprofile", "robust_z", "iforest", "pca_recon", "conformance",
              "contracts_domain", "contracts", "contracts_ds"]:
        rows.append([NAMES[m]] + [f"{S[k]['methods'][m]['recall'][0]:.3f} ({per_m(S[k]['methods'][m]['ff'])})" for k, _ in have])
    table(os.path.join(a.v3, "..", a.out, "t_main.tex") if False else os.path.join(a.out, "t_main.tex"), hdr, rows,
          "Mean recall over core fault runs, with realized false flags per million clean requests in brackets. "
          "Rates differ by method; Table~\\ref{tab:matched} compares at matched rates.", "tab:main", wide=True)

    # T: matched-rate comparison, with paired bootstrap over runs
    rows = []
    rng = np.random.default_rng(0)
    for k, n in have:
        sw = os.path.join(a.v3, f"{k}_budget_sweep.csv")
        if not os.path.exists(sw) and k in ("cc", "ccv", "default"):
            sw = os.path.join(a.v2, f"{k}_budget_sweep.csv")  # no missing values: v2 and v3 contracts flag identically
        if not os.path.exists(sw):
            continue
        s = pd.read_csv(sw)
        key = ["fault", "prevalence", "breadth", "seed"]
        apr = s[s.method == "autoprofile"].set_index(key).recall
        ff_ap = float(s[s.method == "autoprofile"].ff.iloc[0])
        con = s[s.method == "contracts"]
        ffa = con.groupby("alpha").ff.first().sort_values()
        piv = con.pivot_table(index=key, columns="alpha", values="recall").reindex(apr.index)[ffa.index]
        x = np.log10(np.maximum(ffa.values, 0.1)); t = np.log10(max(ff_ap, 0.1))
        if t < x.min() or t > x.max():
            rows.append([n, per_m(ff_ap), f"{apr.mean():.3f}", "outside range", "--"]); continue
        P = piv.values; R = apr.values
        def stat(idx):
            return float(np.interp(t, x, np.nanmean(P[idx], axis=0))) - float(np.nanmean(R[idx]))
        d0 = stat(np.arange(len(R)))
        bs = [stat(rng.integers(0, len(R), len(R))) for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append([n, per_m(ff_ap), f"{R.mean():.3f}", f"{R.mean() + d0:.3f}", fci([d0, lo, hi])])
    table(os.path.join(a.out, "t_matched.tex"),
          ["Dataset", "Auto-profiled false flags / M", "Auto-profiled recall", "Contracts recall at same rate", "Difference [95\\% CI]"],
          rows, "Detection at matched realized false-flag rates. Contracts' recall is interpolated (log scale) on the budget sweep at the "
          "auto-profiled rule's realized rate; intervals are paired bootstrap intervals over runs. The sweep is a separate run "
          "(development configurations: an earlier run of the same benchmark, with identical contracts because these data have no missing "
          "values; IEEE-CIS and Bank Account Fraud: 10\\% prevalence, three seeds), so recall can differ slightly from Table~\\ref{tab:main}.", "tab:matched")

    # T: localization and detection differences with CIs
    rows = []
    for lab, key in [("Recall: contracts $-$ auto-profiled", "recall_contracts_minus_autoprofile"),
                     ("Recall: relation checks (full $-$ domain only)", "recall_relations_gain"),
                     ("Localized share: contracts $-$ min-max", "loc_contracts_minus_minmax"),
                     ("Localized share: contracts $-$ auto-profiled", "loc_contracts_minus_autoprofile"),
                     ("Localized share: contracts $-$ conformance", "loc_contracts_minus_conformance"),
                     ("Recall: drift screen $-$ contracts", "recall_drift_screen_cost")]:
        rows.append([lab] + [fci(S[k]["diffs"][key]) for k, _ in have])
    table(os.path.join(a.out, "t_diffs.tex"), ["Paired difference (95\\% bootstrap interval)"] + [n for _, n in have], rows,
          "Paired differences over the same fault runs. The localized share is the fraction of all corrupted requests that are flagged "
          "and whose highest-evidence feature is a changed one.", "tab:diffs", wide=True)

    # T: paging ablation and operating points
    rows = []
    for v, lab in [("contracts|pooled", "Contracts, one pooled stream (no rates)"),
                   ("contracts|rates", "Contracts, rate checks only"),
                   ("contracts|all", "Contracts, per feature + rates"),
                   ("contracts|hard", "\\quad + severity tier (default)"),
                   ("contracts|burst", "\\quad burst-aware instead of tier"),
                   ("contracts_ds|hard", "\\quad + drift screen"),
                   ("autoprofile|pooled", "Auto-profiled, pooled"),
                   ("autoprofile|perfeature", "Auto-profiled, per feature"),
                   ("conformance|pooled", "Conformance, pooled"),
                   ("psi", "Population stability index")]:
        cells = []
        for k, _ in have:
            p = S[k]["paging"].get(v)
            o = S[k]["operating"].get(v, {}).get("pages", [np.nan] * 3)
            if p is None:
                cells.append("--"); continue
            dd = p["detected"][0]; dl = p["delay_median"]
            cells.append(f"{dd:.2f} / {dl:.0f} / {per_m(o[0])}" if not np.isnan(dl) else f"{dd:.2f} / -- / {per_m(o[0])}")
        rows.append([lab] + cells)
    table(os.path.join(a.out, "t_paging.tex"), ["Paging variant"] + [n for _, n in have], rows,
          "Event detection / median delay in requests / false pages per million clean requests. Core faults; rate-check alarms are "
          "included in every contracts variant except the pooled one.", "tab:paging", wide=True)

    rows = []
    for lab, key in [("Per feature + rates $-$ pooled", "perfeature_minus_pooled_contracts"),
                     ("Contracts $-$ auto-profiled (pooled)", "contracts_minus_autoprofile_pooled"),
                     ("Contracts $-$ auto-profiled (per feature)", "contracts_minus_autoprofile_perfeature"),
                     ("Burst-aware $-$ severity tier", "burst_minus_hard")]:
        rows.append([lab] + [fci(S[k]["paging_diffs"].get(key, [np.nan] * 3)) for k, _ in have])
    table(os.path.join(a.out, "t_pagingdiff.tex"), ["Event detection difference"] + [n for _, n in have], rows,
          "Paired differences in event detection (95\\% bootstrap intervals).", "tab:pagingdiff", wide=True)

    # T: operating points with intervals; pages split into rate-check alarms and violation-test pages
    alpha_m = {k: S[k]["alpha"] * 1e6 for k, _ in have}
    rows = [["Target $\\alpha$, flags"] + [f"{alpha_m[k]:,.0f}" for k, _ in have]]
    for m, lab in [("autoprofile", "Auto-profiled, flags"), ("contracts", "Contracts, flags")]:
        rows.append([lab] + [f"{per_m(S[k]['operating'][m]['flags'][0])} [{per_m(S[k]['operating'][m]['flags'][1])}, "
                             f"{per_m(S[k]['operating'][m]['flags'][2])}]" for k, _ in have])
    rows.append("MID")
    cleanj = {k: json.load(open(os.path.join(a.v3, f"{k}_clean.json"))) for k, _ in have}
    def cnt(k, v):
        n = S[k]["n_test"]; r = cleanj[k]["false_pages_per_100k"].get(v, np.nan) * n / 1e5
        return n, r
    def cell(k, per100k):
        n = S[k]["n_test"]; c = per100k * n / 1e5
        return f"{per100k * 10:,.0f} ({c:,.0f})" if per100k * 10 >= 10 else f"{per100k * 10:.1f} ({c:,.0f})"
    fp = lambda k, v: cleanj[k]["false_pages_per_100k"].get(v, np.nan)
    rows.append(["Contracts, rate-check alarms"] + [cell(k, fp(k, "contracts|rates")) for k, _ in have])
    rows.append(["Contracts (tier), violation-test pages"] + [cell(k, fp(k, "contracts|hard") - fp(k, "contracts|rates")) for k, _ in have])
    rows.append(["Contracts (burst-aware), violation-test pages"] + [cell(k, fp(k, "contracts|burst") - fp(k, "contracts|rates")) for k, _ in have])
    rows.append(["Auto-profiled (pooled), pages"] + [cell(k, fp(k, "autoprofile|pooled")) for k, _ in have])
    table(os.path.join(a.out, "t_operating.tex"), ["Per million clean requests"] + [n for _, n in have], rows,
          "Operating points on the clean test stream, per million requests. Flags: 95\\% block-bootstrap intervals over blocks of "
          "1,000 consecutive requests. Pages: rate per million (number of pages). The nominal bound of Proposition~\\ref{prop:arl} for "
          "violation-test pages is $1/A = 0.1$ per million ($A = 10^7$); rate-check alarms are not covered by it.", "tab:operating", wide=True)

    # T: event detection by fault type
    lab_ev = {"magnitude": "Magnitude", "scale": "Scale", "encoding": "Encoding", "default_fill": "Default fill",
              "inconsistency": "Inconsistency", "stale": "Stale (frozen)", "misjoin": "Misjoin (shuffled)", "clipping": "Clipping",
              "stale_lag": "Stale (lagged cache)", "misjoin_ref": "Misjoin (other record)", "nan_fill": "Missing fill"}
    ev = {}
    for k, _ in have:
        r = pd.read_csv(os.path.join(a.v3, f"{k}_runs.csv"))
        ev[k] = r[r.method.isin(["contracts|burst", "autoprofile|perfeature", "contracts|rates"])].groupby(["fault", "method"]).detected.mean().unstack()
    rows = []
    for f in ["magnitude", "scale", "encoding", "default_fill", "inconsistency", "stale", "misjoin", "clipping", "MID",
              "stale_lag", "misjoin_ref", "nan_fill"]:
        if f == "MID":
            rows.append("MID"); continue
        rows.append([lab_ev[f]] + [f"{ev[k].loc[f, 'contracts|burst']:.2f} / {ev[k].loc[f, 'contracts|rates']:.2f} / "
                                   f"{ev[k].loc[f, 'autoprofile|perfeature']:.2f}" for k, _ in have])
    table(os.path.join(a.out, "t_events.tex"), ["Fault"] + [n for _, n in have], rows,
          "Event detection by fault type: contracts (burst-aware, with rate checks) / rate checks alone / auto-profiled rule with "
          "per-feature paging. An event is detected when a page at or after onset names a corrupted feature and no page is raised on the "
          "matched control. Below the line: supplementary operators.", "tab:events", wide=True)

    # T: fault types (core + supplementary)
    order = ["magnitude", "scale", "encoding", "default_fill", "inconsistency", "stale", "misjoin", "clipping"]
    lab = {"magnitude": "Magnitude", "scale": "Scale", "encoding": "Encoding", "default_fill": "Default fill",
           "inconsistency": "Inconsistency", "stale": "Stale (frozen)", "misjoin": "Misjoin (shuffled)", "clipping": "Clipping",
           "stale_lag": "Stale (lagged cache)", "misjoin_ref": "Misjoin (other record)", "nan_fill": "Missing fill"}
    rows = []
    for f in order:
        rows.append([lab[f]] + [f"{S[k]['fault_recall'][f]['contracts']:.2f} / {S[k]['fault_recall'][f]['autoprofile']:.2f}" for k, _ in have])
    rows.append("MID")
    for f in ["stale_lag", "misjoin_ref", "nan_fill"]:
        rows.append([lab[f]] + [f"{S[k]['supplementary'][f]['contracts']:.2f} / {S[k]['supplementary'][f]['autoprofile']:.2f}" for k, _ in have])
    table(os.path.join(a.out, "t_fault.tex"), ["Fault"] + [n for _, n in have], rows,
          "Per-request recall by fault type, contracts / auto-profiled. Below the line: supplementary operators.", "tab:fault", wide=True)

    # T: closed loop
    rows = []
    nruns = {}
    arms = [("unprotected", "Unprotected"), ("mask", "Mask (clean threshold)"),
            ("contract_mask|hard", "Contract-driven response"), ("oracle", "Oracle fallback"), ("clean", "No fault")]
    for k, n in have:
        cl = S[k].get("closedloop")
        if not cl or "systemic" not in cl:
            continue
        sy = cl["systemic"]
        for arm, al in arms:
            if arm not in sy:
                continue
            v = sy[arm]
            extra = ""
            if arm == "contract_mask|hard":
                q = pd.read_csv(os.path.join(a.v3, f"{k}_closedloop.csv"))
                q = q[(q.arm == arm) & (q.prevalence == 1.0) & (q.fault != "none")]
                timely = float(((q.correct_localization == 1) & (q.response_delay < 10000)).mean())
                dl = v["response_delay_median"]
                dls = "before onset" if (dl is None or np.isnan(dl)) and v.get("pre_onset_response", 0) >= 0.5 else (
                    "--" if dl is None or np.isnan(dl) else f"{dl:.0f}")
                extra = f"{v['correct_localization']:.2f} ({timely:.2f}) / {v['wrong_response']:.2f} / {dls}"
                nruns[k] = len(q)
            rows.append([n if arm == "unprotected" else "", al, f"{v['recall']:.3f}", f"{v['fpr']:.3f}", f"{v['precision']:.3f}",
                         f"{v['cost50']:.1f}", extra])
        rows.append("MID")
    if rows and rows[-1] == "MID":
        rows = rows[:-1]
    table(os.path.join(a.out, "t_closedloop.tex"),
          ["Dataset", "Response", "Recall", "FPR", "Precision", "Cost / 1,000 req.", "Correct loc.\\ / wrong resp.\\ / delay"], rows,
          "Closed loop under systemic faults (every request after onset): the first page switches scoring to a fallback model trained "
          "without the paged features. The decision threshold is fixed at a 1\\% false positive rate on clean calibration traffic. "
          "Cost counts a false alarm as 1 and a missed fraud as 50, per 1,000 requests, and is used only to compare responses. "
          "Correct loc.: share of runs whose first page names a corrupted feature (in brackets: and arrives within 10,000 requests of "
          "onset; later pages are mostly the clean-traffic rate alarm of Section~\\ref{sec:res_paging}); wrong resp.: share whose first "
          "switch names none; delay: median requests from onset to the switch. Runs: seven fault types (five on IEEE-CIS) on the top-1 "
          "and top-5 important features (plus one unimportant feature on the development configurations), with three seeds on the development "
          "configurations and two on IEEE-CIS and Bank Account Fraud.", "tab:closedloop", wide=True)

    rows = []
    for k, n in have:
        cl = S[k].get("closedloop")
        if not cl or "sporadic" not in cl:
            continue
        for scen, sl in (("sporadic", "Sporadic (10\\%)"), ("none", "No fault")):
            if scen not in cl:
                continue
            u = cl[scen]["unprotected"]; c = cl[scen]["contract_mask|hard"]; o = cl[scen].get("oracle")
            dl = c.get("response_delay_median", np.nan)
            rows.append([n, sl, f"{u['recall']:.3f} / {c['recall']:.3f}", f"{u['cost50']:.1f}", f"{c['cost50']:.1f}",
                         f"{o['cost50']:.1f}" if o else "--", f"{c.get('wrong_response', np.nan):.2f}",
                         "--" if dl is None or np.isnan(dl) else f"{dl:,.0f}"])
    table(os.path.join(a.out, "t_closedloop2.tex"),
          ["Dataset", "Scenario", "Recall, unprot.\\ / contract", "Cost, unprot.", "Cost, contract", "Cost, oracle fallback",
           "Wrong-response rate", "Median delay to switch"], rows,
          "Responding when the fault is sporadic (10\\% of requests in the window) or absent. Wrong-response rate: share of runs whose "
          "first switch named no corrupted feature (in no-fault runs, any switch). Delay: requests from fault onset (or window start) to the "
          "switch; delays near the end of the stream are the late rate alarm described in Section~\\ref{sec:res_paging}, not responses to "
          "the injected fault. Cost per 1,000 requests, false alarm 1, missed fraud 50.", "tab:closedloop2", wide=True)

    # T: sensitivity (pooled over datasets)
    frames = []
    for k, n in have:
        fp = os.path.join(a.v3, f"{k}_sensitivity.csv")
        if os.path.exists(fp):
            frames.append(pd.read_csv(fp))
    if frames:
        se = pd.concat(frames)
        g = se.groupby(["variant", "pattern", "duration"]).agg(det=("detected", "mean"), dl=("delay", "median")).reset_index()
        rows = []
        for v, vl in [("contracts|hard", "Contracts (tier)"), ("contracts|burst", "Contracts (burst-aware)"),
                      ("autoprofile|pooled", "Auto-profiled, pooled")]:
            for p in ["continuous", "intermittent"]:
                q = g[(g.variant == v) & (g.pattern == p)].set_index("duration")
                rows.append([vl, p] + [f"{q.loc[d, 'det']:.2f} / {q.loc[d, 'dl']:.0f}" if d in q.index and not np.isnan(q.loc[d, 'dl'])
                                       else (f"{q.loc[d, 'det']:.2f} / --" if d in q.index else "--") for d in (500, 2000, 10000)])
        table(os.path.join(a.out, "t_sens.tex"), ["Variant", "Pattern", "500 requests", "2,000 requests", "10,000 requests"], rows,
              "Sensitivity to fault duration and persistence: event detection / median delay, single-feature faults (magnitude, scale, "
              "default fill, lagged stale), five seeds per dataset (three on IEEE-CIS), pooled over datasets. Intermittent faults are "
              "on for 50 of every 500 requests.", "tab:sens")

    # T: temporal windows
    rows = []
    for k, n in have:
        fp = os.path.join(a.v3, f"{k}_temporal.json")
        if not os.path.exists(fp):
            continue
        for r in json.load(open(fp)):
            rows.append([n, f"{r['origin']:.0%}".replace("%", "\\%"), per_m(r["contracts_flags"]), per_m(r["contracts_ds_flags"]),
                         per_m(r["autoprofile_flags"]), per_m(r["contracts_pages_hard"]), per_m(r["contracts_pages_burst"]),
                         str(r["exempt"])])
    table(os.path.join(a.out, "t_temporal.tex"),
          ["Dataset", "Origin", "Contracts flags / M", "+ drift screen", "Auto-profiled", "Pages (tier) / M", "Pages (burst) / M", "Exempt features"],
          rows, "Rolling-origin evaluation: contracts learned on the 30\\% of the stream before each origin and evaluated on the next 15\\%.",
          "tab:temporal")
    print("tables written to", a.out)


if __name__ == "__main__":
    main()


def extra_tables(v3, v2, out):
    # simulation table
    fp = os.path.join(v3, "paging_sim.json")
    if os.path.exists(fp):
        sim = json.load(open(fp))
        rows = []
        for L, r in sim.items():
            rows.append([L, f"{r['false_pages_per_100k_sr']:.1f}", f"{r['false_pages_per_100k_burst']:.1f}",
                         f"{r['sporadic_median_delay_sr']:.0f} / {r['sporadic_median_delay_burst']:.0f}",
                         f"{r['stuck_median_delay_sr']:.0f} / {r['stuck_median_delay_burst']:.0f}",
                         f"{r['sporadic_detected_sr']:.2f} / {r['sporadic_detected_burst']:.2f}",
                         f"{r['stuck_detected_sr']:.2f} / {r['stuck_detected_burst']:.2f}"])
        table(os.path.join(out, "t_sim.tex"),
              ["Mean burst length", "False pages / 100k, per-violation", "False pages / 100k, burst-aware",
               "Sporadic delay (per-violation / burst)", "Stuck delay (per-violation / burst)", "Detected, sporadic", "Detected, stuck"],
              rows, "Controlled paging simulation (bound: 50 false pages per 100,000). Delays are median requests from fault onset to a page "
              "on the faulty feature; detection is the share of 50 replicates with such a page within 2,000 requests.", "tab:sim")
    # latency table
    for src in (v3, v2):
        fp = os.path.join(src, "latency.json")
        if os.path.exists(fp):
            lat = json.load(open(fp))
            rows = [[f"{int(d):,}", f"{r['check']['p50_us']:.0f} / {r['check']['p99_us']:.0f}",
                     f"{r['check_plus_paging']['p50_us']:.0f} / {r['check_plus_paging']['p99_us']:.0f}",
                     f"{r['contract_kb']:,.0f} KB"] for d, r in lat.items()]
            table(os.path.join(out, "t_latency.tex"), ["Features", "Check (median / p99, $\\mu$s)",
                  "Check + paging update", "Contract size"], rows,
                  "Single-request cost on one core (controlled benchmark, synthetic contracts).", "tab:cost")
            break


if __name__ == "__main__":
    import sys
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    extra_tables(args.get("--v3", "results_v3"), args.get("--v2", "results"), args.get("--out", "../eswa/tables"))
