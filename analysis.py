"""Tables from results/: python3 analysis.py [--out results] [--datasets cc ccv default]

Prints and writes results/tables.md: Table 5 (recall, false flags), Table 7
(budget vs realized), localization (unpaired and paired), RQ2 cell counts,
Wilcoxon + Holm, bootstrap intervals (Table 11), event detection (6.3).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DETECTORS = ["schema", "minmax", "autoprofile", "robust_z", "iforest", "pca_recon",
             "conformance", "contracts_domain", "contracts"]
PAPER_FAULTS = ["magnitude", "scale", "encoding", "default_fill", "stale", "misjoin",
                "clipping", "inconsistency"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def boot(diff, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    m = [rng.choice(diff, len(diff)).mean() for _ in range(B)]
    return diff.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="*", default=["cc", "ccv", "default"])
    ap.add_argument("--all-faults", action="store_true", help="include nan_fill")
    a = ap.parse_args()
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    for ds in a.datasets:
        fr = os.path.join(a.out, f"{ds}_runs.csv")
        if not os.path.exists(fr):
            continue
        r = pd.read_csv(fr)
        if not a.all_faults:
            r = r[r.fault.isin(PAPER_FAULTS)]
        c = json.load(open(os.path.join(a.out, f"{ds}_clean.json")))
        key = ["fault", "prevalence", "breadth", "seed"]
        det = r[r.method.isin(DETECTORS)].copy()
        det["loc_unpaired"] = det.loc1 / det.flagged.replace(0, np.nan)
        det["loc_paired"] = det.loc1 / det.n_corrupted
        P(f"\n## {ds}  (runs per method: {det[det.method=='contracts'].shape[0]}, "
          f"kappa={c['kappa']}, t={c['t']}, alpha={c['alpha']})\n")
        P("| Method | Mean recall | False flags /100k | Top-1 loc (unpaired) | Correctly localized share |")
        P("| --- | --- | --- | --- | --- |")
        for m in DETECTORS:
            s = det[det.method == m]
            P(f"| {m} | {s.recall.mean():.3f} | {c['false_flags_per_100k'][m]:.0f} | "
              f"{s.loc_unpaired.mean():.3f} | {s.loc_paired.mean():.3f} |")

        # by fault type
        P("\nRecall by fault type (contracts / autoprofile / minmax):")
        g = det.pivot_table(index="fault", columns="method", values="recall", aggfunc="mean")
        for f in g.index:
            P(f"- {f}: {g.loc[f,'contracts']:.2f} / {g.loc[f,'autoprofile']:.2f} / {g.loc[f,'minmax']:.2f}")

        # RQ2 and Wilcoxon over cells
        cell = det.groupby(["method", "fault", "prevalence", "breadth"]).recall.mean().unstack(0)
        dlt = cell["contracts"] - cell["contracts_domain"]
        P(f"\nRQ2 relation checks vs domain only: mean +{dlt.mean():.3f}; "
          f"wins {(dlt>1e-9).sum()}, losses {(dlt<-1e-9).sum()}, ties {(dlt.abs()<=1e-9).sum()} over {dlt.notna().sum()} cells")
        ps, names, diffs = [], [], []
        for b in ["minmax", "autoprofile", "robust_z", "iforest", "pca_recon", "conformance"]:
            dd = (cell["contracts"] - cell[b]).dropna()
            try:
                p = wilcoxon(dd).pvalue if (dd != 0).any() else 1.0
            except ValueError:
                p = 1.0
            ps.append(p); names.append(b); diffs.append(dd.mean())
        P("Wilcoxon over cells, contracts minus baseline (Holm-adjusted p): " +
          "; ".join(f"{n} {d:+.3f} (p={q:.3g})" for n, d, q in zip(names, diffs, holm(ps))))

        # bootstrap paired over runs
        wide = det.pivot_table(index=key, columns="method", values=["recall", "loc_paired"])
        P("\nBootstrap 95% intervals over runs:")
        for lab, col, b in [("Recall, contracts - autoprofile", "recall", "autoprofile"),
                            ("Recall, contracts - conformance", "recall", "conformance"),
                            ("Correctly localized, contracts - minmax", "loc_paired", "minmax"),
                            ("Correctly localized, contracts - conformance", "loc_paired", "conformance")]:
            m_, lo, hi = boot(wide[(col, "contracts")] - wide[(col, b)])
            P(f"- {lab}: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")

        # paging
        pg = r[r.method.str.contains(r"\||psi", regex=True)]
        P("\n| Paging variant | Events detected | Median delay | False pages /100k |")
        P("| --- | --- | --- | --- |")
        for m in sorted(pg.method.unique()):
            s = pg[pg.method == m]
            P(f"| {m} | {s.detected.mean():.2f} | {s.delay.median():.0f} | "
              f"{c['false_pages_per_100k'].get(m, float('nan')):.1f} |")
        ev = pg[pg.method.isin(["contracts|all", "autoprofile|pooled"])].pivot_table(
            index=key, columns="method", values="detected")
        if ev.shape[1] == 2:
            m_, lo, hi = boot(ev["contracts|all"] - ev["autoprofile|pooled"])
            P(f"\nEvent detection, contracts(all) - autoprofile: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        s = pg[pg.method == "contracts|all"]
        P("Detection by prevalence (contracts|all): " + ", ".join(
            f"{p:g}: {v:.2f}" for p, v in s.groupby("prevalence").detected.mean().items()))
        P("Detection by fault (contracts|all): " + ", ".join(
            f"{f}: {v:.2f}" for f, v in s.groupby("fault").detected.mean().items()))
        P(f"\nCalibration: budget {c['alpha']*1e5:.0f}/100k, realized contracts "
          f"{c['false_flags_per_100k']['contracts']:.0f} (ratio {c['false_flags_per_100k']['contracts']/(c['alpha']*1e5):.2f}); "
          f"flag bursts {c['flag_bursts']['contracts']}; top features {c['top_false_flag_features']}")

    with open(os.path.join(a.out, "tables.md"), "w") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
