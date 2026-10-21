"""Generate manuscript figures from result files.

python3 make_figs.py --v3 results_v3 --v2 results --out ../eswa/figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

C = {"contracts": "#2a78d6", "contracts_domain": "#1baf7a", "conformance": "#eb6834",
     "robust_z": "#eda100", "ref": "#6b6b6b"}
LAB = {"contracts": "Contracts", "contracts_domain": "Contracts (domain only)",
       "conformance": "Conformance", "robust_z": "Robust z"}
NAME = {"cc": "Credit card", "ccv": "Credit card + velocity", "default": "Default-credit",
        "ieee": "IEEE-CIS", "baf": "Bank Account Fraud"}
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", default="results_v3")
    ap.add_argument("--v2", default="results")
    ap.add_argument("--out", default="../eswa/figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # budget sweep (v3 files preferred; public configurations have no missing values, so v2 sweeps are identical)
    files = []
    for ds in ["cc", "ccv", "default", "ieee", "baf"]:
        for src in (a.v3, a.v2):
            f = os.path.join(src, f"{ds}_budget_sweep.csv")
            if os.path.exists(f):
                files.append((ds, f)); break
    fig, axes = plt.subplots(1, len(files), figsize=(1.75 * len(files) + 0.5, 1.9))
    for ax, (ds, f) in zip(np.atleast_1d(axes), files):
        s = pd.read_csv(f)
        for m in ["contracts", "contracts_domain", "conformance", "robust_z"]:
            g = s[s.method == m].groupby("alpha").agg(r=("recall", "mean"), ff=("ff", "first")).reset_index().sort_values("ff")
            ax.plot(np.maximum(g.ff, 1), g.r, "-o", color=C[m], lw=1.3, ms=2.6, label=LAB[m])
        for m, mk in [("minmax", "s"), ("autoprofile", "D")]:
            g = s[s.method == m]
            ax.plot(max(g.ff.iloc[0], 1), g.recall.mean(), mk, color="black", ms=3.5, mfc="white", mew=1,
                    label={"minmax": "Min-max", "autoprofile": "Auto-profiled"}[m])
        ax.set_xscale("log"); ax.set_title(NAME[ds], fontsize=7.5); ax.set_xlabel("False flags / 100k")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    np.atleast_1d(axes)[0].set_ylabel("Mean recall")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.1), fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "budget.pdf"), bbox_inches="tight"); plt.close(fig)

    # IEEE drift with refresh
    for src in (a.v3, a.v2):
        f = os.path.join(src, "ieee_refresh.json")
        if os.path.exists(f):
            rf = json.load(open(f)); break
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(1, 7)
    for k, lab, col in [("period_0", "No refresh", C["contracts"]), ("period_50000", "Refresh every 50k", C["conformance"]),
                        ("period_20000", "Refresh every 20k", C["contracts_domain"])]:
        ax.plot(x, rf[k]["false_flags_by_sixth"], "-o", color=col, lw=1.3, ms=3, label=lab)
    ax.axhline(100, color=C["ref"], lw=1, ls="--"); ax.text(6.05, 100, "target", va="center", fontsize=7, color=C["ref"])
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{i}/6" for i in x])
    ax.set_xlabel("Sixth of the IEEE-CIS test stream (time order)"); ax.set_ylabel("False flags / 100k")
    ax.legend(fontsize=6.5, loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "drift.pdf"), bbox_inches="tight"); plt.close(fig)

    # paging simulation: false pages and stuck-fault delay
    sim = json.load(open(os.path.join(a.v3, "paging_sim.json")))
    L = [int(k) for k in sim]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.0))
    a1.plot(L, [sim[str(k)]["false_pages_per_100k_sr"] for k in L], "-o", color=C["contracts"], lw=1.3, ms=3, label="Per-violation test")
    a1.plot(L, [sim[str(k)]["false_pages_per_100k_burst"] for k in L], "-o", color=C["conformance"], lw=1.3, ms=3, label="Burst-aware test")
    a1.axhline(50, color=C["ref"], lw=1, ls="--"); a1.text(1.05, 60, "bound (50)", fontsize=7, color=C["ref"])
    a1.set_xscale("log"); a1.set_xlabel("Mean burst length of clean violations"); a1.set_ylabel("False pages / 100k")
    a1.legend(fontsize=6.5)
    a2.plot(L, [sim[str(k)]["stuck_median_delay_sr"] for k in L], "-o", color=C["contracts"], lw=1.3, ms=3)
    a2.plot(L, [sim[str(k)]["stuck_median_delay_burst"] or 0 for k in L], "-o", color=C["conformance"], lw=1.3, ms=3)
    a2.set_xscale("log"); a2.set_xlabel("Mean burst length of clean violations"); a2.set_ylabel("Stuck-fault delay (requests)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "persistence.pdf"), bbox_inches="tight"); plt.close(fig)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
