"""Synthetic quickstart: learn a feature contract, break one feature, watch the pages.

Runs in a few seconds with no datasets. Everything here is generated, so it shows how
the pieces connect, not how the method performs on real data.

    python3 examples/quickstart_synthetic.py            # prints a summary
    python3 examples/quickstart_synthetic.py --fig docs/quickstart_synthetic.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fcontracts.config import CONFIG                      # noqa: E402
from fcontracts.contracts import FeatureContract          # noqa: E402
from fcontracts.inject import inject, reference_stats     # noqa: E402
from fcontracts.paging import burst_aware_pages           # noqa: E402

FEATURES = ["amount", "amount_x2", "balance", "age_days", "channel", "noise"]
FAULT_START, FAULT_END, FAULT_FEATURE = 3000, 4500, 1


def make_stream(n, rng):
    amount = rng.lognormal(3.0, 0.6, n)
    amount_x2 = 2.0 * amount + rng.normal(0, 1.5, n)      # tied to amount
    balance = 40.0 * amount + rng.normal(0, 60, n)         # loosely tied to amount
    age_days = rng.integers(0, 3000, n).astype(float)
    channel = rng.integers(0, 4, n).astype(float)          # discrete, four values
    noise = rng.normal(0, 1, n)
    return np.column_stack([amount, amount_x2, balance, age_days, channel, noise])


def run(seed=0, prevalence=0.2):
    rng = np.random.default_rng(seed)
    ref = make_stream(25000, rng)
    X_fit, X_cal = ref[:20000], ref[20000:]
    contract = FeatureContract(groups=[[0, 1, 2]], alpha=CONFIG["alpha"]["synthetic"],
                               calibration="conformal").fit(X_fit, X_cal)

    stream = make_stream(6000, rng)
    window = stream[FAULT_START:FAULT_END]
    stats = reference_stats(ref)
    # "misjoin_ref" swaps in values from other records, so every single value stays plausible
    corrupted, _ = inject(window, "misjoin_ref", [FAULT_FEATURE], prevalence, rng, stats, pool=ref)
    bad = stream.copy()
    bad[FAULT_START:FAULT_END] = corrupted

    clean_res, bad_res = contract.check(stream), contract.check(bad)
    pages = burst_aware_pages(bad_res.viol_rows, bad_res.viol_cols, contract.d,
                              contract.p0_onset, contract.max_run)
    return contract, stream, bad, clean_res, bad_res, pages


def plot(contract, bad, bad_res, pages, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(bad)
    t = np.arange(n)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax[0].plot(t, bad[:, 1], lw=0.5, color="#4c78a8")
    ax[0].axvspan(FAULT_START, FAULT_END, color="#e45756", alpha=0.10, label="fault window")
    ax[0].axhline(contract.env_hi[1], ls="--", lw=0.8, color="#555", label="learned envelope")
    ax[0].axhline(contract.env_lo[1], ls="--", lw=0.8, color="#555")
    hit = np.unique(bad_res.viol_rows[bad_res.viol_cols == 1])
    ax[0].scatter(hit, bad[hit, 1], s=7, color="#e45756", zorder=3, label="contract violation")
    ax[0].set_ylabel("amount_x2")
    ax[0].set_title("Synthetic stream: one feature loses its relation to its neighbours")
    ax[0].set_ylim(top=float(contract.env_hi[1]) * 1.25)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)

    cnt = np.zeros(n // 100)
    np.add.at(cnt, hit // 100, 1)
    ax[1].bar(np.arange(len(cnt)) * 100 + 50, cnt, width=90, color="#e45756")
    ax[1].set_ylabel("violations\nper 100")

    ax[2].set_ylim(-0.5, len(FEATURES) - 0.5)
    for ti, j in pages:
        ax[2].plot([ti], [j], "v", color="#222", ms=8)
    ax[2].set_yticks(range(len(FEATURES)))
    ax[2].set_yticklabels(FEATURES, fontsize=8)
    ax[2].set_ylabel("pages")
    ax[2].set_xlabel("request index")
    for a in ax:
        a.axvspan(FAULT_START, FAULT_END, color="#e45756", alpha=0.06)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "All data synthetic. Illustrates the workflow, not results.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", help="write a PNG here (needs matplotlib)")
    a = ap.parse_args()
    contract, stream, bad, clean_res, bad_res, pages = run(a.seed)
    in_win = np.zeros(len(bad), bool)
    in_win[FAULT_START:FAULT_END] = True
    print(f"calibrated flag rate on clean calibration slice: {contract.cal_flag_rate:.4%}")
    print(f"clean stream flag rate:                           {clean_res.flag.mean():.4%}")
    print(f"corrupted stream flag rate inside fault window:   {bad_res.flag[in_win].mean():.4%}")
    print(f"corrupted stream flag rate outside fault window:  {bad_res.flag[~in_win].mean():.4%}")
    first = {}
    for t, j in pages:
        first.setdefault(FEATURES[j], int(t))
    print(f"fault window starts at request {FAULT_START}; pages raised: {len(pages)}")
    for name, t in first.items():
        print(f"  first page on {name}: request {t} ({t - FAULT_START} after fault start)")
    if not first:
        print("  no pages")
    if a.fig:
        plot(contract, bad, bad_res, pages, a.fig)
        print("wrote", a.fig)


if __name__ == "__main__":
    main()
