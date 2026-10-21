"""Controlled paging simulation (v3, Appendix B).

Clean streams with the same average violation rate p0 but different clustering
(two-state Markov chain, mean burst length L). Compares per-violation
Shiryaev-Roberts paging (threshold dA) with burst-aware paging (onsets + run
length rule), reporting false pages per 100,000 clean requests and detection
delay for two fault types injected at t0: a sporadic fault (violation rate
rises to 5% per request, independent) and a stuck fault (every request violates).

python3 paging_sim.py [--out results_v3]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

D, A, P0 = 20, 2000.0, 1e-3
P1 = max(0.005, 10 * P0)
MIN_RUN, RUN_MULT = 20, 2.0


def markov(rng, reps, d, T, p, L):
    p_exit = 1.0 / L
    p_enter = p * p_exit / (1 - p)
    z = np.zeros((T, reps, d), bool)
    s = rng.random((reps, d)) < p
    for t in range(T):
        u = rng.random((reps, d))
        s = np.where(s, u >= p_exit, u < p_enter)
        z[t] = s
    return z


def sr_step(R, z, p0, p1, thr):
    lr1, lr0 = p1 / p0, (1 - p1) / (1 - p0)
    R = (1 + R) * np.where(z, lr1, lr0)
    hit = R >= thr
    R[hit] = 0.0
    return R, hit


def run(z, p0_on, max_run):
    """z: (T, reps, d) violations. Returns page indicators for both methods."""
    T, reps, d = z.shape
    thr = d * A
    R_all = np.zeros((reps, d)); R_on = np.zeros((reps, d)); run_len = np.zeros((reps, d))
    prev = np.zeros((reps, d), bool)
    limit = max(MIN_RUN, RUN_MULT * max_run)
    page_all = np.zeros((T, reps), int); page_burst = np.zeros((T, reps), int)
    f0_all = np.zeros((T, reps), bool); f0_burst = np.zeros((T, reps), bool)
    p1_on = max(0.005, 10 * p0_on)
    for t in range(T):
        R_all, h1 = sr_step(R_all, z[t], P0, P1, thr)
        onset = z[t] & ~prev
        R_on, h2 = sr_step(R_on, onset, p0_on, p1_on, thr)
        run_len = np.where(z[t], run_len + 1, 0)
        h3 = run_len == np.ceil(limit) + 1
        page_all[t] = h1.sum(1)
        page_burst[t] = (h2 | h3).sum(1)
        f0_all[t] = h1[:, 0]; f0_burst[t] = (h2 | h3)[:, 0]
        prev = z[t]
    return page_all, page_burst, f0_all, f0_burst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v3")
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    out = {}
    for L in [1, 2, 5, 20, 100]:
        # calibration of the burst-aware test on a clean slice with the same dependence
        cal = markov(rng, 4, D, 30000, P0, L)
        onsets = cal & ~np.concatenate([np.zeros((1, 4, D), bool), cal[:-1]])
        p0_on = max(onsets.mean() * 1.5, 1e-6)
        runs = 0; cur = np.zeros((4, D))
        for t in range(len(cal)):
            cur = np.where(cal[t], cur + 1, 0); runs = max(runs, cur.max())
        clean = markov(rng, 20, D, 50000, P0, L)
        pa, pb, _, _ = run(clean, p0_on, runs)
        res = {"false_pages_per_100k_sr": pa.sum() / (50000 * 20) * 1e5,
               "false_pages_per_100k_burst": pb.sum() / (50000 * 20) * 1e5,
               "bound_per_100k": 1e5 / A}
        for kind in ("sporadic", "stuck"):
            T0, T = 2000, 4000
            z = markov(rng, 50, D, T, P0, L)
            if kind == "sporadic":
                z[T0:, :, 0] |= rng.random((T - T0, 50)) < 0.05
            else:
                z[T0:, :, 0] = True
            _, _, pa, pb = run(z, p0_on, runs)  # detection: a page on the faulty feature (feature 0)
            for name, pg in (("sr", pa), ("burst", pb)):
                delays = []
                for r_ in range(50):
                    idx = np.flatnonzero(pg[T0:, r_])
                    delays.append(idx[0] if idx.size else np.nan)
                delays = np.array(delays, float)
                res[f"{kind}_detected_{name}"] = float(np.mean(~np.isnan(delays)))
                res[f"{kind}_median_delay_{name}"] = float(np.nanmedian(delays)) if np.any(~np.isnan(delays)) else None
        out[L] = res
        print(L, json.dumps(res), flush=True)
    with open(os.path.join(a.out, "paging_sim.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
