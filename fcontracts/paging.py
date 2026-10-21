"""Per-feature sequential paging tests (Section 4.6), computed exactly but only
at violation events: between events every statistic decays deterministically
and cannot cross its threshold, so it is advanced in closed form.
"""
from __future__ import annotations

import math
import numpy as np

from .config import CONFIG


def p1_for(p0):
    c = CONFIG["paging"]
    return np.maximum(c["p1_floor"], c["p1_mult"] * np.asarray(p0))


def page_times_one(events, p0, p1, threshold, test="sr"):
    """Times (sorted event indices) at which one feature pages, restarting after
    each page. events: sorted int array of times with z_t = 1."""
    lr1, lr0 = p1 / p0, (1 - p1) / (1 - p0)
    pages = []
    if test == "sr":
        fix = lr0 / (1 - lr0)
        R, last = 0.0, -1
        for t in events:
            gap = t - last - 1
            if gap > 0:
                f = lr0 ** gap
                R = f * R + fix * (1 - f)
            R = (1 + R) * lr1
            last = t
            if R >= threshold:
                pages.append(t)
                R = 0.0
    else:   # one-sided CUSUM
        l1, l0, h = math.log(lr1), math.log(lr0), math.log(threshold)
        W, last = 0.0, -1
        for t in events:
            gap = t - last - 1
            W = max(0.0, W + gap * l0) if gap > 0 else W
            W = max(0.0, W + l1)
            last = t
            if W >= h:
                pages.append(t)
                W = 0.0
    return pages


def run_paging(rows, cols, d, p0, A=None, test=None, pooled=False):
    """rows, cols: sparse indicator events (time, feature). Returns list of
    (time, feature) pages, sorted by time. Pooled: one stream, threshold A."""
    A = A or CONFIG["paging"]["A"]
    test = test or CONFIG["paging"]["test"]
    out = []
    if pooled:
        ev = np.unique(rows)
        p = float(np.atleast_1d(p0)[0])
        out = [(t, -1) for t in page_times_one(ev, p, float(p1_for(p)), A, test)]
    else:
        p0 = np.asarray(p0, float)
        p1 = p1_for(p0)
        order = np.lexsort((rows, cols))
        rows, cols = rows[order], cols[order]
        bounds = np.searchsorted(cols, np.arange(d + 1))
        for j in range(d):
            ev = np.unique(rows[bounds[j]:bounds[j + 1]])
            if ev.size:
                out += [(t, j) for t in page_times_one(ev, p0[j], p1[j], d * A, test)]
    out.sort()
    return out


def pooled_p0(flag_cal):
    from .contracts import wilson_upper
    n = len(flag_cal)
    return max(wilson_upper(int(flag_cal.sum()), n, CONFIG["paging"]["wilson_z"]),
               CONFIG["paging"]["p0_floor"])


def burst_aware_pages(rows, cols, d, p0_onset, max_run, A=None, min_run=20, run_mult=2.0):
    """Burst-aware paging (v3). Per feature, a Shiryaev-Roberts test on violation
    ONSETS (z_t = 1 and z_{t-1} = 0) instead of every violation, plus a run-length
    rule that pages once per run when a run of consecutive violations exceeds
    max(min_run, run_mult * longest clean run seen in calibration). Onsets of
    clustered violations are far closer to independent than the violations
    themselves, and the run-length rule keeps stuck or systemic faults detectable.
    Returns sorted (time, feature) pages."""
    A = A or CONFIG["paging"]["A"]
    p0_onset = np.asarray(p0_onset, float)
    p1 = p1_for(p0_onset)
    order = np.lexsort((rows, cols))
    rows, cols = rows[order], cols[order]
    bounds = np.searchsorted(cols, np.arange(d + 1))
    out = []
    for j in range(d):
        ev = np.unique(rows[bounds[j]:bounds[j + 1]])
        if not ev.size:
            continue
        new_run = np.r_[True, np.diff(ev) > 1]
        onsets = ev[new_run]
        out += [(t, j) for t in page_times_one(onsets, p0_onset[j], p1[j], d * A, "sr")]
        limit = max(min_run, run_mult * max_run[j])
        run_id = np.cumsum(new_run)
        starts = ev[new_run]
        pos = ev - starts[run_id - 1]          # 0-based position within its run
        hit = ev[pos == int(np.ceil(limit))]
        out += [(int(t), j) for t in hit]
    out.sort()
    return out


def onset_stats(rows, cols, d, n):
    """Per-feature onset counts and longest run of consecutive violations."""
    from .contracts import wilson_upper
    cnt = np.zeros(d); mrun = np.zeros(d)
    order = np.lexsort((rows, cols)); rows, cols = rows[order], cols[order]
    b = np.searchsorted(cols, np.arange(d + 1))
    for j in range(d):
        ev = np.unique(rows[b[j]:b[j + 1]])
        if not ev.size:
            continue
        new_run = np.r_[True, np.diff(ev) > 1]
        cnt[j] = new_run.sum()
        rid = np.cumsum(new_run)
        mrun[j] = np.bincount(rid).max()
    z = CONFIG["paging"]["wilson_z"]
    p0 = np.array([max(wilson_upper(c, n, z), CONFIG["paging"]["p0_floor"]) for c in cnt])
    return p0, mrun
