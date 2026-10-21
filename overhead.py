"""Monitoring overhead on each dataset: per-request check time (single request and
batched), contract size, and the time to fit a contract (post-confirmation analysis).

python3 overhead.py --out results_v3
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from fcontracts.config import CONFIG
from fcontracts.contracts import FeatureContract
from fcontracts.data import load, splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--datasets", nargs="*", default=["cc", "ccv", "default", "baf", "ieee"])
    a = ap.parse_args()
    out = {}
    for ds in a.datasets:
        X, y, groups, cols = load(ds)
        sf, sc, st = splits(len(X))
        t0 = time.perf_counter()
        c = FeatureContract(groups, CONFIG["alpha"][ds]).fit(X[sf], X[sc])
        fit_s = time.perf_counter() - t0
        Xt = X[st][:20000]
        t0 = time.perf_counter(); c.check(Xt); batch_us = (time.perf_counter() - t0) / len(Xt) * 1e6
        single = []
        for i in range(300):
            t0 = time.perf_counter(); c.check(Xt[i:i + 1]); single.append((time.perf_counter() - t0) * 1e6)
        size_kb = len(pickle.dumps(c)) / 1024
        out[ds] = {"d": int(X.shape[1]), "fit_seconds": fit_s, "batch_us_per_request": batch_us,
                   "single_us_median": float(np.median(single)), "single_us_p99": float(np.percentile(single, 99)),
                   "contract_kb": size_kb}
        print(ds, out[ds], flush=True)
    with open(os.path.join(a.out, "overhead.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
