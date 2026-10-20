"""Dataset loaders. Each returns (X, y, groups, cols):
X float64 array in time order, y labels, groups = list of lists of column
indices (lineage groups; groups of one are allowed), cols = feature names.

Data files live under data/ (see README). Nothing here downloads anything.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from .config import CONFIG

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("FC_DATA", os.path.join(HERE, "data"))


def _path(*names):
    for n in names:
        p = os.path.join(DATA, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of {names} found under {DATA}")


def _groups_by_name(cols, spec):
    """spec: list of lists of column names; unnamed columns become singletons."""
    idx = {c: i for i, c in enumerate(cols)}
    groups, used = [], set()
    for g in spec:
        gi = [idx[c] for c in g if c in idx]
        if gi:
            groups.append(gi)
            used.update(gi)
    groups += [[i] for i in range(len(cols)) if i not in used]
    return groups


def load_cc():
    df = pd.read_csv(_path("creditcard.csv", "cc/creditcard.csv"))
    # source row order is time order; Time is not used as a feature
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[cols].to_numpy(float)
    y = df["Class"].to_numpy(int)
    groups = _groups_by_name(cols, [[f"V{i}" for i in range(1, 29)]])
    return X, y, groups, cols


def velocity(amount, windows=(10, 100, 1000)):
    """Causal aggregates of Amount over the previous w requests (current excluded)."""
    s = pd.Series(amount).shift(1)
    feats, names = [], []
    for w in windows:
        r = s.rolling(w, min_periods=1)
        feats += [r.sum(), r.max(), (s > 100).astype(float).rolling(w, min_periods=1).sum()]
        names += [f"amt_sum_{w}", f"amt_max_{w}", f"amt_cnt100_{w}"]
    F = np.column_stack([f.fillna(0.0).to_numpy() for f in feats])
    return F, names


def load_ccv():
    X, y, _, cols = load_cc()
    F, names = velocity(X[:, cols.index("Amount")])
    X = np.hstack([X, F])
    cols = cols + names
    groups = _groups_by_name(cols, [[f"V{i}" for i in range(1, 29)], ["Amount"] + names])
    return X, y, groups, cols


DEFAULT_COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                 "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
                + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)])


def load_default():
    try:
        p = _path("default of credit card clients.xls", "default/default of credit card clients.xls")
        df = pd.read_excel(p, header=1)
        y = df["default payment next month"].to_numpy(int)
        X = df[DEFAULT_COLS].to_numpy(float)
    except FileNotFoundError:
        p = _path("UCI_Credit_Card.csv", "default.csv")
        df = pd.read_csv(p)
        ycol = [c for c in df.columns if "default" in c.lower() or c.lower() == "y"][0]
        y = df[ycol].to_numpy(int)
        feat = [c for c in df.columns if c not in (ycol, "ID")]
        X = df[feat].to_numpy(float)          # positional: same 23 features in order
    # no timestamp: shuffle with a fixed seed (no natural drift)
    perm = np.random.default_rng(CONFIG["default_shuffle_seed"]).permutation(len(y))
    X, y = X[perm], y[perm]
    cols = DEFAULT_COLS
    groups = _groups_by_name(cols, [
        ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"],
        ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
        [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)],
    ])
    return X, y, groups, cols


def _encode_objects(df):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            codes = df[c].astype("category").cat.codes.astype(float)
            df[c] = codes.where(codes >= 0, np.nan)
    return df


def _chunked(names, prefix, size=30):
    sel = [n for n in names if n.startswith(prefix) and n[len(prefix):].isdigit()]
    sel.sort(key=lambda n: int(n[len(prefix):]))
    return [sel[i:i + size] for i in range(0, len(sel), size)]


def per_entity_velocity(entity, amount, windows=(10, 100)):
    """Causal per-entity aggregates over that entity's previous w transactions."""
    df = pd.DataFrame({"e": entity, "a": amount})
    g = df.groupby("e", sort=False)["a"]
    prev = g.shift(1)
    df["p"] = prev
    gp = df.groupby("e", sort=False)["p"]
    out, names = [], []
    for w in windows:
        r = gp.rolling(w, min_periods=1)
        for agg, nm in ((r.sum(), "sum"), (r.max(), "max"), (r.count(), "cnt")):
            v = agg.reset_index(level=0, drop=True).sort_index()
            out.append(v.reindex(df.index).fillna(0.0))
            names.append(f"ent_{nm}_{w}")
    return np.column_stack([o.to_numpy() for o in out]), names


def load_ieee(entity_velocity=True):
    p = _path("ieee/train_transaction.csv", "train_transaction.csv")
    head = pd.read_csv(p, nrows=1000)
    strcols = [c for c in head.columns if not pd.api.types.is_numeric_dtype(head[c])]
    dtypes = {c: ("category" if c in strcols else np.float32) for c in head.columns}
    dtypes.update({"TransactionID": np.int64, "isFraud": np.int8, "TransactionDT": np.int64})
    df = pd.read_csv(p, dtype=dtypes)      # float32 keeps memory near 1 GB
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    y = df["isFraud"].to_numpy(int)
    feat = df.drop(columns=["TransactionID", "isFraud", "TransactionDT"])
    for c in strcols:
        codes = feat[c].cat.codes.astype(np.float32)
        feat[c] = codes.where(codes >= 0, np.nan)
    cols = list(feat.columns)
    X = feat.to_numpy(np.float32)
    del feat
    spec = [["TransactionAmt", "dist1", "dist2"], [c for c in cols if c.startswith("card")],
            ["addr1", "addr2"], [f"M{i}" for i in range(1, 10)]]
    spec += _chunked(cols, "C") + _chunked(cols, "D") + _chunked(cols, "V")
    if entity_velocity:
        ent = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        F, names = per_entity_velocity(ent.to_numpy(), df["TransactionAmt"].to_numpy(float))
        X = np.hstack([X, F.astype(np.float32)])
        cols = cols + names
        spec[0] = spec[0] + names
    return X, y, _groups_by_name(cols, spec), cols


def load_baf():
    p = _path("baf/Base.csv", "Base.csv")
    df = pd.read_csv(p)
    df = df.sort_values("month", kind="stable").reset_index(drop=True)
    y = df["fraud_bool"].to_numpy(int)
    feat = _encode_objects(df.drop(columns=["fraud_bool", "month"]))
    cols = list(feat.columns)
    X = feat.to_numpy(np.float64)
    spec = [["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
             "bank_branch_count_8w", "date_of_birth_distinct_emails_4w"],
            ["income", "credit_risk_score", "proposed_credit_limit", "customer_age"],
            ["prev_address_months_count", "current_address_months_count", "bank_months_count"],
            ["session_length_in_minutes", "device_distinct_emails_8w", "device_fraud_count"]]
    return X, y, _groups_by_name(cols, spec), cols


def load_synthetic(n=60000, d=12, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    W = rng.normal(size=(3, d))
    X = base @ W + 0.5 * rng.normal(size=(n, d))
    X[:, -2] = rng.integers(0, 2, n)
    X[:, -1] = np.exp(rng.normal(size=n))
    y = (base[:, 0] + 0.3 * rng.normal(size=n) > 2.0).astype(int)
    cols = [f"f{i}" for i in range(d)]
    groups = [list(range(0, d - 2)), [d - 2], [d - 1]]
    return X, y, groups, cols


LOADERS = {"cc": load_cc, "ccv": load_ccv, "default": load_default,
           "ieee": load_ieee, "baf": load_baf, "synthetic": load_synthetic}


def load(name):
    return LOADERS[name]()


def splits(n):
    r = CONFIG["splits"]
    n_ref = int(n * r["reference"])
    n_fit = int(n_ref * r["fit_share_of_reference"])
    return slice(0, n_fit), slice(n_fit, n_ref), slice(n_ref, n)
