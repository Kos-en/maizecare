"""Leakage-free splitting (proposal 3.4).

1. Hold out a test set, whole groups together, class balance preserved.
2. Stratified Group K-Fold (K=5) on the rest -> `fold` column for model comparison.
3. Verify independently that no near-copy pair straddles test / folds.
Augmentation happens later, online, on training images only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .hashing import hex_to_u64, near_pairs


def make_splits(df: pd.DataFrame, test_frac: float = 0.15, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """df needs `label` and `group_id`. Adds `split` ('test'|'pool') and `fold` (-1 for test)."""
    df = df.copy()
    k = max(2, round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], df["group_id"]))
    df["split"] = "pool"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df["fold"] = -1

    pool = df[df["split"] == "pool"]
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, val_idx) in enumerate(cv.split(pool, pool["label"], pool["group_id"])):
        df.loc[pool.index[val_idx], "fold"] = f
    return df


def set_id(df: pd.DataFrame) -> pd.Series:
    return np.where(df["split"] == "test", "test", "fold" + df["fold"].astype(str))


def find_leaks(df: pd.DataFrame, a_max=4, p_max=6, mode="both") -> pd.DataFrame:
    """Near-copy pairs whose two images sit in different sets (test / fold k). Must be empty."""
    ah, ph = hex_to_u64(df["ahash"]), hex_to_u64(df["phash"])
    pairs = near_pairs(ah, ph, a_max, p_max, mode)
    sid = np.asarray(set_id(df))
    bad = pairs[sid[pairs[:, 0]] != sid[pairs[:, 1]]]
    return pd.DataFrame({"i": bad[:, 0], "j": bad[:, 1],
                         "set_i": sid[bad[:, 0]], "set_j": sid[bad[:, 1]]})


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Actual split sizes per class (report these: large groups can distort 70/15/15)."""
    t = df.assign(set=set_id(df)).groupby(["set", "label"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    t["share_%"] = (100 * t["total"] / t["total"].sum()).round(1)
    return t
