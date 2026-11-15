import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from maize.data.hashing import group_ids, hamming, near_pairs
from maize.data.split import find_leaks, make_splits


def fake_df(n_groups=120, per_group=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = ["Healthy", "MLND", "MSV", "Blight"]
    for g in range(n_groups):
        base_a, base_p = rng.integers(0, 2**63, 2, dtype=np.uint64)
        for _ in range(per_group):
            flip = lambda v: int(v) ^ (1 << int(rng.integers(0, 64)))  # 1-bit change = near copy
            rows.append({"label": labels[g % 4], "ahash": f"{flip(base_a):016x}", "phash": f"{flip(base_p):016x}"})
    df = pd.DataFrame(rows)
    return df


def test_hamming_and_groups():
    a = np.array([0, 0xFF], dtype=np.uint64)
    assert hamming(a, a)[0, 1] == 8
    df = fake_df()
    from maize.data.hashing import hex_to_u64
    ah, ph = hex_to_u64(df["ahash"]), hex_to_u64(df["phash"])
    gid = group_ids(len(df), near_pairs(ah, ph, 4, 6, "both"))
    assert gid.max() + 1 == 120, "each synthetic group of near copies must collapse to one group"


def test_split_has_no_leakage_and_keeps_classes():
    df = fake_df()
    from maize.data.hashing import hex_to_u64
    ah, ph = hex_to_u64(df["ahash"]), hex_to_u64(df["phash"])
    df["group_id"] = group_ids(len(df), near_pairs(ah, ph, 4, 6, "both"))
    out = make_splits(df, 0.15, 5, 42)
    assert find_leaks(out).empty
    assert set(out["split"]) == {"test", "pool"} and set(out.loc[out.split == "pool", "fold"]) == {0, 1, 2, 3, 4}
    assert out[out.split == "test"]["label"].nunique() == 4
    # a group never straddles sets
    sid = np.where(out["split"] == "test", -1, out["fold"])
    assert (pd.Series(sid).groupby(out["group_id"]).nunique() == 1).all()
