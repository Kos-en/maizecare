import numpy as np
import pandas as pd
import pytest

from maize.data import COLUMNS, load_manifest
from maize.prepare import Index, count_leaks, decide, prepare


# ---- pure decision logic (numpy only) ---------------------------------------------------
def make_index(rows):
    """rows: (phash_int, ahash_int, group, split, label)"""
    df = pd.DataFrame([{"phash": f"{p:016x}", "ahash": f"{a:016x}", "group": g, "split": s, "label": l}
                       for p, a, g, s, l in rows])
    return Index(df)


def test_new_image_gets_new_group_and_deterministic_split():
    idx = make_index([(0, 0, 0, "train", "Healthy")])
    far = (1 << 40) - 1                                   # 40 bits away: not a near copy
    s1 = decide("Healthy", far, far, idx, 6, 0.15)
    s2 = decide("Healthy", far, far, idx, 6, 0.15)
    assert s1 == s2 and s1[0] == "ok" and s1[1] == 1 and s1[2] in ("train", "val")


def test_near_copy_inherits_group_and_split():
    idx = make_index([(0, 0, 7, "val", "Healthy")])
    assert decide("Healthy", 0b101, 0b101, idx, 6, 0.15) == ("ok", 7, "val")


def test_near_copy_bridging_train_and_val_is_quarantined():
    a, b = 0, (1 << 12) - 1                               # 12 bits apart -> distinct groups
    idx = make_index([(a, a, 0, "train", "Healthy"), (b, b, 1, "val", "Healthy")])
    new = (1 << 6) - 1                                    # 6 bits from both
    assert decide("Healthy", new, new, idx, 6, 0.15)[0] == "split_conflict"


def test_same_picture_in_another_class_is_quarantined():
    idx = make_index([(0, 0, 0, "train", "MLND")])
    assert decide("MSV", 0b11, 0b11, idx, 6, 0.15)[0] == "label_conflict"


# ---- end to end on real files (needs imagehash) -----------------------------------------
pytest.importorskip("imagehash")
from PIL import Image  # noqa: E402


def blob(seed, noise=0.0):
    """Smooth random image: stable perceptual hash, different seeds give different hashes."""
    rng = np.random.default_rng(seed)
    base = np.kron(rng.random((8, 8, 3)), np.ones((16, 16, 1))) * 255
    if noise:
        base = base + np.random.default_rng(seed + 999).normal(0, noise, base.shape)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def save(raw, cls, name, img):
    d = raw / cls
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / name)


@pytest.fixture
def world(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "data"
    for i in range(30):
        save(raw, "Healthy", f"h{i}.png", blob(i))
    for i in range(30, 40):
        save(raw, "MLND", f"m{i}.png", blob(i))
    return raw, root


def test_first_run_then_rerun_is_a_noop(world):
    raw, root = world
    df1 = prepare(raw, root, val_frac=0.5)
    assert (df1["status"] == "ok").sum() == 40 and set(df1["split"]) == {"train", "val"}
    before = load_manifest(root).copy()
    prepare(raw, root, val_frac=0.5)
    pd.testing.assert_frame_equal(before, load_manifest(root))


def test_new_data_cleans_itself_and_never_moves_old_splits(world):
    raw, root = world
    prepare(raw, root, val_frac=0.5)
    old_df = load_manifest(root)
    old = old_df.set_index("path")[["group", "split"]]
    healthy = old_df[old_df["label"] == "Healthy"].set_index("path")
    h_train = healthy[healthy["split"] == "train"].index[0]
    h_val = healthy[healthy["split"] == "val"].index[0]
    seed = lambda p: int(p.split("/")[-1].lstrip("hm").split(".")[0])  # file h12.png -> blob seed 12

    # a new drop: exact copy, near copies of a train and a val image, corrupt, tiny, brand new
    save(raw, "Healthy", "copy_of_h0.png", Image.open(raw / "Healthy" / "h0.png"))
    save(raw, "Healthy", "near_train.png", blob(seed(h_train), noise=2))
    save(raw, "Healthy", "near_val.png", blob(seed(h_val), noise=2))
    (raw / "Healthy" / "corrupt.png").write_bytes(b"not an image")
    save(raw, "Healthy", "tiny.png", Image.new("RGB", (20, 20)))
    save(raw, "Healthy", "brand_new.png", blob(500))

    df = prepare(raw, root, val_frac=0.5).set_index("path")
    st = lambda name: df.loc[[p for p in df.index if p.endswith(name)][0]]  # noqa: E731
    assert st("copy_of_h0.png")["status"] == "duplicate_exact"
    assert st("corrupt.png")["status"] == "bad_image"
    assert st("tiny.png")["status"] == "too_small"
    assert st("brand_new.png")["status"] == "ok"
    assert st("near_train.png")["split"] == old.loc[h_train, "split"]
    assert st("near_train.png")["group"] == old.loc[h_train, "group"]
    assert st("near_val.png")["split"] == old.loc[h_val, "split"]
    # existing images did not move
    assert (df.loc[old.index, ["group", "split"]].values == old.values).all()
    assert count_leaks(df.reset_index()) == 0


def test_manifest_has_expected_columns(world):
    raw, root = world
    assert list(prepare(raw, root).columns) == COLUMNS
