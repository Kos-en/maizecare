import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from maize.pipeline import COLUMNS, DataPipeline, load_manifest

FAR = (1 << 40) - 1  # 40 bits away from 0: never a near copy


def manifest(*images):
    """images: (label, hash as int, sha1 or None). Same hash used for pHash and aHash."""
    rows = [{"path": f"{i:04d}.jpg", "label": label, "status": "ok", "phash": f"{h:016x}", "ahash": f"{h:016x}",
             "sha1": sha or hashlib.sha1(str(i).encode()).hexdigest()} for i, (label, h, sha) in enumerate(images)]
    return pd.DataFrame(rows, columns=COLUMNS)


def cleaned(df):
    """Run the cleaning steps (2-4) on a made-up manifest; no image files needed."""
    pipeline = DataPipeline(raw=Path("unused"), root=Path("unused"))
    pipeline.df = df
    pipeline.clean()
    return pipeline


# ---- cleaning rules on made-up hashes -----------------------------------------------------------
def test_chain_of_near_copies_is_one_group_in_one_split():
    a, b, c = 0, (1 << 5) - 1, (1 << 10) - 1  # a-b and b-c are 5 bits apart, a-c is 10
    df = cleaned(manifest(("Healthy", a, None), ("Healthy", b, None), ("Healthy", c, None), ("Healthy", FAR, None))).df
    assert df["group"].nunique() == 2
    assert df["group"][0] == df["group"][2]
    assert df.groupby("group")["split"].nunique().max() == 1


def test_close_on_one_hash_only_is_not_a_near_copy():
    rows = manifest(("Healthy", 0, None), ("MSV", 0, None), ("Healthy", FAR, None))
    rows.loc[1, "phash"] = f"{FAR:016x}"   # same rough brightness layout (aHash), different picture (pHash)
    df = cleaned(rows).df
    assert df["group"][0] != df["group"][1]
    assert list(df["status"]) == ["ok", "ok", "ok"]


def test_group_with_two_labels_is_dropped_whole():
    df = cleaned(manifest(("MSV", 0, None), ("MLND", 0b11, None), ("MLND", FAR, None))).df
    assert list(df["status"]) == ["label_conflict", "label_conflict", "ok"]


def test_exact_copy_keeps_one():
    df = cleaned(manifest(("Healthy", 0, "ab"), ("Healthy", 0, "ab"))).df
    assert list(df["status"]) == ["ok", "duplicate_exact"]


def test_file_faults_stay_and_old_statuses_are_recomputed():
    df = manifest(("Healthy", 0, None), ("Healthy", FAR, None)).assign(status=["bad_image", "split_conflict"])
    df = cleaned(df).df
    assert list(df["status"]) == ["bad_image", "ok"]
    assert list(df["split"]) == ["", df["split"][1]] and df["split"][1] in {"train", "val", "test"}


def test_near_copies_never_cross_splits():
    hashes = np.random.default_rng(0).integers(0, 2 ** 63, 3000)
    images = [("Healthy", int(h), None) for h in hashes] + [("Healthy", int(h) ^ 1, None) for h in hashes[:300]]
    pipeline = cleaned(manifest(*images))
    assert len(pipeline.pairs) == 300 and pipeline.count_leaks() == 0


def test_split_is_stratified_by_class_and_repeatable():
    rng = np.random.default_rng(0)
    images = [(label, int(h), None) for label, n in (("Healthy", 2000), ("MSV", 300))
              for h in rng.integers(0, 2 ** 63, n)]
    df = cleaned(manifest(*images)).df
    share = df.groupby("label")["split"].value_counts(normalize=True)
    for label in ("Healthy", "MSV"):   # each class on its own gets ~15 / 15 / 70
        assert abs(share[label, "test"] - 0.15) < 0.01 and abs(share[label, "val"] - 0.15) < 0.01
    again = cleaned(manifest(*images)).df
    assert (again["split"] == df["split"]).all()


# ---- end to end on real image files -------------------------------------------------------------
def blob(seed, noise=0.0):
    """Smooth random image: stable perceptual hash; different seeds give different hashes."""
    base = np.kron(np.random.default_rng(seed).random((8, 8, 3)), np.ones((16, 16, 1))) * 255
    base += np.random.default_rng(seed + 999).normal(0, noise, base.shape) if noise else 0
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def save(raw, label, name, image):
    (raw / label).mkdir(parents=True, exist_ok=True)
    image.save(raw / label / name)


def test_run_end_to_end_and_rerun_changes_nothing(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "data"
    for i in range(20):
        save(raw, "Healthy", f"h{i}.png", blob(i))
    for i in range(20, 30):
        save(raw, "MLND", f"m{i}.png", blob(i))
    save(raw, "Healthy", "copy_of_h0.png", blob(0))
    save(raw, "MLND", "h1_again.png", blob(1, noise=2))  # the same leaf filed under two classes
    save(raw, "Healthy", "tiny.png", Image.new("RGB", (20, 20)))
    (raw / "Healthy" / "corrupt.png").write_bytes(b"not an image")

    df = DataPipeline(raw, root).run()
    status = {Path(p).name: s for p, s in zip(df["path"], df["status"])}
    assert sorted([status["h0.png"], status["copy_of_h0.png"]]) == ["duplicate_exact", "ok"]
    assert status["h1.png"] == status["h1_again.png"] == "label_conflict"
    assert status["tiny.png"] == "too_small" and status["corrupt.png"] == "bad_image"
    assert (df["status"] == "ok").sum() == 34 - 5  # 34 files; 5 set aside above

    before = load_manifest(root)
    DataPipeline(raw, root).run()
    pd.testing.assert_frame_equal(before, load_manifest(root))


def test_run_also_caches_images_and_leaf_masks(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "data"
    for i in range(6):
        save(raw, "Healthy", f"h{i}.png", blob(i))
    DataPipeline(raw, root).run()
    for name in ("cache_128.npz", "mask_v2_128.npz", "cache_224.npz", "mask_v2_224.npz"):
        with np.load(root / name) as z:
            assert len(z["sha1"]) == 6
