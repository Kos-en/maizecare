import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from maize.data import COLUMNS, clean, count_leaks, load_manifest, prepare

FAR = (1 << 40) - 1  # 40 bits away from 0: never a near copy


def manifest(*images):
    """images: (label, hash as int, sha1 or None). Same hash used for pHash and aHash."""
    rows = [{"path": f"{i:04d}.jpg", "label": label, "status": "ok", "phash": f"{h:016x}", "ahash": f"{h:016x}",
             "sha1": sha or hashlib.sha1(str(i).encode()).hexdigest()} for i, (label, h, sha) in enumerate(images)]
    return pd.DataFrame(rows, columns=COLUMNS)


# ---- cleaning rules on made-up hashes -----------------------------------------------------------
def test_chain_of_near_copies_is_one_group_in_one_split():
    a, b, c = 0, (1 << 5) - 1, (1 << 10) - 1  # a-b and b-c are 5 bits apart, a-c is 10
    df = clean(manifest(("Healthy", a, None), ("Healthy", b, None), ("Healthy", c, None), ("Healthy", FAR, None)))
    assert df["group"].nunique() == 2
    assert df["group"][0] == df["group"][2]
    assert df.groupby("group")["split"].nunique().max() == 1


def test_group_with_two_labels_is_dropped_whole():
    df = clean(manifest(("MSV", 0, None), ("MLND", 0b11, None), ("MLND", FAR, None)))
    assert list(df["status"]) == ["label_conflict", "label_conflict", "ok"]


def test_exact_copy_keeps_one():
    df = clean(manifest(("Healthy", 0, "ab"), ("Healthy", 0, "ab")))
    assert list(df["status"]) == ["ok", "duplicate_exact"]


def test_file_faults_stay_and_old_statuses_are_recomputed():
    df = manifest(("Healthy", 0, None), ("Healthy", FAR, None)).assign(status=["bad_image", "split_conflict"])
    df = clean(df)
    assert list(df["status"]) == ["bad_image", "ok"]
    assert list(df["split"]) == ["", df["split"][1]] and df["split"][1] in {"train", "val", "test"}


def test_split_sizes_and_no_leaks():
    hashes = np.random.default_rng(0).integers(0, 2 ** 63, 3000)
    images = [("Healthy", int(h), None) for h in hashes] + [("Healthy", int(h) ^ 1, None) for h in hashes[:300]]
    df = clean(manifest(*images))
    share = df["split"].value_counts(normalize=True)
    assert abs(share["test"] - 0.15) < 0.03 and abs(share["val"] - 0.15) < 0.03
    assert count_leaks(df) == 0


# ---- end to end on real image files -------------------------------------------------------------
def blob(seed, noise=0.0):
    """Smooth random image: stable perceptual hash; different seeds give different hashes."""
    base = np.kron(np.random.default_rng(seed).random((8, 8, 3)), np.ones((16, 16, 1))) * 255
    base += np.random.default_rng(seed + 999).normal(0, noise, base.shape) if noise else 0
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def save(raw, label, name, image):
    (raw / label).mkdir(parents=True, exist_ok=True)
    image.save(raw / label / name)


def test_prepare_end_to_end_and_rerun_changes_nothing(tmp_path):
    raw, root = tmp_path / "raw", tmp_path / "data"
    for i in range(20):
        save(raw, "Healthy", f"h{i}.png", blob(i))
    for i in range(20, 30):
        save(raw, "MLND", f"m{i}.png", blob(i))
    save(raw, "Healthy", "copy_of_h0.png", blob(0))
    save(raw, "MLND", "h1_again.png", blob(1, noise=2))  # the same leaf filed under two classes
    save(raw, "Healthy", "tiny.png", Image.new("RGB", (20, 20)))
    (raw / "Healthy" / "corrupt.png").write_bytes(b"not an image")

    df = prepare(raw, root)
    status = {Path(p).name: s for p, s in zip(df["path"], df["status"])}
    assert sorted([status["h0.png"], status["copy_of_h0.png"]]) == ["duplicate_exact", "ok"]
    assert status["h1.png"] == status["h1_again.png"] == "label_conflict"
    assert status["tiny.png"] == "too_small" and status["corrupt.png"] == "bad_image"
    assert (df["status"] == "ok").sum() == 34 - 5  # 34 files; 5 set aside above

    before = load_manifest(root)
    prepare(raw, root)
    pd.testing.assert_frame_equal(before, load_manifest(root))
