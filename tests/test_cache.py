import shutil

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tqdm")
from maize.pipeline import load_images
from maize.preprocess import resize


def make_rows(tmp_path, ids):
    rows = []
    for i in ids:
        img = np.random.default_rng(i).integers(0, 255, (200, 150, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{i}.png"), img)
        rows.append({"path": f"{i}.png", "sha1": f"sha{i}"})
    return pd.DataFrame(rows)


def test_cache_matches_direct_preprocessing_and_keeps_order(tmp_path):
    rows = make_rows(tmp_path, [3, 1, 2])
    out = load_images(rows, tmp_path, 128)
    assert out.shape == (3, 128, 128, 3) and out.dtype == np.uint8
    direct = resize(cv2.cvtColor(cv2.imread(str(tmp_path / "1.png")), cv2.COLOR_BGR2RGB), 128)
    assert (out[1] == direct).all()


def test_second_call_does_not_read_images_and_new_images_are_added(tmp_path):
    rows = make_rows(tmp_path, [1, 2])
    first = load_images(rows, tmp_path, 128)
    for i in (1, 2):
        (tmp_path / f"{i}.png").unlink()          # originals gone: must be served from the cache
    assert (load_images(rows, tmp_path, 128) == first).all()
    more = make_rows(tmp_path, [3])
    both = load_images(pd.concat([rows, more], ignore_index=True), tmp_path, 128)
    assert both.shape[0] == 3 and (both[:2] == first).all()


def test_masks_are_cached_with_the_same_keys(tmp_path):
    from maize.pipeline import load, load_masks
    from maize.preprocess import prepare
    rows = make_rows(tmp_path, [1, 2])
    images, masks = load(rows, tmp_path, 128)
    assert masks.shape == (2, 128, 128) and set(np.unique(masks)) <= {0, 1}
    direct = prepare(cv2.cvtColor(cv2.imread(str(tmp_path / "2.png")), cv2.COLOR_BGR2RGB), 128)
    assert (images[1] == direct[0]).all() and (masks[1] == direct[1]).all()
    assert (tmp_path / "mask_v2_128" / "part_0000.npz").exists()
    for i in (1, 2):
        (tmp_path / f"{i}.png").unlink()
    assert (load_masks(rows, tmp_path, 128) == masks).all()


def test_a_missing_mask_is_made_from_the_cached_image_without_the_photo(tmp_path):
    from maize.pipeline import load
    rows = make_rows(tmp_path, [1, 2])
    images, masks = load(rows, tmp_path, 128)
    shutil.rmtree(tmp_path / "mask_v2_128")               # e.g. a new mask version
    for i in (1, 2):
        (tmp_path / f"{i}.png").unlink()                  # the photos are not needed for it
    images2, masks2 = load(rows, tmp_path, 128)
    assert (images2 == images).all() and (masks2 == masks).all()
    assert not (tmp_path / "cache_128" / "part_0001.npz").exists()   # the images were not saved again


def test_the_cache_is_saved_in_chunks(tmp_path, monkeypatch):
    import maize.pipeline as pipeline
    monkeypatch.setattr(pipeline, "CACHE_SAVE_EVERY", 2)
    rows = make_rows(tmp_path, [1, 2, 3, 4, 5])
    images, masks = pipeline.load(rows, tmp_path, 128)
    assert images.shape[0] == masks.shape[0] == 5
    assert (images[4] == resize(cv2.cvtColor(cv2.imread(str(tmp_path / "5.png")), cv2.COLOR_BGR2RGB), 128)).all()


def test_a_damaged_cache_file_is_rebuilt(tmp_path):
    from maize.pipeline import load
    rows = make_rows(tmp_path, [1, 2])
    images, masks = load(rows, tmp_path, 128)
    (tmp_path / "mask_v2_128" / "part_0000.npz").write_bytes(b"half a file")   # an interrupted write
    assert (load(rows, tmp_path, 128)[1] == masks).all()
    assert (tmp_path / "mask_v2_128" / "part_0001.npz").exists()     # redone as a new part


def test_a_single_cache_file_from_an_older_run_is_still_read(tmp_path):
    from maize.pipeline import load
    rows = make_rows(tmp_path, [1, 2])
    images, masks = load(rows, tmp_path, 128)
    with np.load(tmp_path / "cache_128" / "part_0000.npz") as z:
        np.savez(tmp_path / "cache_128.npz", sha1=z["sha1"], x=z["x"])   # how older runs stored it
    shutil.rmtree(tmp_path / "cache_128")
    for i in (1, 2):
        (tmp_path / f"{i}.png").unlink()
    assert (load(rows, tmp_path, 128)[0] == images).all()
