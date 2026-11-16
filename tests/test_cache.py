import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tqdm")
from maize.cache import load_images
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
