import cv2
import numpy as np

from maize.preprocess import leaf_fraction, leaf_mask, prepare, resize

GREEN, BROWN, SOIL = (60, 160, 50), (140, 90, 30), (120, 80, 40)


def leaf_photo(h=120, w=200):
    """Brown soil, a green leaf (ellipse) in the middle with a brown lesion inside it."""
    img = np.full((h, w, 3), SOIL, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, h // 4), 0, 0, 360, GREEN, -1)
    cv2.circle(img, (w // 2, h // 2), 8, BROWN, -1)
    return img


def test_mask_keeps_lesion_and_drops_soil():
    mask = leaf_mask(leaf_photo())
    assert mask.shape == (120, 200) and mask.dtype == np.uint8
    assert mask[60, 100] == 1   # lesion in the middle of the leaf
    assert mask[60, 80] == 1    # green tissue
    assert mask[5, 5] == 0      # soil in the corner
    assert 0.10 < leaf_fraction(mask) < 0.95


def test_prepare_pads_with_non_leaf_and_matches_resize():
    photo = leaf_photo(120, 200)
    image, mask = prepare(photo, 128)
    assert image.shape == (128, 128, 3) and mask.shape == (128, 128)
    assert (image == resize(photo, 128)).all()
    assert mask[:10].sum() == 0 and mask[-10:].sum() == 0   # the black padding above and below the photo
    assert mask[64, 64] == 1


def test_resize_keeps_shape_with_black_padding():
    out = resize(np.full((50, 100, 3), 200, np.uint8), 100)
    assert out.shape == (100, 100, 3)
    assert out[:25].max() == 0 and out[50, 50].min() == 200
