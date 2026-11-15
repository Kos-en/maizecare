import numpy as np

from maize.masking import apply_mask, leaf_fraction, leaf_mask, mask_iou
from maize.preprocess import prepare


def synthetic_leaf(h=300, w=400):
    """green leaf with a brown lesion INSIDE and a yellow dead tip, on a gray-brown soil background"""
    img = np.full((h, w, 3), (120, 100, 80), np.uint8)               # soil
    truth = np.zeros((h, w), np.uint8)
    truth[60:240, 80:320] = 1
    img[60:240, 80:320] = (40, 150, 50)                              # green leaf
    img[120:160, 150:200] = (110, 70, 30)                            # brown lesion inside
    img[60:90, 280:320] = (200, 190, 40)                             # yellow tip at the edge
    return img, truth


def test_mask_keeps_lesions_and_drops_soil():
    img, truth = synthetic_leaf()
    m = leaf_mask(img)
    assert m.shape == truth.shape
    assert m[140, 175] == 1, "brown lesion inside the leaf must stay in the mask"
    assert m[75, 300] == 1, "yellow tip touching the green must stay in the mask"
    assert m[10, 10] == 0 and m[290, 390] == 0, "soil must be excluded"
    assert mask_iou(m, truth) > 0.9


def test_leaf_fraction_and_apply_mask():
    img, truth = synthetic_leaf()
    m = leaf_mask(img)
    assert abs(leaf_fraction(m) - truth.mean()) < 0.05
    assert (apply_mask(img, m)[m == 0] == 0).all()


def test_prepare_is_square_and_consistent():
    img, _ = synthetic_leaf()
    p = prepare(img, 224)
    assert p.image.shape == (224, 224, 3) and p.mask.shape == (224, 224)
    assert set(np.unique(p.mask)) <= {0, 1}
    p2 = prepare(img, 224)  # deterministic: training and serving get identical output
    assert (p.image == p2.image).all() and (p.mask == p2.mask).all()
