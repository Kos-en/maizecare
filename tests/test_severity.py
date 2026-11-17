import numpy as np
import pytest

from maize.severity import symptomatic_area

GREEN, YELLOW, BROWN = (60, 160, 50), (220, 200, 40), (130, 70, 20)


def test_area_is_the_share_of_the_leaf_with_symptoms():
    leaf = np.full((10, 10, 3), GREEN, np.uint8)
    leaf[:2] = YELLOW               # 20 of 100 pixels yellow
    leaf[2:3] = BROWN               # 10 brown
    assert symptomatic_area(leaf, np.ones((10, 10), np.uint8)) == pytest.approx(30)


def test_pixels_outside_the_mask_do_not_count():
    image = np.full((10, 10, 3), BROWN, np.uint8)
    image[:, :5] = GREEN
    mask = np.zeros((10, 10), np.uint8)
    mask[:, :5] = 1                 # only the green half is leaf
    assert symptomatic_area(image, mask) == 0
    assert symptomatic_area(image, np.zeros((10, 10), np.uint8)) == 0   # no leaf at all
