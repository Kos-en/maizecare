import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from maize.severity import SEVERITY_WEIGHTS, extract_quick_features, formula_severity


def test_features_cover_all_weights_and_score_is_bounded(tmp_path):
    green, brown = np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8)
    green[...] = (40, 160, 50)[::-1]   # BGR green
    brown[...] = (20, 60, 120)         # BGR brown
    scores = []
    for name, img in (("green", green), ("brown", brown)):
        p = tmp_path / f"{name}.png"
        cv2.imwrite(str(p), img)
        feats = extract_quick_features(p)
        assert set(SEVERITY_WEIGHTS) <= set(feats)
        scores.append(formula_severity(feats))
    assert all(0 <= s <= 1 for s in scores)
    assert scores[1] > scores[0], "a brown leaf must score as more severe than a green one"
