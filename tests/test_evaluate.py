import numpy as np
import pandas as pd
import pytest

from maize.evaluate import auroc, calibrate, score, symptomatic_area


def errors(healthy, diseased):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "label": ["Healthy"] * 200 + ["MSV"] * 200,
        "split": (["val"] * 100 + ["train"] * 100) * 2,
        "error": np.r_[rng.uniform(*healthy, 200), rng.uniform(*diseased, 200)],
    })


def test_calibration_follows_the_proposal():
    rows = errors(healthy=(0, 1), diseased=(1, 3))
    cal = calibrate(rows)
    assert cal["e_healthy_median"] < cal["e_healthy_p95"] < cal["e_disease_p99"]
    s = score(rows["error"], cal)
    assert s.between(0, 1).all()
    assert s[rows["label"] == "Healthy"].median() < s[rows["label"] == "MSV"].median()
    mild_up_to, moderate_up_to = cal["grades"]["MSV"]
    assert 0 < mild_up_to < moderate_up_to < 1


def test_refuses_a_model_that_does_not_separate_healthy_from_diseased():
    with pytest.raises(SystemExit):
        calibrate(errors(healthy=(2, 3), diseased=(0, 1)))


def test_auroc():
    assert auroc(pd.Series([0.1, 0.2]), pd.Series([0.3, 0.4])) == 1.0
    assert auroc(pd.Series([0.3, 0.4]), pd.Series([0.1, 0.2])) == 0.0


def test_scaled_mae_ignores_units():
    from maize.evaluate import scaled_mae
    assert scaled_mae(pd.Series([0, 1, 2]), pd.Series([0, 50, 100])) == pytest.approx(0)
    assert scaled_mae(pd.Series([0, 1]), pd.Series([1, 0])) == pytest.approx(1)


def test_colour_distance_counts_leaf_pixels_only():
    from maize.evaluate import colour_distance, mean_leaf_colour
    images = np.zeros((1, 4, 4, 3), np.uint8)
    images[0, :2] = (30, 40, 0)                 # leaf: 50 away from black
    images[0, 2:] = (255, 255, 255)             # background: ignored
    masks = np.zeros((1, 4, 4), np.uint8)
    masks[0, :2] = 1
    assert colour_distance(images, masks, np.zeros(3))[0] == pytest.approx(50)
    assert mean_leaf_colour(images, masks).tolist() == [30, 40, 0]


def test_report_has_scores_baselines_and_mask_checks():
    from maize.evaluate import report
    rng = np.random.default_rng(0)
    n = 40
    test = pd.DataFrame({
        "label": ["Healthy"] * 20 + ["MSV"] * 20, "split": "test", "group": np.arange(n) // 2,
        "error": np.r_[rng.uniform(0, 1, 20), rng.uniform(1, 3, 20)],
        "leaf_fraction": [0.5] * 39 + [0.02],
        "area": np.r_[np.zeros(20), np.linspace(0, 60, 20)],
        "colour_distance": rng.random(n), "p_diseased": np.nan,
    })
    test["score"] = test["error"] / 3
    results = report(test, {"e_healthy_p95": 0.9, "error": "smoothed"}, {"Healthy": 0.0, "MSV": 0.05})
    assert results["diseases"]["MSV"]["auroc_vs_healthy"] == 1.0
    assert results["diseases"]["MSV"]["n"] == 20
    assert set(results["baselines"]["MSV"]) == {"colour_distance"}   # no classifier given
    assert results["doubtful_mask_share"] == {"Healthy": 0.0, "MSV": 0.05}
    assert results["error"] == "smoothed" and results["within_group_error_sd"] > 0


# ---- symptomatic area A(x) ----
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


def test_onnx_batch_size_follows_the_exported_model():
    from types import SimpleNamespace

    from maize.evaluate import onnx_batch_size

    def session(first_dim):
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(shape=[first_dim, 224, 224, 3])])

    assert onnx_batch_size(session(1)) == 1              # exported for one image at a time
    assert onnx_batch_size(session("batch")) == 64       # any number of images
    assert onnx_batch_size(session(None)) == 64
