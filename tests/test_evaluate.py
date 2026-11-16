import numpy as np
import pandas as pd
import pytest

from maize.evaluate import auroc, calibrate, score


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
