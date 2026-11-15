import numpy as np
import pytest

from maize import severity as sev


def test_masked_error_ignores_background():
    x, xh = np.zeros((8, 8, 3)), np.zeros((8, 8, 3))
    xh[:, 4:] = 1.0                       # error only on the right half
    m = np.zeros((8, 8))
    m[:, :4] = 1                          # leaf is the left half
    assert sev.masked_error(x, xh, m) == 0.0
    m[:, :] = 1
    assert sev.masked_error(x, xh, m) == pytest.approx(0.5)


def test_calibration_score_and_grades():
    rng = np.random.default_rng(0)
    healthy = rng.normal(0.010, 0.002, 500)
    diseased = {d: rng.normal(0.020 + 0.01 * i, 0.008, 400) for i, d in enumerate(["MLND", "MSV", "Blight"])}
    allv = np.concatenate(list(diseased.values()))
    cal = sev.fit_calibration(healthy, allv, diseased)
    assert sev.score(cal["e_healthy_median"], cal) == 0.0
    assert sev.score(cal["e_disease_p99"] * 2, cal) == 1.0
    assert sev.grade(cal["e_healthy_p95"] - 1e-6, "MLND", cal) == "Within healthy range"
    lo, hi = cal["e_healthy_median"] + 0.3 * (cal["e_disease_p99"] - cal["e_healthy_median"]), cal["e_disease_p99"]
    grades = [sev.grade(e, "MLND", cal) for e in np.linspace(cal["e_healthy_p95"] + 1e-6, hi, 50)]
    assert grades[0] in ("Mild", "Moderate") and grades[-1] == "Severe"
    order = {"Mild": 0, "Moderate": 1, "Severe": 2}
    seq = [order[g] for g in grades if g in order]
    assert seq == sorted(seq), "grades must be monotone in the error"


def test_calibration_rejects_non_separating_model():
    with pytest.raises(ValueError):
        sev.fit_calibration(np.full(10, 0.02), np.full(10, 0.01), {"MLND": np.full(10, 0.01)})
