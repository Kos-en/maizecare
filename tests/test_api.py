"""API + pipeline tests with fake TFLite models (no TensorFlow, no model files needed)."""
import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from maize import severity as sev
from maize.constants import CLASSES
from maize.serving.api import create_app
from maize.serving.pipeline import MaizePipeline
from tests.test_masking import synthetic_leaf


class FakeClassifier:
    """Returns fixed class probabilities and a 7x7x32 feature map."""
    def __init__(self, k=1, conf=0.9):
        self.k, self.conf = k, conf

    def __call__(self, x):
        p = np.full((1, 4), (1 - self.conf) / 3, np.float32)
        p[0, self.k] = self.conf
        feats = np.random.default_rng(0).random((1, 7, 7, 32)).astype(np.float32)
        return [feats, p]


class FakeAE:
    def __call__(self, x):
        return [np.clip(x * 0.9, 0, 1)]


def make_pipeline(**kw):
    cal = {"e_healthy_median": 0.001, "e_disease_p99": 0.05, "e_healthy_p95": 0.004,
           "tertiles": {d: [0.1, 0.5] for d in ("MLND", "MSV", "Blight")}}
    cfg = {"blur_min": 0.0, **kw.pop("cfg", {})}
    return MaizePipeline(kw.get("clf", FakeClassifier()), FakeAE(), np.random.default_rng(1).random((32, 4)), cal, cfg)


def jpeg(img):
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()


def test_pipeline_ok_diseased():
    img, _ = synthetic_leaf()
    r = make_pipeline().predict(jpeg(img))
    assert r["status"] == "ok" and r["disease"] == CLASSES[1]
    assert 0 <= r["severity"]["score"] <= 1 and r["severity"]["grade"]
    assert r["advice"]["action"] in ("monitor", "treat", "remove")
    assert r["cam_png"] and r["severity"]["heatmap_png"]


def test_pipeline_healthy_has_no_severity():
    img, _ = synthetic_leaf()
    r = make_pipeline(clf=FakeClassifier(k=0)).predict(jpeg(img))
    assert r["status"] == "ok" and r["disease"] == "Healthy" and r["severity"] is None


def test_pipeline_retakes():
    img, _ = synthetic_leaf()
    assert make_pipeline(clf=FakeClassifier(conf=0.4)).predict(jpeg(img))["status"] == "retake"      # unsure
    assert make_pipeline().predict(b"not an image")["status"] == "retake"                           # unreadable
    assert make_pipeline(cfg={"blur_min": 1e9}).predict(jpeg(img))["status"] == "retake"             # blurry
    plain = np.full((200, 200, 3), (120, 100, 80), np.uint8)                                          # no leaf
    assert make_pipeline().predict(jpeg(plain))["status"] == "retake"


def test_api_endpoints(tmp_path):
    img, _ = synthetic_leaf()
    with TestClient(create_app(make_pipeline(), static_dir=tmp_path / "none")) as c:
        assert c.get("/health").json()["models_loaded"] is True
        r = c.post("/api/predict", files={"file": ("leaf.jpg", jpeg(img), "image/jpeg")})
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_api_without_models_is_503_but_healthy(monkeypatch, tmp_path):
    monkeypatch.setenv("MAIZE_ARTIFACTS", str(tmp_path))  # empty folder
    with TestClient(create_app(static_dir=tmp_path / "none")) as c:
        assert c.get("/health").json()["models_loaded"] is False
        assert c.post("/api/predict", files={"file": ("a.jpg", b"x", "image/jpeg")}).status_code == 503
