"""Web service tests with a fake model (no torch, no onnxruntime, no model files needed)."""
import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from maize.serve import Scorer, create_app

CAL = {"e_healthy_median": 0.001, "e_healthy_p95": 0.004, "e_disease_p99": 0.05, "data_version": "abc123"}
CLASSES = {"labels": ["Healthy", "MLB", "MLND", "MSV"], "min_confidence": 0.6}


def jpeg(rgb):
    return cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()


def leaf(h=300, w=200, color=(40, 160, 50)):
    return np.full((h, w, 3), color, np.uint8)


def scorer(factor, probs=(0.1, 0.1, 0.1, 0.7)):
    def classify(x):
        assert x.shape == (1, 224, 224, 3) and x.dtype == np.float32
        return np.array([probs], np.float32)
    return Scorer(classify, lambda x: np.clip(x * factor, 0, 1).astype(np.float32), CLASSES, CAL)


def test_scorer_output_shape_and_ranges():
    r = scorer(0.9).score(leaf())
    assert set(r) == {"label", "confidence", "confident", "probabilities", "error", "score",
                      "within_healthy_range", "model_version"}
    assert r["label"] == "MSV" and r["confident"] and r["confidence"] == pytest.approx(0.7)
    assert 0 <= r["score"] <= 1 and r["error"] >= 0 and r["model_version"] == "abc123"


def test_low_confidence_is_flagged():
    assert not scorer(0.9, probs=(0.3, 0.2, 0.25, 0.25)).score(leaf())["confident"]


def test_worse_reconstruction_means_higher_score():
    img = leaf()
    good, bad = scorer(0.99).score(img), scorer(0.3).score(img)
    assert bad["error"] > good["error"] and bad["score"] >= good["score"]
    assert good["within_healthy_range"] and not bad["within_healthy_range"]


def test_api_happy_path_and_errors(tmp_path):
    with TestClient(create_app(scorer(0.9), static_dir=str(tmp_path / "none"))) as c:
        assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
        ok = c.post("/api/score", files={"file": ("a.jpg", jpeg(leaf()), "image/jpeg")})
        assert ok.status_code == 200 and 0 <= ok.json()["score"] <= 1
        assert c.post("/api/score", files={"file": ("a.jpg", b"not an image", "image/jpeg")}).status_code == 400


def test_starts_without_model_and_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MAIZE_MODEL_DIR", str(tmp_path))  # empty folder
    with TestClient(create_app(static_dir=str(tmp_path / "none"))) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/api/score", files={"file": ("a.jpg", b"x", "image/jpeg")}).status_code == 503


def test_serves_frontend_files(tmp_path):
    (tmp_path / "index.html").write_text("<h1>hi</h1>")
    with TestClient(create_app(scorer(0.9), static_dir=str(tmp_path))) as c:
        assert "hi" in c.get("/").text
        assert c.get("/health").status_code == 200  # API routes are not shadowed by the static mount
