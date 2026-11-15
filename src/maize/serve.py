"""FastAPI app: classifies a leaf photo, grades its severity, and serves the built frontend.

    uvicorn maize.serve:app --port 8000

GET /health, POST /api/score (image -> disease label + confidence, 0-1 severity), / for the frontend.
Stage 1 is the Keras classifier (classifier.onnx), stage 2 the healthy-only autoencoder (model.onnx).
Photos are processed in memory, never stored. With no model files the app still starts
and /api/score returns 503, so a first deploy works. Heat maps are for developers only
(maize-evaluate); see docs/observability.md.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .preprocess import CLS_SIZE, decode, resize, to_input

log = logging.getLogger("maize.serve")
MAX_UPLOAD = 10 * 1024 * 1024
MODEL_FILES = ("classifier.onnx", "classifier.json", "model.onnx", "calibration.json")

Run = Callable[[np.ndarray], np.ndarray]


def onnx_runner(path: Path) -> Run:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0]


class Scorer:
    """classify: (1,224,224,3) RGB 0-255 -> class probabilities.
    reconstruct: (1,3,128,128) in [0,1] -> its reconstruction by the healthy-only autoencoder."""

    def __init__(self, classify: Run, reconstruct: Run, classes: dict, calibration: dict):
        self.classify, self.reconstruct = classify, reconstruct
        self.labels, self.min_conf = classes["labels"], classes["min_confidence"]
        self.cal = calibration

    @classmethod
    def from_dir(cls, d: Path) -> "Scorer":
        return cls(onnx_runner(d / "classifier.onnx"), onnx_runner(d / "model.onnx"),
                   json.loads((d / "classifier.json").read_text()), json.loads((d / "calibration.json").read_text()))

    def score(self, rgb: np.ndarray) -> dict:
        probs = self.classify(resize(rgb, CLS_SIZE)[None].astype(np.float32))[0]
        top = int(probs.argmax())
        x = to_input(rgb)[None]
        e = float(((x - self.reconstruct(x)) ** 2).mean())
        med, p99 = self.cal["e_healthy_median"], self.cal["e_disease_p99"]
        return {
            "label": self.labels[top],
            "confidence": float(probs[top]),
            "confident": bool(probs[top] >= self.min_conf),
            "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.labels, probs)},
            "error": e,
            "score": float(min(1.0, max(0.0, (e - med) / (p99 - med)))),
            "within_healthy_range": bool(e <= self.cal["e_healthy_p95"]),
            "model_version": self.cal.get("data_version"),
        }


def load_scorer() -> Scorer | None:
    d = Path(os.environ.get("MAIZE_MODEL_DIR", "model"))
    missing = [f for f in MODEL_FILES if not (d / f).exists()]
    if missing:
        log.warning("Missing %s in %s: /api/score will return 503", missing, d)
        return None
    return Scorer.from_dir(d)


def create_app(scorer: Scorer | None = None, static_dir: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = scorer if scorer is not None else load_scorer()
        yield

    app = FastAPI(title="Maize leaf severity", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.scorer is not None}

    @app.post("/api/score")
    async def score(file: UploadFile = File(...)):
        s = app.state.scorer
        if s is None:
            raise HTTPException(503, "The model is not loaded on this server.")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image too large (max 10 MB).")
        rgb = decode(data)
        if rgb is None:
            raise HTTPException(400, "Could not read that image. Please upload a JPEG or PNG photo.")
        return await run_in_threadpool(s.score, rgb)

    static = Path(static_dir or os.environ.get("MAIZE_STATIC_DIR", "static"))
    if static.is_dir():
        app.mount("/", StaticFiles(directory=str(static), html=True), name="static")
    return app


app = create_app()
