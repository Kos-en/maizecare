"""FastAPI app: scores a leaf photo with the ONNX model and serves the built frontend.

    uvicorn maize.serve:app --port 8000

GET /health, POST /api/score (image -> error, 0-1 score), and / for the frontend.
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

from .preprocess import decode, to_input

log = logging.getLogger("maize.serve")
MAX_UPLOAD = 10 * 1024 * 1024


class Scorer:
    """run(x) maps a (1,3,128,128) float32 batch to its reconstruction (same shape)."""

    def __init__(self, run: Callable[[np.ndarray], np.ndarray], calibration: dict):
        self.run, self.cal = run, calibration

    @classmethod
    def from_dir(cls, model_dir: Path) -> "Scorer":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        cal = json.loads((model_dir / "calibration.json").read_text())
        return cls(lambda x: sess.run(None, {name: x})[0], cal)

    def score(self, rgb: np.ndarray) -> dict:
        x = to_input(rgb)[None]
        e = float(((x - self.run(x)) ** 2).mean())
        med, p99 = self.cal["e_healthy_median"], self.cal["e_disease_p99"]
        return {
            "error": e,
            "score": float(min(1.0, max(0.0, (e - med) / (p99 - med)))),
            "within_healthy_range": bool(e <= self.cal["e_healthy_p95"]),
            "model_version": self.cal.get("data_version"),
        }


def load_scorer() -> Scorer | None:
    d = Path(os.environ.get("MAIZE_MODEL_DIR", "model"))
    if not (d / "model.onnx").exists() or not (d / "calibration.json").exists():
        log.warning("No model in %s: /api/score will return 503", d)
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
