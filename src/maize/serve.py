"""Web service: FastAPI + the built TypeScript frontend, served from one process.

    uvicorn maize.serve:app --port 8000        # model in ./model, frontend in ./static

  GET  /health       liveness + whether the model is loaded (Render health check)
  POST /api/score    multipart image -> reconstruction error, 0-1 severity score, heat map
  GET  /             the frontend (static/), mounted last so it never shadows the API

Torch-free: runs the exported ONNX model with onnxruntime. Photos are processed in memory and
never stored. Without a model the app still starts (so a first deploy works); /api/score
then answers 503.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .preprocess import decode, to_input

log = logging.getLogger("maize.serve")
MAX_UPLOAD = 10 * 1024 * 1024
DISPLAY = 256  # heat map overlay size shown to the user


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
        recon = self.run(x)
        err_map = ((x[0] - recon[0]) ** 2).mean(axis=0)
        e = float(err_map.mean())
        med, p99 = self.cal["e_healthy_median"], self.cal["e_disease_p99"]
        heat = err_map / max(float(err_map.max()), 1e-8)
        color = cv2.applyColorMap((cv2.resize(heat, (DISPLAY, DISPLAY)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        base = cv2.cvtColor(cv2.resize(rgb, (DISPLAY, DISPLAY)), cv2.COLOR_RGB2BGR)
        ok, png = cv2.imencode(".png", cv2.addWeighted(base, 0.55, color, 0.45, 0))
        return {
            "error": e,
            "score": float(min(1.0, max(0.0, (e - med) / (p99 - med)))),
            "within_healthy_range": bool(e <= self.cal["e_healthy_p95"]),
            "heatmap_png": base64.b64encode(png.tobytes()).decode(),
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

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        log.info("%s %s -> %s in %.0f ms", request.method, request.url.path, resp.status_code,
                 (time.perf_counter() - t0) * 1000)
        return resp

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
