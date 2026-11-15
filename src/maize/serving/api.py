"""FastAPI app: /health, /api/predict, and the built React front end (static/) at /.

Starts even when no models are present (so a first Render deploy works); /api/predict
then answers 503 until artifacts are added.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import __version__

log = logging.getLogger("maize.api")
MAX_UPLOAD = 10 * 1024 * 1024


def load_pipeline():
    d = Path(os.environ.get("MAIZE_ARTIFACTS", "artifacts"))
    if not (d / "pipeline_config.json").exists():
        log.warning("No models found in %s: /api/predict will return 503", d)
        return None
    from .pipeline import MaizePipeline
    return MaizePipeline.from_dir(d)


def create_app(pipeline=None, static_dir: str | Path = "static") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline if pipeline is not None else load_pipeline()
        yield

    app = FastAPI(title="Maize disease + severity", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        log.info("%s %s -> %s in %.0f ms", request.method, request.url.path, resp.status_code,
                 (time.perf_counter() - t0) * 1000)
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": app.state.pipeline is not None, "version": __version__}

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)):
        pipe = app.state.pipeline
        if pipe is None:
            raise HTTPException(503, "Models are not loaded on this server.")
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "Image too large (max 10 MB).")
        # photos are processed in memory and never stored
        return await run_in_threadpool(pipe.predict, data)

    if Path(static_dir).is_dir():  # mounted last so it does not shadow the API routes
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


app = create_app()
