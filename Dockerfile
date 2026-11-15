# Needs release/ (model.onnx, calibration.json, meta.json): create with `make release`.

# 1) compile the TypeScript frontend
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2) Python app + model
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MAIZE_MODEL_DIR=/app/model MAIZE_STATIC_DIR=/app/static
WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-deps .

COPY release/ model/
COPY --from=web /web/dist static/

RUN useradd -m app && chown -R app /app
USER app

# Render sets $PORT; locally pass -e PORT=8000
CMD ["sh", "-c", "uvicorn maize.serve:app --host 0.0.0.0 --port ${PORT:-10000}"]
