# Built and pushed to GHCR by .github/workflows/ci.yml, then deployed on Render.
# Model files come from release/ (`make release`). With none yet, the app starts and /api/score returns 503.

# 1) compile the TypeScript frontend
FROM node:26-slim AS web
WORKDIR /web
COPY frontend/package*.json frontend/.npmrc ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# 2) install the [serve] dependencies from pyproject.toml, at the exact versions + hashes in uv.lock
FROM python:3.13-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra serve

# 3) runtime: only the venv, the code, the model and the static files (no uv, no Node)
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/src \
    MAIZE_MODEL_DIR=/app/model MAIZE_STATIC_DIR=/app/static
WORKDIR /app
COPY --from=deps /app/.venv .venv
COPY src/ src/
COPY release/ model/
COPY --from=web /web/dist static/

RUN useradd -m app
USER app

# Render sets $PORT; locally pass -e PORT=8000
CMD ["sh", "-c", "exec uvicorn maize.serve:app --host 0.0.0.0 --port ${PORT:-10000}"]
