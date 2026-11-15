# RAW is the folder with <Class>/ subfolders of images.
RAW ?= data/raw
DATA ?= data
MODEL ?= models/ae

.PHONY: install data verify train eval release frontend serve docker test lock all

install:                             # everything except torch (Colab has it; add ,train elsewhere)
	pip install -e ".[data,serve]" --group dev

data:                                # clean, dedup and split new images (safe to re-run)
	maize-prepare --raw $(RAW) --data-root $(DATA)

verify:                              # fail if any near-copy crosses train/val
	maize-prepare --verify --data-root $(DATA)

train:                               # also exports model.onnx
	maize-train --data-root $(DATA) --out $(MODEL)

eval:                                # also writes calibration.json
	maize-evaluate --data-root $(DATA) --model-dir $(MODEL)

release:                             # the 3 small files the web app ships with (commit them)
	mkdir -p release
	cp $(MODEL)/model.onnx $(MODEL)/calibration.json $(MODEL)/meta.json release/

frontend:                            # compile TypeScript -> frontend/dist
	cd frontend && npm ci --ignore-scripts && npm run build

lock:                                # re-pin uv.lock from pyproject.toml (only releases >= 7 days old)
	uv lock --upgrade

serve: frontend                      # run the app locally on :8000 (needs the serve extra)
	MAIZE_MODEL_DIR=release MAIZE_STATIC_DIR=frontend/dist uvicorn maize.serve:app --port 8000

docker:
	docker build -t maize . && docker run --rm -p 8000:8000 -e PORT=8000 maize

test:
	pytest -q

all: data verify train eval release
