# RAW is the folder with <Class>/ subfolders of images.
RAW ?= data/raw
DATA ?= data
MODEL ?= models/ae

.PHONY: install data verify train eval release frontend serve docker test all

install:
	pip install -e ".[dev]"          # add ,train if torch is not installed

data:                                # clean, dedup and split new images (safe to re-run)
	python -m maize.prepare --raw $(RAW) --data-root $(DATA)

verify:                              # fail if any near-copy crosses train/val
	python -m maize.prepare --verify --data-root $(DATA)

train:                               # also exports model.onnx
	python -m maize.train --data-root $(DATA) --out $(MODEL)

eval:                                # also writes calibration.json
	python -m maize.evaluate --data-root $(DATA) --model-dir $(MODEL)

release:                             # the 3 small files the web app ships with (commit them)
	mkdir -p release
	cp $(MODEL)/model.onnx $(MODEL)/calibration.json $(MODEL)/meta.json release/

frontend:                            # compile TypeScript -> frontend/dist
	cd frontend && npm ci && npm run build

serve: frontend                      # run the app locally on :8000 (needs `pip install -e ".[serve]"`)
	MAIZE_MODEL_DIR=release MAIZE_STATIC_DIR=frontend/dist uvicorn maize.serve:app --port 8000

docker:
	docker build -t maize . && docker run --rm -p 8000:8000 -e PORT=8000 maize

test:
	pytest -q

all: data verify train eval release
