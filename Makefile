# RAW is the folder with one subfolder of images per class (Healthy/ MLB/ MLND/ MSV/).
RAW ?= data/raw
DATA ?= data
MODELS ?= models
CLF ?= clf_mobilenet_v2

.PHONY: install data release frontend serve docker test lock

install:                             # everything except torch (Colab has it; add ,train elsewhere)
	pip install -e ".[data,serve]" --group dev

data:                                # the data pipeline: check, group, split, cache images + leaf masks
	maize prepare --raw $(RAW) --data-root $(DATA)

release:                             # copy the 4 files the web app ships with (commit them)
	mkdir -p release
	cp $(MODELS)/$(CLF)/classifier.onnx $(MODELS)/$(CLF)/classifier.json release/
	cp $(MODELS)/ae/model.onnx $(MODELS)/ae/calibration.json release/

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
