# One command per stage. Set RAW to the folder with <Class>/ subfolders of images.
RAW ?= data/raw
DATA ?= data
MODEL ?= models/ae

.PHONY: install data verify train eval test all

install:
	pip install -e ".[dev]"          # add ,train if torch is not already installed

data:                                # clean + dedup + split NEW images (safe to re-run any time)
	python -m maize.prepare --raw $(RAW) --data-root $(DATA)

verify:                              # fail if any near-copy crosses train/val
	python -m maize.prepare --verify --data-root $(DATA)

train:
	python -m maize.train --data-root $(DATA) --out $(MODEL)

eval:
	python -m maize.evaluate --data-root $(DATA) --model-dir $(MODEL)

test:
	pytest -q

all: data verify train eval
