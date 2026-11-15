# Maize leaf disease + severity (two-stage)

Stage 1: classify Healthy / MLND / MSV / Blight with pretrained CNNs (ResNet50 reference vs
MobileNetV2, MobileNetV3-Small, EfficientNetB0). Stage 2: an autoencoder trained on healthy
leaves; its masked reconstruction error becomes a label-free severity score.
Training runs on Colab/conda with Python **scripts**; serving is a **TFLite** FastAPI app on **Render**.

## Setup (conda)

```bash
conda activate "ml graduate"          # your env
pip install -r requirements/train.txt # skip tensorflow if the env already has it (>=2.16, Keras 3)
pytest                                # TF-free tests run anywhere; TF tests skip if TF is missing
```

Set where data and runs live (defaults `./data`, `./runs`):
`export MAIZE_DATA_ROOT=...  MAIZE_RUNS_ROOT=...` (on Colab: data on `/content`, runs on Drive).

## Pipeline (run in order)

| # | Script | What it does | Where |
|---|--------|--------------|-------|
| 1 | `01_build_manifest.py` | scan folders, quality filter, blur/brightness, aHash+pHash → `manifest.csv` | CPU |
| 2 | `02_eda.py [--tsne]` | balance, size/light/blur by class+source, near-duplicates, t-SNE | CPU |
| 3 | `03_split.py [--inspect]` | tune Hamming cut-off by eye, groups, test set + StratifiedGroupKFold(5), leak check | CPU |
| 4 | `04_preprocess.py`, `04b_mask_check.py` | leaf mask + 224px cache; IoU vs ~100 hand masks (+ blurred/dark copies) | CPU |
| 5 | `05_train_classifier.py` | 2-phase training, one backbone, folds resumable | GPU |
| 6 | `06_train_ae.py` | CAE/VAE (MSE or MSE+SSIM) on healthy training folds | GPU |
| 7 | `07_calibrate.py` | severity scaling (Eq 3.2), tertile grades, retake threshold | GPU/CPU |
| 8 | `08_evaluate.py clf\|sev` | test-set metrics, robustness, Spearman vs symptomatic area | GPU/CPU |
| 9 | `09_export.py` | TFLite int8/fp16 + release folder + before/after report | GPU/CPU |

Every script is resumable and takes `--set key=value` overrides, e.g.
`python scripts/05_train_classifier.py --config configs/classifier/mobilenetv2.yaml --fold all --set masked=false`.

Model selection: compare backbones by mean±sd macro-F1 over the 5 folds (masked vs raw), evaluate
the fold models once on the test set (`08`), export the chosen fold's model (`09`).

## Things that must not be skipped

- **Freeze the symptom colour ranges** in `src/maize/symptom_area.py` *before* looking at any
  autoencoder score, then rename `SYMPTOM_VERSION` to `1.0.0` and commit (the proposal promises this).
- **One masking module** (`maize/masking.py` + `maize/preprocess.py`) is used by training and serving.
  Change it → bump `MASK_VERSION` → rerun `04_preprocess.py --force` and retrain. The API refuses to
  start if the model's mask version differs from the code's.
- **Check source confounding in EDA**: if one collection holds all the Healthy images, the classifier can
  learn the source. `02_eda.py` prints stats by source + class.
- **Advice text** in `serving/advice.py` is placeholder wording; agree it with an extension officer.
- Set `blur_min` etc. in `pipeline_config.json` from the EDA percentiles of sharp photos.

## Serving on Render (free, no card)

```bash
python scripts/09_export.py --clf-run runs/clf/<model>/fold0 --ae-run runs/ae/<ae>/fold0 --out artifacts/release-lite
git add artifacts/release-lite && git commit -m "release models" && git push
```
Render → New → Blueprint → select the repo (`render.yaml`). The free instance has ~512 MB RAM, sleeps after
15 min idle (first request ≈ 1 min), and 0.1 CPU, so expect 1-3 s per photo.
The image has no TensorFlow: classifier heat map = **CAM** (`features @ head weights`), exact for the
GAP→Dense head; `evaluate/gradcam.py` checks CAM vs real Grad-CAM in the training env.

Local check of the image:
```bash
docker build -f docker/Dockerfile.lite -t maize-lite .
docker run --rm -p 8000:8000 -e PORT=8000 maize-lite     # http://localhost:8000
```
Frontend dev: `cd frontend && npm install && npm run dev` (proxies to uvicorn on :8000).

## Colab

`colab/launcher.ipynb` only mounts Drive, clones the repo, sets `MAIZE_*` env vars and calls the scripts.
