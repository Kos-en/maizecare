# MaizeCare: maize leaf disease and severity

Two Keras models behind one web app for farmers (English and Kiswahili), following the proposal:
1. **Classifier** (ImageNet backbone, transfer learning): Healthy / MLB / MLND / MSV, with a confidence so
   unclear photos get "please retake" instead of a guess.
2. **Autoencoder** trained on healthy leaves only, inside the leaf mask. Its reconstruction error on a
   diseased leaf is the severity signal, so no severity labels are needed.

```
src/maize/
  preprocess.py   one photo -> model input + leaf mask; shared by training and the web app
  pipeline.py     the data pipeline (DataPipeline), the manifest and the image cache   (no torch)
  classifier.py   stage 1: Keras CNN, 4 backbones compared, exported to ONNX
  autoencoder.py  stage 2: masked Keras autoencoder, exported to ONNX
  evaluate.py     severity score calibrated on val, checked on test against A(x)
  serve.py        FastAPI app: /api/score + the built frontend                         (no torch)
frontend/         plain TypeScript UI, compiled to static files
```

## The data pipeline: `maize-prepare`

Every image goes through this before any training, and new photos get exactly the same treatment.

```bash
maize-prepare RAW ROOT       # RAW/<Class>/*.jpg; re-run after adding photos
```
1. **File check**: opens as JPEG/PNG and the shortest side is at least 50 px, else `bad_image` / `too_small`.
2. **Near copies**: within 6 bits on **both** pHash and aHash. Chains of near copies form one group.
   (Either hash alone chained 18,624 unrelated photos into one group on the real data.)
3. **Mixed labels**: a group holding two classes is dropped whole (`label_conflict`).
4. **Exact copies**: same file contents, keep one (`duplicate_exact`).
5. **Split by group, stratified by class**: within each class the groups are put in a fixed (hash) order
   and filled into test, val and train until each holds 15% / 15% / 70% of that class. Near copies can
   never be on both sides; a leak check confirms it. The report also lists the largest groups.
6. **Resize and leaf mask** (proposal 3.4): scale the longest side to 128 / 224 px and pad with black;
   find the leaf with Excess Green + Otsu, add yellow/brown tissue touching it, close, fill holes, keep
   the largest region. Images and masks are cached in `ROOT/cache_<size>.npz` and `ROOT/mask_<size>.npz`.

Only new files are read (the slow part); steps 2-5 are redone on the whole dataset every run, so the
result does not depend on the order images arrived. Adding photos can move groups between splits, so
retrain after adding data. Unusable images stay in the manifest with a status.

## Training and evaluation

```bash
maize-train-classifier  ROOT MODELS                  # all 4 backbones, or add e.g. mobilenet_v2
maize-train-autoencoder ROOT MODELS                  # healthy train leaves, inside the leaf mask
maize-evaluate          ROOT MODELS MODELS/clf_<best backbone>
```
- **Classifier**: head, then fine-tune; class weights; blur / noise / brightness augmentation (no hue);
  keeps the epoch with the best val macro-F1. `MODELS/classifiers.csv` compares the backbones: pick the
  best on val, report its test scores.
- **Autoencoder**: 4 stride-2 blocks (32 -> 256 filters) and a 1x1 bottleneck to 32 channels (24x
  compression); background set to 0 and ignored by the loss.
- **Evaluate**: score `s = clip((e - healthy median) / (diseased p99 - healthy median), 0, 1)` on val,
  grades from per-disease tertiles. On test, per disease: AUROC vs healthy, Spearman and scaled MAE of
  the score against A(x), and the same Spearman for two baselines (1 - P(healthy) from the classifier,
  colour distance from the average healthy leaf). Writes `calibration.json`, `metrics.json`,
  `test_scores.csv`, `error_by_class.png`, `heatmaps.png` to `MODELS/ae/`.

Each command prints its usage line if run without arguments. The commands are declared in
`pyproject.toml` (`[project.scripts]`), the same way Superset declares `superset`.

## Colab

Data, caches and models live on Drive, so a dead session loses nothing.

```bash
cd /content && git clone -b mask-and-severity https://github.com/Kos-en/maizecare repo && cd repo
pip install -q --no-deps -e . && pip install -q imagehash onnx onnxscript onnxruntime
export KERAS_BACKEND=torch D=/content/drive/MyDrive M=/content/drive/MyDrive/maize_models
maize-prepare $D/cv_datasets_cleaned $D                              # CPU is enough
nohup sh -c "maize-train-classifier $D $M && maize-train-autoencoder $D $M" > $D/train.log 2>&1 &   # GPU (L4)
maize-evaluate $D $M $M/clf_<best backbone>          # after reading $M/classifiers.csv
```

## Web app

```
make release CLF=clf_<best backbone>   # copies the 4 files the app needs into release/ (commit them)
make serve         # builds the frontend, serves http://localhost:8000
make docker        # the same in the production container
```
Both models are exported to ONNX at train time and served with `onnxruntime`, so the image has no
torch, TensorFlow or Keras and fits Render's free 512 MB. `/api/score` returns the predicted label,
its confidence (`confident` is false below `min_confidence` in `classifier.json`, default 0.6), and a
0-1 severity score `(e - median healthy) / (p99 diseased - median healthy)`. The UI shows Healthy for a
healthy label, otherwise grades the disease mild / moderate / severe from the score. All UI text is in
`frontend/public/i18n/{en,sw}.json`; the Kiswahili still needs a review by a native speaker and the
advice by an extension officer. Heat maps are a developer tool (`maize-evaluate`, heatmaps.png), not shown to users; production
monitoring is planned in [docs/observability.md](docs/observability.md).

## Dependencies, CI and deployment

`pyproject.toml` is the only place dependencies are declared, split by use: the base (numpy, OpenCV),
`data`, `train` and `serve` extras, and a `dev` group. `uv.lock` pins every package to an exact version
and hash; `make lock` refreshes it (only releases at least 7 days old). The Docker image installs just
the `serve` extra from the lock, so production runs exactly what CI tested.

`.github/workflows/ci.yml` on every push and PR: ruff, pytest, `pip-audit` on the shipped dependencies,
`npm audit` and the frontend build. On a push to `main` it then builds the image, pushes it to GitHub's
registry (`ghcr.io/kos-en/maizecare`, tagged `latest` and the commit SHA) and tells Render to deploy that
exact image. Dependabot opens weekly update PRs.

One-time setup on Render (free plan):
1. Push to `main` once so the image exists in GHCR, then make the package public
   (GitHub -> Packages -> maize -> Package settings), or add GHCR credentials in Render.
2. Render -> New -> Web Service -> Existing image -> `ghcr.io/kos-en/maizecare:latest`, plan Free,
   health check path `/health` (or put the URL in `render.yaml` and use New -> Blueprint).
3. Copy the service's Deploy Hook URL (Settings) into the GitHub repo secret `RENDER_DEPLOY_HOOK_URL`.
Free instances sleep after 15 min idle (first request takes about a minute). Photos are never stored.

## Not built yet
From the proposal: VAE and SSIM-loss autoencoder variants, 5-fold cross-validation, blurred / low-light
test copies, Grad-CAM, TFLite quantisation, IoU of the leaf mask against hand-drawn masks, and the
monitoring in docs/observability.md.
