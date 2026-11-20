"""Severity from the autoencoder: calibrate the 0-1 score on val, report on test.

    score = clip((e - median healthy val error) / (p99 diseased val error - median healthy val error), 0, 1)

The score is checked against A(x), the share of the leaf in a symptom colour, and against two simple baselines
(1 - P(healthy) from the classifier, and colour distance from an average healthy leaf).

    maize-evaluate ROOT MODELS [CLASSIFIER_DIR]     -> calibration.json, metrics.json, figures
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from maize.pipeline import HEALTHY, command_args, data_version, load, load_usable_manifest, select
from maize.preprocess import AE_SIZE, CLS_SIZE, ERRORS, error_map, mask_background, usable_masks

# Symptom colours for A(x), fixed before any score was seen. OpenCV HSV: hue 0-179, healthy green ~35-85.
SYMPTOM_RANGES = {
    "yellowing": ((20, 60, 80), (34, 255, 255)),
    "brown lesion": ((5, 50, 30), (19, 255, 200)),
    "dead tissue": ((10, 0, 120), (30, 60, 255)),
}


#Symptomatic area A(x)
def symptom_pixels(rgb: np.ndarray) -> np.ndarray:
    """True where a pixel falls in any symptom colour range. rgb: H x W x 3 uint8."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = np.zeros(rgb.shape[:2], bool)
    for low, high in SYMPTOM_RANGES.values():
        found |= cv2.inRange(hsv, low, high) > 0
    return found


def symptomatic_area(rgb: np.ndarray, mask: np.ndarray) -> float:
    """A(x) in percent (0-100): symptom pixels inside the leaf / leaf pixels."""
    leaf = mask > 0
    return float(100 * (symptom_pixels(rgb) & leaf).sum() / max(leaf.sum(), 1))


#Score
def score(errors: pd.Series, cal: dict) -> pd.Series:
    healthy, disease = cal["e_healthy_median"], cal["e_disease_p99"]
    return ((errors - healthy) / (disease - healthy)).clip(0, 1)


def calibrate(rows: pd.DataFrame) -> dict:
    """rows: usable images with label, split and error. Uses val for the scale, train for the grades."""
    healthy = rows["label"] == HEALTHY
    healthy_val = rows.loc[healthy & (rows["split"] == "val"), "error"]
    disease_val = rows.loc[~healthy & (rows["split"] == "val"), "error"]
    cal = {"e_healthy_median": healthy_val.median(), "e_healthy_p95": healthy_val.quantile(0.95),
           "e_disease_p99": disease_val.quantile(0.99)}
    if cal["e_disease_p99"] <= cal["e_healthy_median"]:
        raise SystemExit("Diseased leaves do not reconstruct worse than healthy ones: the autoencoder is not usable.")

    disease_train = rows[~healthy & (rows["split"] == "train")]
    tertiles = score(disease_train["error"], cal).groupby(disease_train["label"]).quantile([1 / 3, 2 / 3])
    cal["grades"] = {d: tertiles[d].tolist() for d in disease_train["label"].unique()}
    return cal


#Measures
def auroc(healthy: pd.Series, diseased: pd.Series) -> float:
    """Chance that a diseased leaf has a higher value than a healthy one (Mann-Whitney U / n1 n2)."""
    return float(mannwhitneyu(diseased, healthy).statistic / (len(diseased) * len(healthy)))


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(spearmanr(a, b).statistic)


def scaled_mae(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute difference after min-max scaling both to 0-1."""
    def to_unit(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-12)
    return float((to_unit(a) - to_unit(b)).abs().mean())


def mean_leaf_colour(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Average RGB over all leaf pixels of these images."""
    leaf = masks.astype(bool)
    return images[leaf].mean(axis=0)


def colour_distance(images: np.ndarray, masks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per image: mean distance (RGB, 0-255) of its leaf pixels from the reference colour."""
    distances = np.linalg.norm(images.astype(np.float32) - reference, axis=-1) * masks
    return distances.sum(axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1)


def onnx_batch_size(session, default: int = 64) -> int:
    """Images per call: some exports accept only a fixed number (often 1), others any number."""
    first = session.get_inputs()[0].shape[0]
    return first if isinstance(first, int) else default


def classifier_p_diseased(classifier_dir: Path, images: np.ndarray) -> np.ndarray:
    """1 - P(healthy) per image from the exported classifier."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(classifier_dir / "classifier.onnx"), providers=["CPUExecutionProvider"])
    labels = json.loads((classifier_dir / "classifier.json").read_text())["labels"]
    name, chunk = session.get_inputs()[0].name, onnx_batch_size(session)
    probs = [session.run(None, {name: images[i:i + chunk].astype(np.float32)})[0] for i in range(0, len(images), chunk)]
    return 1 - np.concatenate(probs)[:, labels.index(HEALTHY)]


#Report
def disease_report(leaves: pd.DataFrame, healthy: pd.DataFrame) -> dict:
    """The autoencoder score on one disease. leaves / healthy: test rows with error, score and area."""
    return {
        "n": len(leaves),
        "auroc_vs_healthy": auroc(healthy["error"], leaves["error"]),
        "median_score": float(leaves["score"].median()),
        "spearman_vs_area": spearman(leaves["score"], leaves["area"]),
        "mae_vs_area_scaled": scaled_mae(leaves["score"], leaves["area"]),
    }


def baseline_report(leaves: pd.DataFrame) -> dict:
    """Spearman vs A(x) of the simple baselines on one disease."""
    baselines = {"colour_distance": spearman(leaves["colour_distance"], leaves["area"])}
    if leaves["p_diseased"].notna().all():
        baselines["classifier_1_minus_p_healthy"] = spearman(leaves["p_diseased"], leaves["area"])
    return baselines


def report(test: pd.DataFrame, cal: dict, doubtful_mask_share: dict) -> dict:
    """test: the test rows with a usable mask, with error, score, area and baselines."""
    healthy = test[test["label"] == HEALTHY]
    diseases = dict(list(test[test["label"] != HEALTHY].groupby("label")))
    return {
        "error": cal["error"],
        "false_high_rate": float((healthy["error"] > cal["e_healthy_p95"]).mean()),
        # raw error, not the score: the score is 0 for most healthy leaves
        "within_group_error_sd": float(test.groupby("group")["error"].std().median()),
        "doubtful_mask_share": doubtful_mask_share,
        "diseases": {d: disease_report(leaves, healthy) for d, leaves in diseases.items()},
        "baselines": {d: baseline_report(leaves) for d, leaves in diseases.items()},
    }


#Figures
def figures(rows: pd.DataFrame, autoencoder, root: Path, out: Path, sigma: float) -> None:
    """error_by_class.png and heatmaps.png (one test leaf per class)."""
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    test = rows[rows["split"] == "test"]

    fig, ax = plt.subplots(figsize=(7, 5))
    classes = sorted(test["label"].unique())
    ax.boxplot([test.loc[test["label"] == c, "error"] for c in classes], tick_labels=classes)
    ax.set(title="Reconstruction error on test leaves", ylabel="error e(x)")
    fig.savefig(out / "error_by_class.png", dpi=120)
    plt.close(fig)

    examples = test.groupby("label").sample(1, random_state=0)
    images, masks = load(examples, root, AE_SIZE)
    inputs = mask_background(images, masks)
    recons = autoencoder.reconstruct(images, masks)
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)), squeeze=False)
    for row, image, recon, mask, label in zip(axes, inputs, recons, masks, examples["label"]):
        error = error_map(image[None], recon[None], mask[None], sigma)[0]
        for ax, picture, title in zip(row, (image, recon, error), (f"{label} (masked)", "reconstruction", "error")):
            ax.imshow(picture, cmap="hot")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120)
    plt.close(fig)


#Evaluation
def run(df: pd.DataFrame, root: Path, folder: Path, error: str = "plain", classifier_dir: Path | None = None) -> None:
    """Writes calibration.json, metrics.json, test_scores.csv and figures to folder."""
    from maize.autoencoder import Autoencoder

    autoencoder = Autoencoder.load(folder)
    sigma = ERRORS[error]
    rows = select(df).copy()
    images, masks = load(rows, root, AE_SIZE)
    ok = usable_masks(masks)
    doubtful_mask_share = (~pd.Series(ok, index=rows.index)).groupby(rows["label"]).mean().to_dict()
    rows, images, masks = rows[ok].copy(), images[ok], masks[ok]

    rows["error"] = autoencoder.errors(images, masks, sigma)
    rows["leaf_fraction"] = masks.mean(axis=(1, 2))
    cal = calibrate(rows) | {"error": error, "smooth_sigma": sigma, "data_version": data_version(df)}
    rows["score"] = score(rows["error"], cal)

    healthy_train = ((rows["label"] == HEALTHY) & (rows["split"] == "train")).to_numpy()
    is_test = (rows["split"] == "test").to_numpy()
    test = rows[is_test].copy()
    test["colour_distance"] = colour_distance(images[is_test], masks[is_test],
                                              mean_leaf_colour(images[healthy_train], masks[healthy_train]))
    del images, masks

    test_images, test_masks = load(test, root, CLS_SIZE)  # A(x) and the classifier work at 224 px
    test["area"] = [symptomatic_area(image, mask) for image, mask in zip(test_images, test_masks)]
    test["p_diseased"] = classifier_p_diseased(classifier_dir, test_images) if classifier_dir else np.nan
    results = report(test, cal, doubtful_mask_share)

    (folder / "calibration.json").write_text(json.dumps(cal, indent=2))
    (folder / "metrics.json").write_text(json.dumps(results, indent=2))
    test[["path", "label", "group", "error", "score", "leaf_fraction", "area", "colour_distance", "p_diseased"]].to_csv(
        folder / "test_scores.csv", index=False)
    figures(rows, autoencoder, root, folder, sigma)
    print(json.dumps(results, indent=2))


def main():
    from maize.autoencoder import chosen

    root, models, *classifier = command_args("maize-evaluate DATA_ROOT MODELS [CLASSIFIER_DIR]", required=2)
    classifier_dir = Path(classifier[0]) if classifier else None
    folder, error = chosen(Path(models))
    print(f"Evaluating {folder.name} with the {error} error: the pair that separates healthy from diseased "
          "val leaves best.", flush=True)
    run(load_usable_manifest(Path(root)), Path(root), folder, error, classifier_dir)
