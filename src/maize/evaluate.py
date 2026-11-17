"""Severity from the autoencoder (proposal 3.5.2 and 3.6): calibrate on val, report on test.

    score s(x) = clip((e - median healthy val error) / (p99 diseased val error - median healthy val error), 0, 1)
    healthy range: e <= p95 of healthy val errors
    grades: per disease, the tertiles of training scores split mild / moderate / severe

Reported on test, per disease: AUROC healthy vs disease, and agreement of the score with the
symptomatic area A(x) (severity.py) as Spearman rho and MAE after scaling both to 0-1. Two simple
baselines get the same Spearman, so the autoencoder has something to beat:
    classifier   1 - P(healthy) from the chosen classifier (if one was trained)
    colour       mean colour distance of the leaf pixels from the average healthy training leaf
Also: false-high rate on healthy leaves, score spread inside near-copy groups (stability), and per
class the share of leaf masks flagged as doubtful (leaf fraction < 0.10 or > 0.95).

    maize-evaluate ROOT MODELS [CLASSIFIER_DIR]     -> MODELS/ae/calibration.json, metrics.json, figures
        CLASSIFIER_DIR (e.g. MODELS/clf_mobilenet_v2) adds the classifier baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from maize.cache import load
from maize.pipeline import HEALTHY, command_args, data_version, load_usable_manifest, select
from maize.preprocess import AE_SIZE, CLS_SIZE, mask_background
from maize.severity import symptomatic_area

MASK_OK = (0.10, 0.95)  # leaf fraction outside this range: the mask is doubtful


# ---- the score -----------------------------------------------------------------------------------
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


# ---- measures ------------------------------------------------------------------------------------
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


def classifier_p_diseased(classifier_dir: Path, images: np.ndarray, chunk: int = 64) -> np.ndarray:
    """1 - P(healthy) per image from the exported classifier (the same ONNX file the web app runs)."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(classifier_dir / "classifier.onnx"), providers=["CPUExecutionProvider"])
    labels = json.loads((classifier_dir / "classifier.json").read_text())["labels"]
    name = session.get_inputs()[0].name
    probs = [session.run(None, {name: images[i:i + chunk].astype(np.float32)})[0] for i in range(0, len(images), chunk)]
    return 1 - np.concatenate(probs)[:, labels.index(HEALTHY)]


# ---- report --------------------------------------------------------------------------------------
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
    """Spearman vs A(x) of the simple baselines on one disease, to compare with spearman_vs_area above."""
    baselines = {"colour_distance": spearman(leaves["colour_distance"], leaves["area"])}
    if leaves["p_diseased"].notna().all():  # only when a classifier was trained
        baselines["classifier_1_minus_p_healthy"] = spearman(leaves["p_diseased"], leaves["area"])
    return baselines


def report(rows: pd.DataFrame, test: pd.DataFrame, cal: dict) -> dict:
    """rows: every usable image (error, score, leaf_fraction). test: the test rows with area and baselines."""
    healthy = test[test["label"] == HEALTHY]
    diseases = dict(list(test[test["label"] != HEALTHY].groupby("label")))
    doubtful = ~rows["leaf_fraction"].between(*MASK_OK)
    return {
        "false_high_rate": float((healthy["error"] > cal["e_healthy_p95"]).mean()),
        "within_group_score_sd": float(test.groupby("group")["score"].std().median()),
        "doubtful_mask_share": doubtful.groupby(rows["label"]).mean().to_dict(),
        "diseases": {d: disease_report(leaves, healthy) for d, leaves in diseases.items()},
        "baselines": {d: baseline_report(leaves) for d, leaves in diseases.items()},
    }


# ---- figures -------------------------------------------------------------------------------------
def figures(rows: pd.DataFrame, autoencoder, root: Path, out: Path) -> None:
    """error_by_class.png (test) and heatmaps.png (one test leaf per class: masked input, reconstruction,
    error inside the leaf)."""
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
        error_map = ((image / 255.0 - recon) ** 2).mean(axis=-1) * mask
        for ax, picture, title in zip(row, (image, recon, error_map), (f"{label} (masked)", "reconstruction", "error")):
            ax.imshow(picture, cmap="hot")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120)
    plt.close(fig)


# ---- the step ------------------------------------------------------------------------------------
def run(df: pd.DataFrame, root: Path, folder: Path, classifier_dir: Path | None = None) -> None:
    """Writes calibration.json (used by the web app), metrics.json, scores.csv and figures to folder."""
    from maize.autoencoder import Autoencoder

    autoencoder = Autoencoder.load(folder)
    rows = select(df).copy()
    images, masks = load(rows, root, AE_SIZE)
    rows["error"] = autoencoder.errors(images, masks)
    rows["leaf_fraction"] = masks.mean(axis=(1, 2))
    cal = calibrate(rows) | {"data_version": data_version(df)}
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
    results = report(rows, test, cal)

    (folder / "calibration.json").write_text(json.dumps(cal, indent=2))
    (folder / "metrics.json").write_text(json.dumps(results, indent=2))
    test[["path", "label", "group", "error", "score", "leaf_fraction", "area", "colour_distance", "p_diseased"]].to_csv(
        folder / "test_scores.csv", index=False)
    figures(rows, autoencoder, root, folder)
    print(json.dumps(results, indent=2))


def main():
    root, out, *classifier = command_args("maize-evaluate DATA_ROOT MODELS [CLASSIFIER_DIR]", required=2)
    classifier_dir = Path(classifier[0]) if classifier else None
    run(load_usable_manifest(Path(root)), Path(root), Path(out) / "ae", classifier_dir)
