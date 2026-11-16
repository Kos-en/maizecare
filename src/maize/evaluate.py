"""Severity from the autoencoder (proposal 3.5.2 and 3.6): calibrate on val, report on test.

    score s(x) = clip((e - median healthy val error) / (p99 diseased val error - median healthy val error), 0, 1)
    healthy range: e <= p95 of healthy val errors
    grades: per disease, the tertiles of training scores split mild / moderate / severe

Reported on test: AUROC healthy vs each disease, false-high rate on healthy leaves, Spearman between
the error and the colour reference index (severity.py), and how much the score varies inside a
near-copy group (stability).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .cache import load_images
from .data import HEALTHY, data_version, select
from .preprocess import AE_SIZE
from .severity import extract_quick_features, formula_severity


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


def auroc(healthy: pd.Series, diseased: pd.Series) -> float:
    """Chance that a diseased leaf has a higher error than a healthy one (Mann-Whitney U / n1 n2)."""
    return float(mannwhitneyu(diseased, healthy).statistic / (len(diseased) * len(healthy)))


def report(rows: pd.DataFrame, cal: dict, root: Path, reference_n: int) -> dict:
    test = rows[rows["split"] == "test"]
    healthy = test[test["label"] == HEALTHY]
    results = {
        "false_high_rate": float((healthy["error"] > cal["e_healthy_p95"]).mean()),
        "within_group_score_sd": float(test.groupby("group")["score"].std().median()),
    }
    for disease, leaves in test[test["label"] != HEALTHY].groupby("label"):
        sample = leaves.sample(min(reference_n, len(leaves)), random_state=0)
        reference = [formula_severity(extract_quick_features(Path(root) / p)) for p in sample["path"]]
        results[disease] = {
            "n": len(leaves),
            "auroc_vs_healthy": auroc(healthy["error"], leaves["error"]),
            "median_score": float(leaves["score"].median()),
            "spearman_vs_reference": float(spearmanr(sample["error"], reference).statistic),
        }
    return results


def figures(rows: pd.DataFrame, model, root: Path, out: Path) -> None:
    """error_by_class.png (test) and heatmaps.png (one test leaf per class: photo, reconstruction, error)."""
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
    images = load_images(examples, root, AE_SIZE)
    recons = model.predict(images.astype("float32"), verbose=0)
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)), squeeze=False)
    for row, image, recon, label in zip(axes, images, recons, examples["label"]):
        error_map = ((image / 255.0 - recon) ** 2).mean(axis=-1)
        for ax, picture, title in zip(row, (image, recon, error_map), (label, "reconstruction", "error")):
            ax.imshow(picture, cmap="hot")
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=120)
    plt.close(fig)


def run(df: pd.DataFrame, root: Path, folder: Path, reference_n: int = 300) -> None:
    """Writes calibration.json (used by the web app), metrics.json, scores.csv and figures to folder."""
    from .models import load_autoencoder, reconstruction_errors

    model = load_autoencoder(folder)
    rows = select(df).copy()
    rows["error"] = reconstruction_errors(model, load_images(rows, root, AE_SIZE))
    cal = calibrate(rows) | {"data_version": data_version(df)}
    rows["score"] = score(rows["error"], cal)
    results = report(rows, cal, root, reference_n)

    (folder / "calibration.json").write_text(json.dumps(cal, indent=2))
    (folder / "metrics.json").write_text(json.dumps(results, indent=2))
    rows[["path", "label", "split", "group", "error", "score"]].to_csv(folder / "scores.csv", index=False)
    figures(rows, model, root, folder)
    print(json.dumps(results, indent=2))
