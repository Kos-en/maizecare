"""Evaluate a trained model: false-positive check on healthy, error by class, correlation with
the formula severity, heat maps. Output goes to <model-dir>/eval/; also writes calibration.json.

    python -m maize.evaluate --model-dir models/ae
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from .autoencoder import load_model, pick_device, reconstruction_errors
from .cache import load_images
from .data import HEALTHY, abs_paths, data_root, load_manifest, select
from .preprocess import decode, to_input
from .severity import extract_quick_features, formula_severity

plt.switch_backend("Agg")  # save figures to files, never open windows

Errors = Callable[[pd.DataFrame], np.ndarray]  # manifest rows -> reconstruction error per image


def healthy_check(train_err: np.ndarray, val_err: np.ndarray) -> dict:
    """Held-out healthy leaves should reconstruct as well as training ones."""
    threshold = float(np.percentile(train_err, 95))
    return {
        "train_error_mean": float(train_err.mean()), "train_error_std": float(train_err.std()),
        "val_error_mean": float(val_err.mean()), "val_error_std": float(val_err.std()),
        "false_positive_threshold_p95_train": threshold,
        "false_positive_rate_val": float((val_err > threshold).mean()),
    }


def error_by_class(class_errors: dict[str, np.ndarray], out: Path) -> dict:
    stats = {c: {"mean": float(e.mean()), "median": float(np.median(e)), "n": len(e)} for c, e in class_errors.items()}
    for c, s in stats.items():
        print(f"{c:8s}: mean={s['mean']:.5f}  median={s['median']:.5f}  n={s['n']}")
    plt.figure(figsize=(8, 6))
    plt.boxplot(list(class_errors.values()))
    plt.xticks(range(1, len(class_errors) + 1), list(class_errors.keys()))
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Reconstruction Error by Class (Healthy-Trained Autoencoder)")
    plt.savefig(out / "error_by_class.png", dpi=120)
    plt.close()
    return stats


def write_calibration(model_dir: Path, val_err: np.ndarray, threshold: float,
                      class_errors: dict[str, np.ndarray]) -> None:
    """Numbers the web app needs: score = clip((e - healthy median) / (diseased p99 - healthy median), 0, 1)."""
    cal_file = model_dir / "calibration.json"
    cal_file.unlink(missing_ok=True)  # never leave a stale calibration from an earlier model
    disease_err = [e for c, e in class_errors.items() if c != HEALTHY]
    if not disease_err:
        print("WARNING: no diseased images, so no calibration.json written.")
        return
    e_med = float(np.median(val_err))
    e_d99 = float(np.percentile(np.concatenate(disease_err), 99))
    if e_d99 <= e_med:
        print("WARNING: diseased errors are not above healthy, so no calibration.json written.")
        return
    meta = json.loads((model_dir / "meta.json").read_text())
    cal_file.write_text(json.dumps({"e_healthy_median": e_med, "e_healthy_p95": threshold,
                                    "e_disease_p99": e_d99, "data_version": meta.get("data_version")}, indent=2))
    print("Wrote calibration.json")


def formula_correlation(name: str, rows: pd.DataFrame, root: Path, errors: Errors, out: Path) -> dict | None:
    """Spearman between the autoencoder error and the notebook's formula severity for one disease."""
    keep, scores = [], []
    for i, p in enumerate(tqdm(abs_paths(rows, root), desc=f"{name} formula features")):
        try:  # the formula needs the full-size photo; skip any that cannot be read
            scores.append(formula_severity(extract_quick_features(p)))
            keep.append(i)
        except Exception:
            pass
    if len(keep) < 3:
        return None
    errs = errors(rows.iloc[keep])
    rho, pval = spearmanr(errs, scores)
    print(f"{name}: Spearman rho={rho:.3f}, p={pval:.4f}  (n={len(keep)})")
    plt.figure(figsize=(6, 6))
    plt.scatter(scores, errs, alpha=0.4, s=15)
    plt.xlabel("Formula-Based Severity Score")
    plt.ylabel("Autoencoder Reconstruction Error")
    plt.title(f"{name}: rho={rho:.3f}, p={pval:.4f}")
    plt.savefig(out / f"correlation_{name}.png", dpi=120)
    plt.close()
    return {"spearman_rho": float(rho), "p_value": float(pval), "n": len(keep)}


@torch.no_grad()
def save_reconstruction_figure(model, image_path, title, out_file, device):
    x = torch.from_numpy(to_input(decode(Path(image_path).read_bytes())))[None].to(device)
    recon = model(x).cpu().numpy()[0].transpose(1, 2, 0)
    original = x.cpu().numpy()[0].transpose(1, 2, 0)
    error_map = np.mean((original - recon) ** 2, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(recon)
    axes[1].set_title("Reconstruction")
    im = axes[2].imshow(error_map, cmap="hot")
    axes[2].set_title("Error Heatmap")
    for a in axes:
        a.axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_file, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models/ae")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--sample-n", type=int, default=1500, help="max images per class for the error comparison")
    ap.add_argument("--corr-n", type=int, default=500, help="images per disease for the formula correlation")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    device = pick_device()
    root = data_root(a.data_root)
    model_dir = Path(a.model_dir)
    out = model_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    df = load_manifest(root)
    model = load_model(model_dir, device)

    def errors(rows: pd.DataFrame) -> np.ndarray:
        return reconstruction_errors(model, load_images(rows, root), device)

    def sample(rows: pd.DataFrame, n: int) -> pd.DataFrame:
        return rows.sample(min(n, len(rows)), random_state=a.seed)

    # healthy is represented by its held-out val split; each disease by all its accepted images
    val_rows = select(df, label=HEALTHY, split="val")
    diseases = {d: select(df, label=d) for d in sorted(set(select(df)["label"]) - {HEALTHY})}
    by_class = {HEALTHY: val_rows, **diseases}
    results: dict = {}

    train_err, val_err = errors(select(df, label=HEALTHY, split="train")), errors(val_rows)
    results["healthy"] = healthy_check(train_err, val_err)
    print("Healthy:", json.dumps(results["healthy"], indent=2))

    class_errors = {c: errors(sample(r, a.sample_n)) for c, r in by_class.items() if len(r)}
    results["error_by_class"] = error_by_class(class_errors, out)
    write_calibration(model_dir, val_err, results["healthy"]["false_positive_threshold_p95_train"], class_errors)

    corr = {d: formula_correlation(d, sample(r, a.corr_n), root, errors, out) for d, r in diseases.items()}
    results["formula_correlation"] = {d: v for d, v in corr.items() if v is not None}

    for c, rows in by_class.items():  # one example heat map per class
        if len(rows):
            save_reconstruction_figure(model, abs_paths(sample(rows, 1), root)[0], c, out / f"heatmap_{c}.png", device)

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
