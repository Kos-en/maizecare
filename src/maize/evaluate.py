"""Notebook sections 5-8 as one script: false-positive check, error by class, correlation
with the formula severity, reconstruction heat maps. Everything lands in <model-dir>/eval/.

    python -m maize.evaluate --model-dir models/ae
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from .autoencoder import load_model, pick_device, reconstruction_errors
from .data import HEALTHY, IMG_SIZE, data_root, load_manifest, select
from .severity import extract_quick_features, formula_severity


@torch.no_grad()
def save_reconstruction_figure(model, image_path, title, out_file, device):
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
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

    rng = random.Random(a.seed)
    sample = lambda paths, n: rng.sample(paths, min(n, len(paths)))
    device = pick_device()
    root = data_root(a.data_root)
    df = load_manifest(root)
    out = Path(a.model_dir) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model_dir, device)
    results: dict = {}

    train_paths = select(df, root, label=HEALTHY, split="train")
    val_paths = select(df, root, label=HEALTHY, split="val")
    diseases = sorted(set(df.loc[df["status"] == "ok", "label"]) - {HEALTHY})

    # 5. false-positive check on held-out healthy
    train_err, val_err = reconstruction_errors(model, train_paths, device), reconstruction_errors(model, val_paths, device)
    threshold = float(np.percentile(train_err, 95))
    results["healthy"] = {
        "train_error_mean": float(train_err.mean()), "train_error_std": float(train_err.std()),
        "val_error_mean": float(val_err.mean()), "val_error_std": float(val_err.std()),
        "false_positive_threshold_p95_train": threshold,
        "false_positive_rate_val": float((val_err > threshold).mean()),
    }
    print("Healthy:", json.dumps(results["healthy"], indent=2))

    # 6. error by class
    class_paths = {HEALTHY: sample(val_paths, a.sample_n)}
    class_paths.update({d: sample(select(df, root, label=d), a.sample_n) for d in diseases})
    class_errors = {c: reconstruction_errors(model, p, device) for c, p in class_paths.items() if p}
    results["error_by_class"] = {c: {"mean": float(e.mean()), "median": float(np.median(e)), "n": len(e)}
                                 for c, e in class_errors.items()}
    for c, s in results["error_by_class"].items():
        print(f"{c:8s}: mean={s['mean']:.5f}  median={s['median']:.5f}  n={s['n']}")
    plt.figure(figsize=(8, 6))
    plt.boxplot(list(class_errors.values()))
    plt.xticks(range(1, len(class_errors) + 1), list(class_errors.keys()))
    plt.ylabel("Reconstruction Error (MSE)")
    plt.title("Reconstruction Error by Class (Healthy-Trained Autoencoder)")
    plt.savefig(out / "error_by_class.png", dpi=120)
    plt.close()

    # calibration for the web service: score = clip((e - median_healthy) / (p99_diseased - median_healthy), 0, 1)
    disease_err = [e for c, e in class_errors.items() if c != HEALTHY]
    if disease_err:
        e_d99 = float(np.percentile(np.concatenate(disease_err), 99))
        e_med = float(np.median(val_err))
        if e_d99 > e_med:
            meta = json.loads((Path(a.model_dir) / "meta.json").read_text())
            (Path(a.model_dir) / "calibration.json").write_text(json.dumps({
                "e_healthy_median": e_med, "e_healthy_p95": threshold, "e_disease_p99": e_d99,
                "data_version": meta.get("data_version")}, indent=2))
            print("Wrote calibration.json")
        else:
            print("WARNING: diseased errors are not above healthy: no calibration.json written.")

    # 7. correlation with formula severity (corroboration, not ground truth)
    results["formula_correlation"] = {}
    for d in diseases:
        paths, scores = [], []
        for p in tqdm(sample(select(df, root, label=d), a.corr_n), desc=f"{d} formula features"):
            try:
                scores.append(formula_severity(extract_quick_features(p)))
                paths.append(p)
            except Exception:
                pass
        if len(paths) < 3:
            continue
        errs = reconstruction_errors(model, paths, device)
        rho, pval = spearmanr(errs, scores)
        results["formula_correlation"][d] = {"spearman_rho": float(rho), "p_value": float(pval), "n": len(paths)}
        print(f"{d}: Spearman rho={rho:.3f}, p={pval:.4f}  (n={len(paths)})")
        plt.figure(figsize=(6, 6))
        plt.scatter(scores, errs, alpha=0.4, s=15)
        plt.xlabel("Formula-Based Severity Score"), plt.ylabel("Autoencoder Reconstruction Error")
        plt.title(f"{d}: rho={rho:.3f}, p={pval:.4f}")
        plt.savefig(out / f"correlation_{d}.png", dpi=120)
        plt.close()

    # 8. heat maps: one random example per class
    for c, paths in [(HEALTHY, val_paths)] + [(d, select(df, root, label=d)) for d in diseases]:
        if paths:
            save_reconstruction_figure(model, rng.choice(paths), c, out / f"heatmap_{c}.png", device)

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
