"""Final evaluation on the held-out TEST set. Run once models are chosen.

  # Stage 1: every fold model of one or more configurations, clean + blurred + low-light
  python scripts/08_evaluate.py clf --runs runs/clf/mobilenetv2_masked runs/clf/resnet50_masked
  # Stage 2: severity score vs symptomatic area, per disease
  python scripts/08_evaluate.py sev --ae-run runs/ae/cae_ssim/fold0
"""
import argparse
import json
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

import maize.models.autoencoder  # noqa: F401
import maize.models.classifier  # noqa: F401
from maize import severity as sev
from maize.config import runs_root, write_json
from maize.constants import CLASS_TO_ID, DISEASES, HEALTHY
from maize.data.degrade import DEGRADATIONS
from maize.data.io import load_ae_arrays, load_clf_array, load_manifest, load_proc
from maize.evaluate import classification as ce
from maize.evaluate import severity_eval as se
from maize.symptom_area import SYMPTOM_VERSION, symptomatic_area

ap = argparse.ArgumentParser()
ap.add_argument("task", choices=["clf", "sev"])
ap.add_argument("--runs", nargs="*", default=[], help="classifier run dirs (each holding fold*/best.keras)")
ap.add_argument("--ae-run")
args = ap.parse_args()
df = load_manifest()
test = df[df["split"] == "test"].reset_index(drop=True)
out = runs_root() / "eval"

if args.task == "clf":
    y = test["label"].map(CLASS_TO_ID).values
    summary, mean_probs = [], {}
    for run in args.runs:
        run = Path(run)
        masked = json.loads((sorted(run.glob("fold*/config.json"))[0]).read_text())["masked"]
        per_fold = {c: [] for c in DEGRADATIONS}
        for fold_dir in sorted(run.glob("fold*")):
            model = keras.saving.load_model(fold_dir / "best.keras", compile=False)
            for cond in DEGRADATIONS:
                x = load_clf_array(test, masked, cond)
                t0 = time.perf_counter()
                p = model.predict(x, batch_size=32, verbose=0)
                per_fold[cond].append((p, (time.perf_counter() - t0) / len(x) * 1000))
        for cond, items in per_fold.items():
            f1s = [ce.summarize(y, p)["macro_f1"] for p, _ in items]
            p_avg = np.mean([p for p, _ in items], 0)
            s = ce.summarize(y, p_avg)
            lo, hi = ce.bootstrap_ci(y, p_avg)
            summary.append({"model": run.name, "condition": cond, "macro_f1_mean_over_folds": float(np.mean(f1s)),
                            "macro_f1_sd": float(np.std(f1s)), "fold_ensemble_macro_f1": s["macro_f1"],
                            "ci95": [lo, hi], "accuracy": s["accuracy"], "ece": s["ece"], "recall": s["recall"],
                            "keras_ms_per_image": float(np.mean([t for _, t in items]))})
            if cond == "clean":
                mean_probs[run.name] = p_avg
                write_json(out / f"{run.name}_confusion.json", s["confusion"])
    names = list(mean_probs)
    if len(names) >= 2:  # McNemar between the two best by clean macro-F1
        best = sorted(names, key=lambda n: -ce.summarize(y, mean_probs[n])["macro_f1"])[:2]
        print("McNemar", best, ce.mcnemar_exact(y, mean_probs[best[0]], mean_probs[best[1]]))
    write_json(out / "classifier_test_results.json", summary)
    print(pd.DataFrame(summary)[["model", "condition", "macro_f1_mean_over_folds", "macro_f1_sd", "accuracy", "ece"]])

else:
    ae_dir = Path(args.ae_run)
    ae = keras.saving.load_model(ae_dir / "best.keras", compile=False)
    cal = sev.load_calibration(ae_dir / "severity_calibration.json")
    x, m = load_ae_arrays(test)
    xh = ae.predict(x, batch_size=64, verbose=0)
    test["e"] = [sev.masked_error(a, b, c) for a, b, c in zip(x, xh, m)]
    test["s"] = test["e"].map(lambda e: sev.score(e, cal))
    area = []
    for _, r in test.iterrows():
        img, msk = load_proc(r)
        area.append(symptomatic_area(img, msk))
    test["area"] = area
    healthy = test[test["label"] == HEALTHY]
    res = {"symptom_version": SYMPTOM_VERSION, "false_high_rate": se.false_high_rate(healthy["e"].values, cal["e_healthy_p95"]),
           "per_disease": {}}
    for d in DISEASES:
        sub = test[test["label"] == d]
        res["per_disease"][d] = {
            "auroc_vs_healthy": se.auroc(healthy["s"].values, sub["s"].values),
            **se.vs_symptomatic_area(sub["s"].values, sub["area"].values),
            "stability": se.group_stability(sub["s"].values, sub["group_id"].values),
        }
    write_json(out / "severity_test_results.json", res)
    test[["id", "label", "e", "s", "area", "group_id"]].to_csv(out / "severity_test_scores.csv", index=False)
    print(json.dumps(res, indent=2))
    if "UNFROZEN" in SYMPTOM_VERSION:
        print("\nWARNING: symptom colour ranges are not frozen (see symptom_area.py). Freeze BEFORE reporting.")
