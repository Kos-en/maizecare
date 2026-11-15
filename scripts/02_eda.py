"""Data understanding (proposal 3.3): balance, size/brightness/blur by class, near duplicates,
class overlap (t-SNE). Writes plots + tables to $MAIZE_DATA_ROOT/eda/.

  python scripts/02_eda.py [--tsne]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maize.config import data_root, load_config
from maize.data.hashing import group_ids, hex_to_u64, near_pairs
from maize.data.io import load_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/data.yaml")
ap.add_argument("--tsne", action="store_true", help="needs TensorFlow; samples <=2000 images")
args = ap.parse_args()
cfg = load_config(args.config)
out = data_root() / "eda"
out.mkdir(exist_ok=True)
df = load_manifest()

# 1. class balance
bal = df.groupby(["source", "label"]).size().unstack(fill_value=0)
print("1. CLASS BALANCE\n", bal, "\n")
bal.to_csv(out / "class_balance.csv")

# 2-3. size / brightness / blur by class  (a class must not be recognisable from these alone)
df["megapixels"] = df["w"] * df["h"] / 1e6
stats = df.groupby("label")[["w", "h", "megapixels", "brightness", "blur"]].describe().T
stats.to_csv(out / "stats_by_class.csv")
fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
for a, col in zip(ax, ["w", "h", "brightness", "blur"]):
    df.boxplot(column=col, by="label", ax=a)
    a.set_title(col), a.set_xlabel("")
    if col == "blur":
        a.set_yscale("log")
plt.suptitle("")
plt.tight_layout()
plt.savefig(out / "stats_by_class.png", dpi=120)
by_source = df.groupby(["source", "label"])[["w", "h", "brightness", "blur"]].median()
print("2-3. MEDIAN SIZE/LIGHT/BLUR by source+class (look for a source or class that stands out)\n", by_source, "\n")

# 4. near duplicates -> how many independent healthy leaves do we really have?
h = cfg["hashing"]
ah, ph = hex_to_u64(df["ahash"]), hex_to_u64(df["phash"])
gid = group_ids(len(df), near_pairs(ah, ph, h["ahash_max"], h["phash_max"], h["mode"]))
df["group_id"] = gid
dup = df.groupby("label").agg(images=("id", "size"), groups=("group_id", "nunique"))
dup["images_per_group"] = (dup["images"] / dup["groups"]).round(2)
print("4. NEAR DUPLICATES (groups = independent leaves/scenes)\n", dup, "\n")
dup.to_csv(out / "near_duplicates.csv")
if dup.loc["Healthy", "groups"] < 500:
    print("   NOTE: few independent healthy groups: the autoencoder may not have enough to learn from.\n")

# 5. class overlap
if args.tsne:
    import cv2
    import tensorflow as tf
    from sklearn.manifold import TSNE
    sample = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 500), random_state=0))
    imgs = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (224, 224)) for p in sample["abs_path"]])
    fx = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg", weights="imagenet")
    feats = fx.predict(tf.keras.applications.mobilenet_v2.preprocess_input(imgs.astype("float32")), verbose=0)
    z = TSNE(n_components=2, init="pca", random_state=0).fit_transform(feats)
    plt.figure(figsize=(6, 5))
    for lab in sample["label"].unique():
        m = (sample["label"] == lab).values
        plt.scatter(z[m, 0], z[m, 1], s=6, label=lab)
    plt.legend(), plt.title("t-SNE of pretrained features"), plt.tight_layout()
    plt.savefig(out / "tsne.png", dpi=120)
    print("5. t-SNE saved:", out / "tsne.png")
print("Outputs in", out)
