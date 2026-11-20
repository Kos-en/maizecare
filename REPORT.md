# MaizeCare: final report

*Detecting maize leaf diseases and grading their severity from one phone photo*

Joseph Kosen, Strathmore University

> **Before submitting:** the two remaining `TODO`s (the exploratory analysis in 4.1 and the robustness numbers in
> 4.2) come from running the results notebook (`notebooks/maizecare.ipynb`).

## 1. The problem

Maize provides roughly a third of the calories eaten in Sub-Saharan Africa. Maize Leaf Blight (MLB), Maize Streak
Virus (MSV) and Maize Lethal Necrosis (MLND) can cost anything from a small share of a harvest to all of it. Most
smallholder farmers cannot reach an extension officer quickly, but most have a phone camera.

A farmer looking at a sick leaf needs to know two things:

1. **What is it?** The treatment depends on the disease. MLND, for example, means removing plants and rotating
   crops, not spraying.
2. **How bad is it?** A few spots and a leaf that is mostly dead call for different actions.

Most published work, including my own earlier version of this project, answers only the first question. This
project answers both, in English and Kiswahili, in a free web app that runs on any phone.

**Constraint that shapes everything:** there are disease labels for the photos, but **no severity labels**. Nobody
has rated how severe each leaf is, so severity cannot be learned as a normal supervised target.

## 2. What I learned from the earlier version

The earlier version (a YOLOv8 classifier with a feature-based ensemble) reported 99.39% test accuracy. Looking
closer, the result could not be trusted:

- Near-duplicate detection chained unrelated photos together, so one "group" held over 3,700 images. Because
  whole groups had to stay in one split, the split ended up 95.7% / 2.8% / 1.4% instead of 70 / 20 / 10, with
  only 164 test images.
- The severity grade came from hand-picked weights over colour features. Nothing checked whether it measured
  severity.

This version was rebuilt around those two lessons: a split that is honest and large enough to test on, and a
severity measure that is learned and then checked against an independent reference.

## 3. Approach and why

### 3.1 Data pipeline

All photos pass through one scripted pipeline (`maize-prepare`):

| Step | Choice | Why |
|---|---|---|
| File check | must open as JPEG/PNG, shortest side >= 50 px | broken or tiny files teach nothing |
| Near copies | within 6 bits on **both** pHash and aHash; chains form groups | the same leaf shot twice must not sit in train and test. Either hash alone chained 18,624 unrelated photos into one group; requiring both gives a largest group of 9 |
| Label conflicts | a group with two classes is dropped | a photo cannot be two diseases |
| Split | whole groups, stratified by class, 70 / 15 / 15 | no leakage, and every class is in every split in the right proportion |
| Leaf mask | GrabCut grown from the middle of the photo | the farmer aims at the leaf; colour rules (Excess Green) kept weeds and lost yellowed leaves |

A leak check recomputes every near-copy pair and confirms that none crosses a split (notebook section 1).

### 3.2 Stage 1: which disease? A transfer-learning classifier

Four ImageNet backbones (MobileNetV2, MobileNetV3-Small, EfficientNet-B0, ResNet50) are trained the same way: first
the new head, then the top 40 backbone layers at a lower learning rate. Class weights balance the classes.
Augmentation imitates field photos (blur, noise, brightness) but never shifts the hue, because yellowing is the
disease signal. The epoch with the best validation macro-F1 is kept.

**Why transfer learning and a small backbone:** there are only a few thousand photos per class, which is too few
to train a CNN from scratch, and the app must run on a free 512 MB server. Target (a) was set to make this
trade-off explicit: **within 3 points of ResNet50's macro-F1, at least 3x smaller.**

**Why ask for a retake:** below 60% confidence the app asks for another photo instead of guessing. A wrong
diagnosis costs a farmer more than a second photo does.

### 3.3 Stage 2: how severe? A healthy-only autoencoder

An autoencoder learns to compress and rebuild **healthy leaves only**, inside the leaf mask. Shown a diseased
leaf, it rebuilds the parts it has never seen (lesions, streaks, yellowing) badly. The reconstruction error
therefore grows with the damage, and it needs no severity labels, which is exactly the constraint in section 1.

- Two variants of the same shape, a convolutional autoencoder and a VAE, compressing 24x so the model cannot just
  copy its input.
- Each is scored on validation with the plain error and with a smoothed error (colour and tone, not fine
  texture). The pair with the best AUROC on its **weakest** disease is kept, because target (b) is per disease.
- The error is turned into a 0-1 score on validation: 0 is a typical healthy leaf, 1 is the 99th percentile of
  diseased leaves.

**How severity is checked without severity labels:** the score is compared with A(x), the percentage of the leaf
in a symptom colour (yellowing, brown lesions, dead tissue), the measure used by tools such as Leaf Doctor and
ASSESS. A(x) is only a reference and is never used for training. Its colour ranges were fixed before any score
was seen. Two simple baselines must be beaten: the classifier's 1 - P(healthy), and the colour distance from an
average healthy leaf.

### 3.4 Targets (set in the proposal, before testing)

- **(a)** Classifier within 3 points of ResNet50, at least 3x smaller.
- **(b)** Autoencoder error separates healthy from **each** disease with AUROC >= 0.80.
- **(c)** Severity score agrees with A(x), Spearman >= 0.40 for MLND and MLB, and beats both baselines.

All choices were made on validation; the test photos were used once, at the end.

## 4. Results

### 4.1 Data (notebook sections 1 and 2)

The pipeline read 35,596 photos. It removed 4,123 exact copies and 31 photos in groups holding two classes,
leaving **31,442 usable photos**.

| | Healthy | MLB | MLND | MSV |
|---|---|---|---|---|
| usable photos | 8,690 | 5,276 | 9,376 | 8,100 |
| train / val / test | 6,083 / 1,303 / 1,304 | 3,693 / 791 / 792 | 6,563 / 1,406 / 1,407 | 5,670 / 1,215 / 1,215 |
| near-copy groups | 8,676 | 5,276 | 9,362 | 8,054 |

Every class landed on the 70 / 15 / 15 target. Requiring both hashes keeps near-copy groups small (largest: 9
photos), so whole groups could be split cleanly. Near-copy pairs crossing a split: **0**. Compare the earlier
version: one group of over 3,700 photos, and a 95.7 / 2.8 / 1.4 split.

Exploratory analysis: TODO, one or two sentences each on class balance, photo sizes per class, the colour and
brightness profiles, and what the t-SNE map shows (separate clusters or overlap).

### 4.2 Stage 1: classifier (notebook section 3)

| Backbone | Val macro-F1 | Test macro-F1 | Test ECE | Size (MB) |
|---|---|---|---|---|
| **MobileNetV2 (chosen)** | 0.963 | **0.963** | 0.006 | **9.4** |
| MobileNetV3-Small | 0.950 | 0.954 | 0.007 | 4.2 |
| EfficientNet-B0 | 0.958 | 0.963 | 0.009 | 16.9 |
| ResNet50 | 0.970 | 0.965 | 0.020 | 94.8 |

**Target (a) is met.** MobileNetV2 is 0.2 points below ResNet50 on test and 10x smaller. It is also the best
calibrated (ECE 0.006): when it says 90%, it is right about 90% of the time, which is what makes the "please
retake" threshold trustworthy. On the 4,718 test photos, recall per class is Healthy 96.6%, MLB 93.8%, MLND
97.7% and MSV 96.5%. MLB is the hardest class, mostly confused with Healthy (27 photos).

Robustness on degraded test photos, macro-F1: clean TODO, blurred TODO, dark TODO.

### 4.3 Stage 2: severity (notebook sections 5 and 6)

The leaf mask mattered most. Replacing the colour rule with GrabCut raised the validation AUROC from **0.65 to
0.86**. The chosen autoencoder is the **convolutional** one with the **plain** error. On validation it had the best
AUROC on its weakest disease (0.84). The VAE was as good on average (0.858) but dropped to 0.80 on MSV.

| Disease | test leaves | (b) AUROC vs healthy | (c) Spearman vs A(x) | Baseline: colour distance | Baseline: classifier |
|---|---|---|---|---|---|
| MLB | 709 | **0.862** | 0.312 | 0.157 | 0.220 |
| MLND | 1,387 | **0.843** | 0.345 | 0.430 | 0.383 |
| MSV | 1,050 | **0.862** | 0.273 | 0.180 | 0.284 |

Healthy test leaves scored above the healthy range: 5.3% (about the 5% expected, since that range is the 95th
percentile of healthy validation leaves). Leaf masks flagged as doubtful and left out: Healthy 5%, MLB 10%,
MLND 2%, MSV 14%.

**Reading the results.**

- **Target (b) is met for every disease.** The autoencoder's error tells a diseased leaf from a healthy one with an
  AUROC of 0.84-0.86, without ever seeing a diseased leaf in training.
- **Target (c) is not met.** The score does rank leaves by severity (every Spearman is positive), but only
  weakly: 0.27-0.35, below the 0.40 target.
- **Against the baselines, it depends on the disease.** For MLB, the autoencoder clearly beats both baselines
  (0.31 against 0.16 and 0.22). For MSV it beats the colour baseline and ties the classifier. For MLND both
  baselines do better.
- **Two reasons for MLND.** First, MLND mostly shows as broad yellowing: a uniformly yellowed leaf is still a
  smooth leaf shape, which the autoencoder partly rebuilds, so its error stays low. Second, A(x) itself measures
  colour, so a colour-based baseline is naturally close to it on a disease that is mostly a colour change.

In short, the autoencoder is a reliable "is something wrong with this leaf" detector and a rough severity
indicator, strongest on lesion-type damage (MLB).

## 5. Deployment

Both models are exported to ONNX and served by a FastAPI app with `onnxruntime`. The Docker image has no deep
learning framework, so it fits Render's free plan. CI runs linting, tests and dependency audits on every push and
deploys `main` automatically. Photos are processed in memory and never stored. See
[ARCHITECTURE.md](ARCHITECTURE.md).

## 6. Limitations

- **No ground-truth severity.** A(x) is a colour rule, not an expert's rating. Agreement with it is evidence, not
  proof, that the score measures severity.
- **Severity ranking is weak (target c not met)**, and weakest for MLND, for the reasons in 4.3.
- **Data quality.** Some photos in the Healthy folder are not maize (for example bean leaves). Photos come from a
  few sources, and results per source (Tanzania / Uganda) are not yet reported.
- **Robustness was tested on synthetic blur and darkening**, not on real poor-quality field photos.
- **No confidence intervals yet.** With a few hundred test photos per class, differences of a few points between
  backbones may not be significant.
- **The Kiswahili text and the advice** still need a review by a native speaker and an extension officer.

## 7. Next steps

1. A colour-reconstruction autoencoder (grey leaf in, colour out) aimed at MLND yellowing.
2. By-eye severity ratings on a sample of leaves, to validate the score directly.
3. Bootstrap confidence intervals, McNemar's test between backbones, and results per data source.
4. Clean the Healthy class, and test on real field photos taken by farmers.
