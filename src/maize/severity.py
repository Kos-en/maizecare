"""Formula severity from OpenCV colour features (from the notebook). Only used to cross-check the
autoencoder score, not as ground truth."""
from __future__ import annotations

import cv2
import numpy as np

SEVERITY_WEIGHTS = {
    "Necrotic_Area": 0.20, "Chlorosis_Intensity": 0.15, "Texture_Roughness": 0.10,
    "Chloro_x_Necrotic": 0.10, "Visual_Distress": 0.10, "Infected_Distribution": 0.10,
    "Lesion_Distribution": 0.05, "Patchiness": 0.05, "Green_Yellow_Ratio": -0.15,
}


def extract_quick_features(image_path) -> dict:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_px = gray.shape[0] * gray.shape[1]

    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
    necro_mask = cv2.inRange(hsv, (0, 10, 10), (20, 200, 200))

    green_frac = cv2.countNonZero(green_mask) / total_px
    yellow_frac = cv2.countNonZero(yellow_mask) / total_px
    necro_frac = cv2.countNonZero(necro_mask) / total_px
    roughness = cv2.Laplacian(gray, cv2.CV_64F).var()

    ch_v = np.clip(yellow_frac, 0, 1)
    gy_v = np.clip((green_frac / (yellow_frac + 1e-6)) / 10.0, 0, 1)
    ne_v = np.clip(necro_frac, 0, 1)
    ro_v = np.clip(roughness / 5000.0, 0, 1)

    return {
        "Necrotic_Area": ne_v, "Chlorosis_Intensity": ch_v, "Texture_Roughness": ro_v,
        "Chloro_x_Necrotic": ch_v * ne_v, "Visual_Distress": (ne_v + ch_v + (1 - gy_v)) / 3.0,
        "Infected_Distribution": ne_v, "Lesion_Distribution": ne_v, "Patchiness": ne_v,
        "Green_Yellow_Ratio": gy_v,
    }


def formula_severity(feats: dict) -> float:
    score = 0.0
    for feat, w in SEVERITY_WEIGHTS.items():
        val = feats[feat]
        score += abs(w) * (1 - val) if w < 0 else w * val
    return float(np.clip(score, 0, 1))
