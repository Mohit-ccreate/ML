"""Multi-modal fusion: tabular ensemble probability + ViT probability → meta classifier."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


def meta_features(p_tab: np.ndarray, p_vit: np.ndarray | None) -> np.ndarray:
    p_tab = np.asarray(p_tab, dtype=float)
    if p_vit is None:
        # vision-only fallback column
        return np.column_stack([p_tab])
    p_vit = np.asarray(p_vit, dtype=float)
    eps = 1e-6
    return np.column_stack([
        p_tab,
        p_vit,
        p_tab - p_vit,                       # disagreement
        p_tab * p_vit,                       # agreement boost
        np.maximum(p_tab, p_vit),
        np.minimum(p_tab, p_vit),
        -np.clip(p_tab, eps, 1 - eps) * np.log(np.clip(p_tab, eps, 1 - eps))
        - np.clip(p_vit, eps, 1 - eps) * np.log(np.clip(p_vit, eps, 1 - eps)),  # joint entropy
    ])


class FusionModel:
    def __init__(self, use_vit: bool = True):
        self.use_vit = use_vit
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, max_iter=2000)
        self.trained_ = False

    def fit(self, p_tab, p_vit, y) -> "FusionModel":
        M = meta_features(p_tab, p_vit if self.use_vit else None)
        Xs = self.scaler.fit_transform(M)
        self.clf.fit(Xs, y)
        self.trained_ = True
        return self

    def predict_proba(self, p_tab, p_vit) -> np.ndarray:
        M = meta_features(p_tab, p_vit if self.use_vit else None)
        Xs = self.scaler.transform(M)
        return self.clf.predict_proba(Xs)[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def tune_threshold(p_up: np.ndarray, y: np.ndarray,
                   target_coverage: float = 0.55,
                   min_coverage: float = 0.25) -> dict:
    """Pick symmetric probability thresholds achieving ~target coverage of
    confident (non-flat) predictions while maximizing selective accuracy.

    A minimum-coverage floor prevents degenerate thresholds that only trade
    a handful of calibration rows.
    """
    p_up = np.asarray(p_up, dtype=float)
    base_acc = float(accuracy_score(y, (p_up >= 0.5).astype(int)))
    best = {"hi": 0.57, "lo": 0.43, "acc": base_acc, "coverage": 1.0,
            "_score": -1.0}
    # explicit grid: robust when probabilities sit in a narrow band around 0.5
    grid = np.round(np.arange(0.51, 0.75, 0.01), 4)
    for hi in grid:
        lo = round(1.0 - hi, 4)
        mask = (p_up >= hi) | (p_up <= lo)
        cov = float(mask.mean())
        if cov < min_coverage:
            continue
        pred = (p_up[mask] >= 0.5).astype(int)
        acc = float(accuracy_score(y[mask], pred))
        score = acc - 0.10 * max(0.0, target_coverage - cov)
        if score > best["_score"]:
            best = {"hi": float(hi), "lo": float(lo), "acc": acc,
                    "coverage": cov, "_score": score}
    best.pop("_score", None)
    return best
