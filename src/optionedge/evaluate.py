"""Evaluation utilities: classification metrics + benchmark comparison tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             average_precision_score, brier_score_loss)


def classification_metrics(y_true, p_up, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_up).astype(float)
    yhat = (p >= threshold).astype(int)
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "baseline_majority": max(base_rate, 1.0 - base_rate),
        "edge_vs_baseline": float(accuracy_score(y_true, yhat)) - max(base_rate, 1.0 - base_rate),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    out["confusion_matrix"] = cm.tolist()
    return out


def selective_metrics(y_true, p_up, hi: float, lo: float) -> dict:
    """Accuracy/coverage when trading only where confidence exceeds thresholds."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p_up).astype(float)
    mask = (p >= hi) | (p <= lo)
    cov = float(mask.mean()) if len(y) else 0.0
    if mask.sum() == 0:
        return {"coverage": 0.0, "selective_accuracy": float("nan"), "n_selected": 0}
    pred = (p[mask] >= 0.5).astype(int)
    return {
        "coverage": cov,
        "n_selected": int(mask.sum()),
        "selective_accuracy": float(accuracy_score(y[mask], pred)),
        "hi": float(hi), "lo": float(lo),
    }


def benchmark_table(results: dict) -> pd.DataFrame:
    """Compare project results vs literature baselines from the proposal."""
    rows = [
        {"model": "Sherasiya 2025 (options-chain LSTM, random split)",
         "modality": "tabular", "accuracy": 0.9010, "validation": "random split (paper)"},
        {"model": "Sherasiya 2025 (XGBoost, random split)",
         "modality": "tabular", "accuracy": 0.8920, "validation": "random split (paper)"},
        {"model": "Sherasiya 2025 (Random Forest, random split)",
         "modality": "tabular", "accuracy": 0.8740, "validation": "random split (paper)"},
        {"model": "Kayit & Ismail 2025 (hybrid voting ensemble)",
         "modality": "tabular", "accuracy": 0.958, "validation": "paper protocol"},
        {"model": "Kpereobong et al. 2025 (ViT + TFT)",
         "modality": "multimodal(vision+price)", "accuracy": 0.96, "validation": "paper protocol"},
        {"model": "Weinberg 2025 (realistic validation note)",
         "modality": "mixed", "accuracy": 0.6014, "validation": "realistic"},
    ]
    for name, m in results.items():
        rows.append({"model": f"OptionEdge — {name}",
                     "modality": m.get("modality", "ours"),
                     "accuracy": m.get("accuracy", float("nan")),
                     "validation": m.get("validation", "walk-forward / holdout")})
    return pd.DataFrame(rows)


def save_metrics(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def plot_equity_and_confusion(preds: pd.DataFrame, backtest_stats: dict,
                              plots_dir: str | Path) -> list[str]:
    """Generate presentation-ready plots; returns list of file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # cumulative accuracy of fused model over time (per symbol pooled by date order)
    if len(preds):
        p = preds.sort_values(["date"]).copy()
        p["hit"] = ((p["p_fused"] >= 0.5).astype(int) == p["y"]).astype(float)
        p["cum"] = p["hit"].cumsum() / (np.arange(len(p)) + 1)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(pd.to_datetime(p["date"]), p["cum"], color="#1f77b4", lw=1.6)
        ax.axhline(0.5, color="grey", ls="--", lw=1, label="coin flip 50%")
        ax.set_title("Walk-forward cumulative accuracy (fused model)")
        ax.set_ylabel("accuracy")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        path = plots_dir / "wf_cumulative_accuracy.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

        # confusion matrix
        y = p["y"].to_numpy()
        yh = (p["p_fused"] >= 0.5).astype(int).to_numpy()
        cm = confusion_matrix(y, yh, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Down", "Up"])
        ax.set_yticks([0, 1], ["Down", "Up"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() * 0.6 else "black", fontsize=13)
        ax.set_title("Fused model confusion (walk-forward)")
        fig.colorbar(im, fraction=0.046)
        path = plots_dir / "confusion_fused.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

        # modality ablation bar
        pass

    # backtest equity
    trades = backtest_stats.get("trades")
    if trades is not None and len(trades):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        cum = trades["net_points"].cumsum()
        ax.plot(range(len(cum)), cum.values, color="#26a69a", lw=1.8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title("Backtest cumulative P&L (index option points, holdout)")
        ax.set_xlabel("trade #")
        ax.set_ylabel("points")
        ax.grid(alpha=0.3)
        path = plots_dir / "backtest_equity.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written
