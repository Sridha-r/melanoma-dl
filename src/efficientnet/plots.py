"""Generate confusion matrix, ROC, PR, and threshold comparison plots."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)

OUTPUT_DIR = Path("outputs")
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
PLOT_DIR = OUTPUT_DIR / "plots"

def main():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError("Run evaluate.py first.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PREDICTIONS_PATH)
    y_true = df["target"].to_numpy(dtype=int)
    y_prob = df["probability"].to_numpy(dtype=float)
    y_pred = df["prediction"].to_numpy(dtype=int)

    # 1. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Non-melanoma", "Melanoma"],
        yticklabels=["Non-melanoma", "Melanoma"],
        xlabel="Predicted label", ylabel="Actual label",
        title="EfficientNet-B0 Confusion Matrix"
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="EfficientNet-B0 ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    # 3. Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Average precision = {ap:.4f}")
    ax.axhline(prevalence, linestyle="--",
               label=f"Positive prevalence = {prevalence:.4f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="EfficientNet-B0 Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    # 4. Threshold comparison (descriptive; do not select a threshold on test data)
    rows = []
    for threshold in np.arange(0.10, 1.00, 0.10):
        pred = (y_prob >= threshold).astype(int)
        rows.append({
            "threshold": round(float(threshold), 2),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(PLOT_DIR / "threshold_comparison.csv", index=False)
    print(threshold_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["precision", "recall", "f1", "accuracy"]:
        ax.plot(threshold_df["threshold"], threshold_df[metric],
                marker="o", label=metric.capitalize())
    ax.set(xlabel="Decision threshold", ylabel="Score",
           title="Metrics Across Classification Thresholds")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "threshold_comparison.png", dpi=300)
    plt.close(fig)

    print(f"\nPlots and threshold table saved to: {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    main()
