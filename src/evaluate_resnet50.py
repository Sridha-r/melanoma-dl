"""Evaluate a trained ResNet50 checkpoint on a validation or test split.

Example (Kaggle):
python evaluate_resnet50.py \
  --split-csv /kaggle/working/test_split.csv \
  --image-dir /kaggle/input/competitions/siim-isic-melanoma-classification/jpeg/train \
  --checkpoint /kaggle/working/resnet50_outputs/resnet50_epoch_3.pth \
  --output-dir /kaggle/working/resnet50_outputs \
  --split-name test
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from resnet50 import MelanomaDataset, create_resnet50, get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a ResNet50 checkpoint.")
    parser.add_argument("--split-csv", required=True, help="Validation or test split CSV.")
    parser.add_argument("--image-dir", required=True, help="Directory containing <image_name>.jpg files.")
    parser.add_argument("--checkpoint", required=True, help="Path to a ResNet50 state_dict checkpoint.")
    parser.add_argument("--output-dir", default="results/resnet50", help="Directory for metrics and plots.")
    parser.add_argument("--split-name", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(args.split_csv)
    dataset = MelanomaDataset(
        dataframe, args.image_dir, transform=get_transforms(training=False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_resnet50(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Supports either a plain state_dict or a training checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)

            probabilities = torch.softmax(outputs.float(), dim=1)[:, 1]
            y_true.extend(labels.numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= args.threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "split": args.split_name,
        "number_of_images": int(len(y_true)),
        "threshold": float(args.threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    print(json.dumps(metrics, indent=2))
    metrics_path = output_dir / f"resnet50_{args.split_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save a confusion matrix plot with the same class order as the report.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"ResNet50 - {args.split_name.title()} Confusion Matrix\n"
                 f"(Threshold = {args.threshold:.2f})")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], labels=["Non-melanoma\n(Class 0)", "Melanoma\n(Class 1)"])
    ax.set_yticks([0, 1], labels=["Non-melanoma\n(Class 0)", "Melanoma\n(Class 1)"])

    midpoint = cm.max() / 2 if cm.max() else 0
    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            ax.text(
                col, row, f"{value:,}",
                ha="center", va="center",
                color="white" if value > midpoint else "black",
                fontsize=13, fontweight="bold",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Number of images")
    fig.tight_layout()
    plot_path = output_dir / f"confusion_matrix_ResNet50_{args.split_name}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {plot_path}")


if __name__ == "__main__":
    main()
