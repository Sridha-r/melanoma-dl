"""Evaluate the best validation checkpoint on the held-out test split."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from tqdm import tqdm

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
THRESHOLD = 0.50
SPLIT_DIR = Path("data/splits")
CHECKPOINT_PATH = Path("checkpoints/efficientnet_b0_best.pt")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

class MelanomaDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row.image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor(float(row.target), dtype=torch.float32)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if not (SPLIT_DIR / "test.csv").exists():
        raise FileNotFoundError("Run cache_data.py first.")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("Run train.py first; best checkpoint is missing.")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(
        MelanomaDataset(SPLIT_DIR / "test.csv", transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                logits = model(images)
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            y_true.extend(labels.numpy().ravel())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_roc_auc": float(checkpoint["best_val_auc"]),
        "test_images": int(len(y_true)),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc_average_precision": float(average_precision_score(y_true, y_prob)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

    print("\n========== TEST RESULTS ==========")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nConfusion matrix:")
    print(pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual non-melanoma", "Actual melanoma"],
        columns=["Predicted non-melanoma", "Predicted melanoma"]
    ))

    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame({
        "target": y_true,
        "probability": y_prob,
        "prediction": y_pred
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

if __name__ == "__main__":
    main()
