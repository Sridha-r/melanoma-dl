"""Train EfficientNet-B0 using the cached patient-wise split."""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

SEED = 42
BATCH_SIZE = 32
EPOCHS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
POSITIVE_WEIGHT = 56.7519
IMAGE_SIZE = 224
NUM_WORKERS = 2

SPLIT_DIR = Path("data/splits")
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

class MelanomaDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, torch.tensor(float(row.target), dtype=torch.float32)

def get_loaders():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    common = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                  pin_memory=DEVICE.type == "cuda",
                  persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(
        MelanomaDataset(SPLIT_DIR / "train.csv", train_tf),
        shuffle=True, **common
    )
    val_loader = DataLoader(
        MelanomaDataset(SPLIT_DIR / "validation.csv", eval_tf),
        shuffle=False, **common
    )
    return train_loader, val_loader

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses, targets, probabilities = [], [], []

    for images, labels in tqdm(loader, desc="Training" if training else "Validation"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).view(-1, 1)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(loss.item())
        probabilities.extend(torch.sigmoid(logits.detach()).cpu().numpy().ravel())
        targets.extend(labels.detach().cpu().numpy().ravel())

    return float(np.mean(losses)), float(roc_auc_score(targets, probabilities))

def main():
    seed_everything()
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    train_loader, val_loader = get_loaders()

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POSITIVE_WEIGHT], device=DEVICE)
    )
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_auc = -1.0
    history = []

    print(f"Device: {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = run_epoch(
            model, train_loader, criterion, optimizer, scaler
        )
        val_loss, val_auc = run_epoch(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_roc_auc": train_auc,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss={train_loss:.4f}, AUC={train_auc:.4f} | "
            f"val loss={val_loss:.4f}, AUC={val_auc:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_auc": max(best_val_auc, val_auc),
            "history": history,
        }
        torch.save(state, CHECKPOINT_DIR / "efficientnet_b0_latest.pt")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            state["best_val_auc"] = best_val_auc
            torch.save(state, CHECKPOINT_DIR / "efficientnet_b0_best.pt")
            print("Saved new best checkpoint.")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation ROC-AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()
