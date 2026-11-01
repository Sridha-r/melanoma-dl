"""Shared ResNet50 components for SIIM-ISIC melanoma classification."""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms


IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def create_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create ImageNet-pretrained ResNet50 with a replacement classification head."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transforms(training: bool = False):
    """Return the same preprocessing used in the final Kaggle experiment."""
    if training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


class MelanomaDataset(Dataset):
    """Dataset reading image_name and target columns from a split CSV."""

    def __init__(self, dataframe: pd.DataFrame, image_dir, transform=None):
        required = {"image_name", "target"}
        missing = required.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / f"{row['image_name']}.jpg"

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["target"]), dtype=torch.long)
        return image, label


def create_weighted_loss(train_df: pd.DataFrame, device):
    """Create class-weighted CrossEntropyLoss using the training split only."""
    counts = train_df["target"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError(
            "Both classes (0 and 1) must be present in the training split."
        )

    total = int(counts.sum())
    weights = total / (2.0 * counts.to_numpy())
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=class_weights), class_weights


def create_optimizer(model, learning_rate: float = 1e-4,
                     weight_decay: float = 1e-4):
    """Create the AdamW optimizer used in the final experiment."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
