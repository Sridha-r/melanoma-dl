import torch
import torch.nn as nn
from torchvision import models


def create_resnet50(num_classes=2):
    """
    Create a ResNet50 model using pretrained ImageNet weights.
    The final layer is replaced for binary classification.
    """

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def create_weighted_loss(train_df, device):
    """
    Create weighted CrossEntropyLoss to handle class imbalance.
    """

    class_counts = train_df["target"].value_counts().sort_index()

    total = class_counts.sum()
    num_classes = 2

    weights = total / (num_classes * class_counts)

    class_weights = torch.tensor(
        weights.to_numpy(),
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion, class_weights


def create_optimizer(model, learning_rate=1e-4, weight_decay=1e-4):
    """
    Create AdamW optimizer.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    return optimizer
