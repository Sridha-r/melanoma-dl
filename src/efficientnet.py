import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_model(device):
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze pretrained feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier for binary classification
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        1
    )

    model = model.to(device)

    return model