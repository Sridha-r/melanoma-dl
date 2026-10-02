import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(
    model,
    data_loader,
    device,
    threshold=0.5
):
    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in data_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)

            probabilities.extend(
                probs.cpu().numpy().ravel()
            )

            labels.extend(
                batch_labels.numpy().ravel()
            )

    predictions = [
        1 if p >= threshold else 0
        for p in probabilities
    ]

    auc = roc_auc_score(
        labels,
        probabilities
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    results = {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return results, probabilities, labels