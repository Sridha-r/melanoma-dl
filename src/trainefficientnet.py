import copy
import torch
from sklearn.metrics import roc_auc_score


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    predictions = []
    labels_all = []

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        probs = torch.sigmoid(outputs)

        predictions.extend(
            probs.detach().cpu().numpy().ravel()
        )

        labels_all.extend(
            labels.detach().cpu().numpy().ravel()
        )

    epoch_loss = running_loss / len(train_loader)

    epoch_auc = roc_auc_score(
        labels_all,
        predictions
    )

    return epoch_loss, epoch_auc


def validate(
    model,
    val_loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    predictions = []
    labels_all = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)

            predictions.extend(
                probs.cpu().numpy().ravel()
            )

            labels_all.extend(
                labels.cpu().numpy().ravel()
            )

    val_loss = running_loss / len(val_loader)

    val_auc = roc_auc_score(
        labels_all,
        predictions
    )

    return val_loss, val_auc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=1
):
    best_val_auc = 0.0
    best_model_state = None

    history = {
        "train_loss": [],
        "train_auc": [],
        "val_loss": [],
        "val_auc": []
    }

    for epoch in range(num_epochs):

        print(
            f"\n========== EPOCH "
            f"{epoch + 1}/{num_epochs} =========="
        )

        train_loss, train_auc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_auc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train AUC:  {train_auc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            print("✓ New best model saved in memory")

    print("\n========== DONE ==========")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

    return model, history, best_model_state