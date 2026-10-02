import time
import torch


def train_resnet50(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    num_epochs=1,
    checkpoint_every=50
):

    history = {
        "train_loss": [],
        "train_accuracy": []
    }

    print("Starting ResNet50 training...")
    print(f"Total batches per epoch: {len(train_loader)}")

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        print(f"\n========== EPOCH {epoch + 1}/{num_epochs} ==========")

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % checkpoint_every == 0:

                elapsed = time.time() - start_time

                avg_time = elapsed / (batch_idx + 1)

                remaining = avg_time * (
                    len(train_loader) - batch_idx - 1
                )

                current_loss = running_loss / total
                current_accuracy = correct / total

                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"({(batch_idx + 1) / len(train_loader):.0%}) | "
                    f"Loss: {current_loss:.4f} | "
                    f"Acc: {current_accuracy:.4f} | "
                    f"Elapsed: {elapsed/60:.1f} min | "
                    f"ETA: {remaining/60:.1f} min"
                )

                checkpoint = {
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": current_loss,
                    "accuracy": current_accuracy
                }

                torch.save(
                    checkpoint,
                    "resnet50_latest_checkpoint.pth"
                )

                print("  ✓ Checkpoint saved")

        train_loss = running_loss / total
        train_accuracy = correct / total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        print("\n========== EPOCH COMPLETE ==========")
        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")

        torch.save(
            model.state_dict(),
            "resnet50_phase1.pth"
        )

        torch.save(
            history,
            "resnet50_training_history.pth"
        )

        print("✓ Final model saved")
        print("✓ Training history saved")

    return history
