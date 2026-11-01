"""Train ResNet50 on the SIIM-ISIC melanoma training split.

Example (Kaggle):
python train_resnet50.py \
  --train-csv /kaggle/working/train_split.csv \
  --image-dir /kaggle/input/competitions/siim-isic-melanoma-classification/jpeg/train \
  --output-dir /kaggle/working/resnet50_outputs
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from resnet50 import (
    MelanomaDataset,
    create_optimizer,
    create_resnet50,
    create_weighted_loss,
    get_transforms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 for melanoma classification.")
    parser.add_argument("--train-csv", required=True, help="CSV containing image_name and target.")
    parser.add_argument("--image-dir", required=True, help="Directory containing <image_name>.jpg files.")
    parser.add_argument("--output-dir", default="results/resnet50", help="Directory for checkpoints and history.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = MelanomaDataset(
        train_df, args.image_dir, transform=get_transforms(training=True)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_resnet50(pretrained=True).to(device)
    criterion, class_weights = create_weighted_loss(train_df, device)
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Mixed precision is enabled only when running on CUDA.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    history = {"train_loss": [], "train_accuracy": []}

    print("Starting ResNet50 training with mixed precision.")
    print(f"Training images: {len(train_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Class weights [class 0, class 1]: {class_weights.detach().cpu().tolist()}")

    checkpoint_path = output_dir / "resnet50_latest_checkpoint.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        print(f"\n========== EPOCH {epoch + 1}/{args.epochs} ==========")

        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            processed = batch_index + 1
            if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
                elapsed = time.time() - start_time
                remaining = elapsed / processed * (len(train_loader) - processed)
                print(
                    f"Batch {processed}/{len(train_loader)} | "
                    f"Loss: {running_loss / total:.4f} | "
                    f"Acc: {correct / total:.4f} | "
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"ETA: {remaining / 60:.1f} min"
                )
                torch.save({
                    "epoch": epoch,
                    "batch": processed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "train_loss_so_far": running_loss / total,
                    "train_accuracy_so_far": correct / total,
                }, checkpoint_path)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)

        print(f"Epoch {epoch + 1} complete")
        print(f"Train loss: {epoch_loss:.4f}")
        print(f"Train accuracy: {epoch_accuracy:.4f}")

        # Save a plain state_dict for straightforward evaluation/inference.
        epoch_weights = output_dir / f"resnet50_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_weights)

        # Save the complete epoch-boundary training state as well.
        torch.save({
            "epoch": epoch,
            "batch": len(train_loader),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        }, checkpoint_path)

        torch.save(history, output_dir / "resnet50_training_history.pth")
        with open(output_dir / "resnet50_training_history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        print(f"Saved: {epoch_weights}")

    print("\nTraining finished.")
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
