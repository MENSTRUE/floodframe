from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "floodframe"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_DIR),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    return parser.parse_args()


def create_transforms():

    weights = EfficientNet_B0_Weights.DEFAULT

    val_transform = weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.8, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def build_model(num_classes: int):

    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def calculate_accuracy(
    outputs,
    labels,
):

    predictions = outputs.argmax(dim=1)

    correct = (
        predictions == labels
    ).sum().item()

    return correct


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        total_correct += calculate_accuracy(
            outputs,
            labels,
        )

        total_samples += images.size(0)

    avg_loss = (
        total_loss
        / total_samples
    )

    avg_accuracy = (
        total_correct
        / total_samples
    )

    return avg_loss, avg_accuracy


def main():

    args = parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train directory tidak ditemukan: {train_dir}"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory tidak ditemukan: {val_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("FLOODFRAME TRAINING")
    print("=" * 70)

    print(f"Device : {device}")
    print(f"Data   : {data_dir}")

    train_transform, val_transform = (
        create_transforms()
    )

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform,
    )

    class_names = train_dataset.classes

    print()
    print("Classes:")
    print(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = build_model(
        num_classes=len(class_names)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )

    best_accuracy = 0.0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_loss, train_acc = (
            run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
            )
        )

        val_loss, val_acc = (
            run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),
                "classes":
                    class_names,
                "image_size":
                    IMAGE_SIZE,
                "model_name":
                    "efficientnet_b0",
            }

            torch.save(
                checkpoint,
                output_dir / "best_model.pth",
            )

            print(
                "[OK] best_model.pth updated"
            )

    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()