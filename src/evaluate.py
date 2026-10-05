from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data"),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            PROJECT_ROOT
            / "runs"
            / "floodframe"
            / "best_model.pth"
        ),
    )

    return parser.parse_args()


def build_model(num_classes):

    model = efficientnet_b0(
        weights=None
    )

    in_features = (
        model.classifier[1].in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def main():

    args = parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)

    test_dir = data_dir / "test"

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory tidak ditemukan: {test_dir}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint["classes"]

    weights = EfficientNet_B0_Weights.DEFAULT

    transform = weights.transforms()

    dataset = datasets.ImageFolder(
        test_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
    )

    model = build_model(
        len(classes)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = (
                outputs.argmax(dim=1)
            )

            y_true.extend(
                labels.tolist()
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("=" * 70)
    print("FLOODFRAME EVALUATION")
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=classes,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    display.plot()

    output_path = (
        PROJECT_ROOT
        / "assets"
        / "confusion_matrix.png"
    )

    plt.title(
        "FloodFrame Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    print()
    print(
        f"Confusion matrix saved: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()