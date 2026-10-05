from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def predict(
    image_path: Path,
    model_path: Path,
):

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint["classes"]

    model = build_model(
        len(classes)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    weights = (
        EfficientNet_B0_Weights.DEFAULT
    )

    transform = weights.transforms()

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = (
        transform(image)
        .unsqueeze(0)
    )

    with torch.no_grad():

        logits = model(tensor)

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

    confidence, index = (
        probabilities.max(dim=1)
    )

    class_name = classes[
        index.item()
    ]

    return (
        class_name,
        confidence.item(),
        probabilities.squeeze().tolist(),
        classes,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--model",
        default=str(
            PROJECT_ROOT
            / "runs"
            / "floodframe"
            / "best_model.pth"
        ),
    )

    args = parser.parse_args()

    result = predict(
        Path(args.image),
        Path(args.model),
    )

    label, confidence, _, _ = result

    print("=" * 60)
    print("FLOODFRAME")
    print("=" * 60)

    print(
        f"Prediction : {label}"
    )

    print(
        f"Confidence : "
        f"{confidence:.2%}"
    )


if __name__ == "__main__":
    main()