from pathlib import Path
import argparse

import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "floodframe_efficientnet_b0.pth"
)


def build_model(
    num_classes: int,
):

    model = efficientnet_b0(
        weights=None
    )

    in_features = (
        model
        .classifier[1]
        .in_features
    )

    model.classifier[1] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def build_transform(
    checkpoint: dict,
):

    image_size = checkpoint.get(
        "image_size",
        224,
    )

    normalization = checkpoint.get(
        "normalization",
        {},
    )

    mean = normalization.get(
        "mean",
        [
            0.485,
            0.456,
            0.406,
        ],
    )

    std = normalization.get(
        "std",
        [
            0.229,
            0.224,
            0.225,
        ],
    )

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    image_size,
                    image_size,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
):

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    classes = checkpoint[
        "classes"
    ]

    model = build_model(
        num_classes=len(classes)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    transform = build_transform(
        checkpoint
    )

    return {
        "model": model,
        "classes": classes,
        "transform": transform,
        "checkpoint": checkpoint,
        "model_path": model_path,
    }


def predict_image(
    image: Image.Image,
    model_data: dict,
):

    model = model_data[
        "model"
    ]

    classes = model_data[
        "classes"
    ]

    transform = model_data[
        "transform"
    ]

    image = image.convert(
        "RGB"
    )

    tensor = (
        transform(
            image
        )
        .unsqueeze(0)
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities_tensor = (
            torch.softmax(
                logits,
                dim=1,
            )[0]
        )

    confidence, index = (
        probabilities_tensor.max(
            dim=0
        )
    )

    prediction = classes[
        index.item()
    ]

    probabilities = (
        probabilities_tensor
        .cpu()
        .tolist()
    )

    return (
        prediction,
        float(
            confidence.item()
        ),
        probabilities,
        classes,
    )


def predict(
    image_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
):

    model_data = load_model(
        model_path
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    return predict_image(
        image=image,
        model_data=model_data,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "FloodFrame local image inference"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path gambar yang ingin diuji",
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help="Path model FloodFrame",
    )

    args = parser.parse_args()

    (
        label,
        confidence,
        probabilities,
        classes,
    ) = predict(
        image_path=Path(
            args.image
        ),
        model_path=Path(
            args.model
        ),
    )

    print(
        "=" * 60
    )

    print(
        "FLOODFRAME"
    )

    print(
        "=" * 60
    )

    print(
        f"Prediction : {label.upper()}"
    )

    print(
        f"Confidence : {confidence:.2%}"
    )

    print()

    print(
        "Probabilities:"
    )

    for class_name, probability in zip(
        classes,
        probabilities,
    ):

        print(
            f"- {class_name}: "
            f"{probability:.2%}"
        )


if __name__ == "__main__":

    main()