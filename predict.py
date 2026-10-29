from src.grad_cam import process_input_image
from app import _get_gradcam
import argparse
import torch

cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def configurable_arguments():
    """Configure options for prediction test"""

    parser = argparse.ArgumentParser(
        description="Prediction test configurable options"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="./",
        help="Images directory"
    )

    parser.add_argument(
        "--image_name",
        type=str,
        required=True,
        help="Image name to predict"
    )

    parser.add_argument(
        "--add_temperature",
        action="store_true",
        help="Apply temperature scaling"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="exp5_v3",
        help="Experiment name"
    )

    parser.add_argument(
        "--temperature_path",
        type=str,
        default=None,
        help="Path to saved temperature value"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = configurable_arguments()

    _model, _gradcam = _get_gradcam()

_, _, top5,_, _=process_input_image(
        image_path=args.image_path,
        image_name=args.image_name,
        add_temperature=args.add_temperature,
        gradcam=_gradcam,
        model=_model, #Note: model file name should be same as mine since it's hardcoded as {experiment_name}_weight.ckpt
        classes=cifar10_classes,
        experiment_name=args.experiment_name, # e.g temperature_{experiment_name}.pth
        temperature_path=args.temperature_path,
        device=device,
    )

print(f"Top-5 scores for model prediction. Highest score is the predicted class: {top5}")