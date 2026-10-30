from pprint import pprint
import argparse
import torch

from src.grad_cam import process_input_image
from config.exp_config import cifar10_class_names
from app import _get_gradcam



# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def configure_arguments():
    """Configure command-line options for prediction."""

    parser = argparse.ArgumentParser(
        description="Prediction test configuration"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="./",
        help="Directory containing the input image"
    )

    parser.add_argument(
        "--image_name",
        type=str,
        required=True,
        help="Image file name to predict"
    )

    parser.add_argument(
        "--add_temperature",
        action="store_true",
        help="Apply temperature scaling during inference"
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
        default="./temperature",
        help="Directory containing saved temperature values"
    )

    return parser.parse_args()


if __name__ == "__main__":

    print("=== Running Prediction ===\n")

    # Read user configuration from CLI
    args = configure_arguments()

    # Load model and Grad-CAM
    model, gradcam = _get_gradcam()

    # Process image and generate prediction
    _, _, top5, _, _ = process_input_image(
        image_path=args.image_path,
        image_name=args.image_name,
        add_temperature=args.add_temperature,
        gradcam=gradcam,
        model=model,
        classes=cifar10_class_names,
        experiment_name=args.experiment_name,
        temperature_path=args.temperature_path,
        device=device,
    )

    print("\n----- Top-5 Predictions (Highest score is the predicted class) --------")
    pprint(top5)