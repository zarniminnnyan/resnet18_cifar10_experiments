from typing import List
import argparse
import torch

from src.calibrate import calibrate_eval_model
from src.test import load_model_for_test
from utils.helper_utils import (
    build_transforms,
    load_prepare_cifar10,
)
from config.exp_config import dataset_dir


# Use GPU if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def configure_params():
    """Configure temperature scaling options."""

    parser = argparse.ArgumentParser(
        description="Temperature scaling configuration"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="exp5_v3",
        help="Experiment name to calibrate"
    )

    parser.add_argument(
        "--temperature_path",
        type=str,
        default="./temperature",
        help="Directory used to save calibrated temperature values"
    )

    return parser.parse_args()


def get_datasets(
    add_standard_augmentation: bool,
    add_random_erasing: bool
):
    """Download CIFAR-10 if needed and prepare train/val/test splits."""

    train_dataset, val_dataset, test_dataset = load_prepare_cifar10(
        dataset_path=dataset_dir,
        build_transforms=build_transforms,
        image_size=224,
        standard_aug=add_standard_augmentation,
        add_random_erasing=add_random_erasing,
    )

    return train_dataset, val_dataset, test_dataset


def load_models(
    model_names: List[str],
    experiment_name: str,
):
    """Load selected models for temperature calibration."""

    models = {}

    for name in model_names:

        models[name] = load_model_for_test(
            is_baseline=(name == "baseline"),
            model_dir="./",
            experiment_name=experiment_name,
            num_classes=10,
            device=device,
        )

    return models


if __name__ == "__main__":

    print("=== Temperature Calibration ===\n")

    # Read user configuration from CLI
    args = configure_params()

    # Keep these values available throughout the script
    EXPERIMENT_NAME = args.experiment_name
    TEMPERATURE_PATH = args.temperature_path

    # Load the model that will be calibrated
    models = load_models(
        model_names=[EXPERIMENT_NAME],
        experiment_name=EXPERIMENT_NAME,
    )

    # Only the validation set is needed for temperature scaling
    _, val_dataset, _ = get_datasets(
        add_standard_augmentation=True,
        add_random_erasing=True,
    )

    # Find the best temperature and save it for future inference
    results = calibrate_eval_model(
        models=models,
        val_dataset=val_dataset,
        batch_size=128,
        device=device,
        split_size=5000,
        bins_size=15,
        initial_T=1.5,
        temperature_path=TEMPERATURE_PATH,
    )

    print("\n=== Calibration Complete ===\n")
    print(f"Optimized temperature: {results["optimized_temperature"]:.2f}")