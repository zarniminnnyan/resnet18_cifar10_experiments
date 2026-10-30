
#define the dataset directory
dataset_dir="./dataset"

#metrics_logs_dir 
logs_dir="./main_model_logs/logs"

#cifar10 classes 
cifar10_classes=10

#seed for reproducibility of ML experiments 
seed=42 
#validation dataset Subset fraction from raw dataset 
val_fraction=0.2

# alpha controls the Beta distribution for the mixing ratio:
# - Small alpha (<1): one sample dominates (less mixing)
# - alpha = 1: wide variety of random mixes (all ratios equally likely)
# - Large alpha (>1): contributions are forced to be balanced (~50/50)

cutmix_alpha=0.8
mixup_alpha=0.4

#cutmix_mixup_adding_probability 
cutmix_mixup_probability=0.5 #every 50% chance, add cutmix/mixup augmentation

#directories 
#resume directory for resuming baseline training
resume_dir="./checkpoints/baseline_resume_checkpoints"
#evaluated baseline weight directory 
baseline_model_dir="./baseline_model"

#main model directory 
main_model_dir="./checkpoints/main_model_eval_weight"

#CONFIGURATION FOR GRADIO APP ONLY

# Default experiment used throughout the project
EXPERIMENT_NAME = "exp5_v3"

# Directory containing saved temperature values
TEMPERATURE_PATH = "./temperature"
ADD_TEMPERATURE=False


# CSV file paths
# Note: When training is resumed multiple times with PyTorch Lightning ModelCheckpoint,
# metrics are saved as separate CSV files. Download and add each CSV file path manually.
csv_file_path="/models/results"

# CIFAR-10 class labels
cifar10_class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
