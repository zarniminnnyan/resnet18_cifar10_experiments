
#define the dataset directory
dataset_dir="./data/cifar10"

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
baseline_model_dir="./checkpoints/baseline_eval_weight"

#main model directory 
main_model_dir="./checkpoints/main_model_eval_weight"