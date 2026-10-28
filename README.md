# Project Overview

This project explores whether a pretrained ResNet-18 model can be effectively adapted for the CIFAR-10 dataset while keeping its original architecture and pretrained ImageNet weights unchanged. Since CIFAR-10 images are only 32×32 pixels, they are first upscaled to 224×224 pixels to match the model's expected input size.

To improve performance, the model is fine-tuned using transfer learning along with several training and regularization techniques, including **learning rate scheduling, data augmentation (Random Horizontal Flip and Random Rotation), Random Erasing, MixUp, CutMix, and hyperparameter tuning**. The goal is to determine how well a large pretrained model can generalize to a smaller image dataset without architectural modifications.

Beyond classification accuracy, the project also evaluates the model's confidence and reliability using calibration metrics such as **Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Log Loss. Temperature scaling** is applied as a post-processing technique to improve probability calibration and produce more trustworthy predictions.

For deployment, the project investigates multiple optimization strategies to improve CPU inference efficiency, including **Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), `torch.compile()`, and the AOTInductor ahead-of-time compiler**. These techniques are compared to understand the trade-offs between model size, inference speed, and predictive performance.

The final model is capable of classifying images into the ten CIFAR-10 categories: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.**


## Built with
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchAO](https://img.shields.io/badge/TorchAO-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Lightning AI](https://img.shields.io/badge/Lightning_AI-792EE5?style=for-the-badge&logo=lightning&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF9A00?style=for-the-badge&logo=gradio&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

## validation Accuracy comparison across Experiments 
| Experiment | Modification | Val Acc (%) | Improvement Over Baseline |
|------------|-------------|-------------|---------------------------|
| Baseline | Pretrained Resnet18 + Unfreezing FC head and modified the head + Minimal training | 79.00 | • Baseline |
| Experiment 2 | + Unfreezing layer4+fc layers (learning rates= (1e-4,1e-3)) | 87.96 | 🔺 +8.96% |
| Experiment 3 | + Data Augmentation (RandomHorizontalFlip (p=0.5)+RandomRotation(degrees=15) + (learning rates= (1e-4,1e-3)) | 90.51 | 🔺 +11.51% |
| Experiment 4 version 1 | + Cosine Annealing (learning rates= (1e-4,1e-3), layers=(fc,layer4)) | 88.66 | 🔺 +9.66% |
| Experiment 4 version 2 | + Cosine Anealing (higher learning rates (3e-4,3e-3),layers=(fc,layer4)) | 91.10 | 🔺 +12.10% |
| Experiment 5 version 1 | + Cutmix/Mixup (learning rates (3e-4,3e-3),layers=(fc,layer4)) | 89.00 | 🔺 +10.00% |
| Experiment 5 version 2 | + unfreezing layer3 (lr=2e-4)+layer4(lr=3e-4)+ fc(lr=3e-3) | 94.44 | 🏆 +15.44% |
| Experiment 5 version 3 | + random Erasing augmentation + adding cosine annealing warmstart (replaced standard cosine annealing) | 95.14 | 🏆 +16.14% |


### CutMix
Combines two training images by replacing a random patch of one image with a patch from another. The labels are mixed proportionally to the patch area, helping improve robustness and generalization.

### MixUp
Generates new training samples by blending two images and their labels. This acts as a regularizer, reducing overfitting and encouraging smoother decision boundaries.

### Random Erasing

Random Erasing is a data augmentation technique that randomly masks a small region of an image during training. This encourages the model to rely on multiple visual features rather than specific image regions, improving robustness and reducing overfitting.

## Per-Class Classification Reports of Best candidates (Experiment 5 v 2 and v 3 ) and Baseline
<table>
<tr>
<td align="center">
<b>Baseline</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/exp1_baseline_classification_report.png" width="450">
</td>

<td align="center">
<b>Experiment 5 Version 2</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/exp5_v2_classification_report.png" width="450">
</td>
</tr>

<tr>
<td colspan="2" align="center">
<b>Experiment 5 Version 3 (Final Model)</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/exp5_v3_classification_report.png" width="450">
</td>
</tr>
</table>

## Per-Class Accuracy Comparison

| Class | Baseline | Exp 5 V2 | Exp 5 V3 |
|---------|---------:|---------:|---------:|
| Airplane | 81.7 | 94.2 | 96.2 |
| Automobile | 88.1 | 96.3 | 97.1 |
| Bird | 73.9 | 92.3 | 93.2 |
| Cat | 63.8 | 86.6 | 88.1 |
| Deer | 74.6 | 92.0 | 93.7 |
| Dog | 75.0 | 90.1 | 89.8 |
| Frog | 82.4 | 97.1 | 96.8 |
| Horse | 82.1 | 94.4 | 95.3 |
| Ship | 84.3 | 95.7 | 96.6 |
| Truck | 86.7 | 95.7 | 96.3 |

## Confusion Matrix Comparison

<table>
<tr>
<td align="center">
<b>Baseline</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/baseline_confusion_matrix.png" width="450">
</td>

<td align="center">
<b>Experiment 5 Version 2</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/exp5_v2_confusion_matrix.png" width="450">
</td>
</tr>

<tr>
<td align="left">
<b>Experiment 5 Version 3 (Final Model)</b><br>
<img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/exp5_v3_confusion_matrix.png" width="450">
</td>

<td></td>
</tr>
</table>

## Model Calibration Evaluation

The final model's calibration performance was evaluated using **Expected Calibration Error (ECE)**, **Negative Log-Likelihood (NLL)**, and **Log Loss**. Lower values indicate better calibrated and more reliable confidence estimates.

<p align="center">
  <img src="https://raw.githubusercontent.com/zarniminnnyan/resnet18_cifar10_experiments/main/gallaries/calibration.png" width="850"/>
</p>

## Calibration Comparison (After Temperature Scaling)

| Model | Validation Accuracy (%) | ECE ↓ | Log Loss ↓ | NLL ↓ | Rank |
|---------|---------:|---------:|---------:|---------:|---------|
| Baseline | 79.00 | 0.012 | 0.559 | 0.559 | 🥉 |
| Exp 5 V2 | 94.44 | 0.010 | 0.162 | 0.162 | 🥈 |
| Exp 5 V3 | 95.14 | 0.012 | 0.146 | 0.146 | 🥇 Best Overall |

- Lower ECE indicates better calibration.
- Lower Log Loss and NLL indicate better confidence quality.
- Exp 5 V2 achieved the lowest ECE (0.010).
- Exp 5 V3 achieved the lowest Log Loss (0.146) and NLL (0.146).
- Exp 5 V3 also achieved the highest validation accuracy (95.14%).

## Model Selection Summary

| Model | Val Acc (%) | Test Acc (%) | ECE ↓ | Log Loss ↓ | NLL ↓ | Verdict |
|---------|---------:|---------:|---------:|---------:|---------:|---------|
| Baseline | 79.00 | 79.26 | 0.012 | 0.559 | 0.559 | Baseline |
| Exp 5 V2 | 94.44 | 93.54 | 🏆 0.010 | 0.162 | 0.162 | Excellent calibration |
| Exp 5 V3 | 🏆 95.14 | 🏆 94.31 | 0.012 | 🏆 0.146 | 🏆 0.146 | 🥇 Best Overall |

Overall, Exp 5 Version 3 was selected as the best model because it provided the highest classification accuracy and also in test while maintaining excellent calibration performance after temperature scaling.




