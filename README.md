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
