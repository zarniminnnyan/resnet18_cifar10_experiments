# Project Overview

To see the capacity of the pretrained Resnet18 model if it can be adapted and perform well with cifar10 dataset (32\*32) images by upscaling to (224\*224) images without changing architecture of the model and 
to retain the original pretrained weights and by using the techniques such has transfer learning, learning rate scheduler, data augmentation (RandomHorizontalFlip and RandomRotation), RandomErasing,CutMix, MixUp, and hyperparameter tuning. It evaluates model reliability using Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), Log Loss, and temperature scaling. It also explores deployment optimizations for efficient CPU inference, including Quantization-Aware Training (QAT), Post-Training Quantization (PTQ), the AOTInductor ahead-of-time compiler, and torch.compile() for just-in-time compilation. The model classifies images into 10 CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

