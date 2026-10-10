import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
import numpy as np


class GradCAM:
    # ---- Grad-CAM class ----
    """
    Grad-CAM implementation for visualizing class-specific
    regions of interest in resnet18 model.

    Attributes:
        model (nn.Module): The PyTorch model (resnet18) to analyze.
        target_layer (nn.Module): The layer to hook for Grad-CAM.
        activations (torch.Tensor): Activations captured during forward pass
        gradients (torch.Tensor): Gradients captured during backward pass.
    """
    def __init__(self, model:nn.Module, target_layer:nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Forward hook: save activations
        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Backward hook: save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach() #grad_output[0] because the grad_ouput shape is ([1,channel,height,weight])

        # Register hooks (At the forward pass forward_hook will be called automatically once it's registered)
        # (At the backward pass at loss.backward(), the backward_hook will be called automatically too)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):

        """
        Generate a Grad-CAM heatmap for a given input and class.

        Args:
            input_tensor (torch.Tensor): Input image tensor of shape [B, C, H, W].
            target_class (int, optional): Index of the target class. Defaults to
            the predicted class.

        Returns:
            numpy.ndarray: Normalized heatmap of shape [H, W].
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Compute weights: average gradients spatially (Height*Weight)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activations (for knowing the importance of activations)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # Apply ReLU (for keeping just important feature maps)
        cam = torch.relu(cam)

        # Normalize and upsample to input size (bilinear is for smooth upsampling...by using weighted average nearby pixels)
        cam = nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy() #remove all size one dimensions
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize 0-1 with min‑max normalization formula
        return cam


def process_input_image(
    image_path:str,
    image_name:str,
    temperature_path:str,
    gradcam:GradCAM,
    model:nn.Module,
    classes:List,
    experiment_name:str,
    device:torch.device
    ):
    """
    process input image for converting to tensor with ImageNet normalization and shape (224,224)

    Args: 
       image_path(str): The path of real world image for testing Grad cam  
       image_name (str): Name of images (eg. .png,.jpg etc .)
       gradcam (GradCam Class): The GradCam class
       model (nn.Module): Resnet18 Model 
       classes: List of classes 
       experiment_name (str):Experiment name
       device (torch.device): cpu or gpu

    Returns: 
       The overlayed Gram Cam plot on image
    """
    def load_calibrated_temperature_helper(): 

        """
        load the calibrated temperature

        Returns: 
          optimized_temperature: Temperature which was saved during model calibration

        """
        temp_file=os.path.join(temperature_path,f"temperature_{experiment_name}.pth")

        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Temperature file not found: {temp_file}")

        T=torch.load(temp_file)
        #i have applied .item() for getting scalar when saving
        return T["optimized_temperature"]

    def predict_class_helper(input_tensor:torch.Tensor):
        
        """
            Predict the class index and confidence score for a given input tensor.

            Args:
                input_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].

            Returns:
                tuple: (pred_idx, pred_conf)
                    pred_idx (int): Index of the predicted class.
                    pred_conf (float): Confidence score (probability) of the predicted class.
        """

        with torch.no_grad():
            #model send to device and set eval model 
            model.eval().to(device)
            outputs=model(input_tensor)
            #Call temperature helper 
            T=load_calibrated_temperature_helper()
            #convert it to probabilites to show confidence score 
            probs=torch.softmax(outputs/T,dim=1)
            #get the idx of predicted class 
            pred_idx=probs.argmax(dim=1).item()
            #get the confidence score 
            pred_conf=probs[0,pred_idx].item()

            return pred_idx,pred_conf


    # ---- Preprocess input image ----
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ) 
            ])
    
    img = Image.open(os.path.join(image_path,image_name)).convert("RGB")

    #apply the transform and add batch 1 at the first dim
    input_tensor = transform(img).unsqueeze(0).to(device)
    #call the predicted class helper 
    pred_idx,pred_conf=predict_class_helper(input_tensor)

    #Ensure the graph registers gradients for Grad-CAM's backward pass
    input_tensor.requires_grad_() 

    # ---- Generate Grad-CAM heatmap ----
    heatmap = gradcam.generate(input_tensor)

    # ---- Overlay heatmap on image ----
    fig, ax = plt.subplots() #(this is more convinient to be used in Gradio instead of just plotting in the notebook)
    ax.set_title(f"Predicted: {classes[pred_idx]} | Confidence: {pred_conf*100:.1f} %",
          color='green', fontsize=14)

    ax.imshow(np.array(img.resize((224,224))))
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    return pred_idx,pred_conf,fig
