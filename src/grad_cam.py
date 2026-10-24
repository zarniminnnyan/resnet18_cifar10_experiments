import os
import torch
import torchao
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import List
from typing import Any
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
    image_path: str,
    image_name: str,
    temperature_path: str,
    gradcam: Any,
    model: Any,
    classes: List[str],
    experiment_name: str,
    device: torch.device
):
    """
    Process input image, apply temperature scaling, calculate top-5 predictions,
    and generate a Grad-CAM overlay.

    Returns:
        tuple: (pred_idx, pred_conf, top5_dict, fig)
    """
    
    def load_calibrated_temperature_helper():
        temp_file = os.path.join(temperature_path, f"temperature_{experiment_name}.pth")
        if not os.path.exists(temp_file):
            raise FileNotFoundError(f"Temperature file not found: {temp_file}")
        T = torch.load(temp_file, map_location=device)
        return T["optimized_temperature"]

    def predict_class_helper(input_tensor: torch.Tensor):
        """
        Predicts top-1 class info and creates a top-5 dictionary of {class_name: confidence}.
        """
        with torch.no_grad():
            if isinstance(model, nn.Module):
                try:
                    model.eval()
                    model.to(device)
                except NotImplementedError: 
                    torchao.quantization.pt2e.move_exported_model_to_eval(model)
                    model.to(device)
            else:
               # AOTInductor models are already compiled for a specific device/mode.
               # Fall back gracefully if it exposes PyTorch 2.0 graph structures.
               if hasattr(model, "eval"):
                   try:
                       model.eval()
                   except Exception:
                       pass
              
           # AOTICompiledModel:
           # No .eval(), .to(), or PT2E mode conversion needed here.
           # The device is determined when the model is compiled/loaded.
            outputs = model(input_tensor)
            
            T = load_calibrated_temperature_helper()
            probs = torch.softmax(outputs / T, dim=1)
            
            # Top-1 extraction
            pred_idx = probs.argmax(dim=1).item()
            pred_conf = probs[0, pred_idx].item()
            
            # Top-K (5) extraction
            topk_probs, topk_indices = torch.topk(probs, k=5, dim=1)
            
            # Flatten to 1D arrays for easy parsing
            topk_probs = topk_probs[0].tolist()
            topk_indices = topk_indices[0].tolist()
            
            # Construct dictionary mapping class name string to its confidence score
            top5_dict = {classes[idx]: conf for idx, conf in zip(topk_indices, topk_probs)}
            
            return pred_idx, pred_conf, top5_dict

    # ---- Preprocess input image ----
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(os.path.join(image_path, image_name)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get top-1 and top-5 values
    pred_idx, pred_conf, top5_predictions = predict_class_helper(input_tensor)
    
    # Ensure the graph registers gradients for Grad-CAM's backward pass
    input_tensor.requires_grad_()
    
    # ---- Generate Grad-CAM heatmap ----
    heatmap = gradcam.generate(input_tensor)
    
    # ---- Overlay heatmap on image ----
    fig, ax = plt.subplots()
    ax.set_title(f"Predicted: {classes[pred_idx]} | Confidence: {pred_conf*100:.1f} %", color='green', fontsize=14)
    ax.imshow(np.array(img.resize((224, 224))))
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    ax.axis('off')
    
    return pred_idx, pred_conf, top5_predictions, fig
