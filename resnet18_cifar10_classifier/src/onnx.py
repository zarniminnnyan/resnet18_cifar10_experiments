import os
import torch 
import torch.nn as nn
import onnxruntime as ort
from typing import Tuple

def export_to_onnx(
    best_model:nn.Module,
    dummy_input_shape:Tuple[int],
    export_path:str,
    onnx_name:str,
    device:torch.device
    ): 

    """
    Export a trained PyTorch model to ONNX format.

    Args:
        best_model (nn.Module): The trained PyTorch model to export.
        dummy_input_shape (Tuple[int]): Shape of the dummy input tensor 
                                        (e.g., (1, 3, 224, 224)).
        export_path (str): Directory path where the ONNX file will be saved.
        onnx_name (str): Name of the ONNX file.
        device (torch.device): Target device (CPU or GPU) for export.

    Returns:
        None. Saves the ONNX model file to the specified path.
    """
    
    best_model.eval().to(device)
    #create dir if no exists
    os.makedirs(export_path,exist_ok=True)
    onnx_path=os.path.join(export_path,f"{onnx_name}.onnx")
    #create dummy data 
    dummy_input=torch.randn(dummy_input_shape).to(device)
    try:
        #export the best candidate to onnx
        torch.onnx.export(
            best_model,           
            dummy_input,                #Define input shape 
            onnx_path,                  #Onnx file name
            export_params=True,          #Include trained weights
            opset_version=11,            #Stable ONNX opset
            do_constant_folding=True,    #Optimize constant ops
            input_names=['input'],       #Define input/output for easier model loading in inference
            output_names=['output'], 
            dynamic_axes={
                'input': {0: 'batch_size'}, #Define the batch size input to be able to set flexible values
                'output': {0: 'batch_size'}
                }
                 )
        print("Exported to ONNX!")
    except Exception as e: 
        raise ValueError(f"Failed exporting to ONNX. Found {e} error")