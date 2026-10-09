from src.main_model import Resnet18Lightning,resnet18_main_model_setup
from tqdm.auto import tqdm
from torchvision.models import resnet18
import torch.nn.functional as F
import torchmetrics.classification as metrics
import torch.nn as nn
import torch
import pickle
import numpy as np
import os


def get_parameters_for_lightning_model_inference(
    model_dir: str,
    experiment_name: str,
    device: torch.device
):
    """
    Extracts training configurations and reconstructs parameters from a legacy checkpoint.

    NOTE: This manual extraction function is explicitly required for previously trained 
    legacy models because the original `__init__` constructor used `ignore=['model']` 
    and added the change_to_warmstart_scheduler argument after training from exp1 to exp5,
    this breaks lighning's original 
    model loading style

    RECOMMENDATION FOR FUTURE TRAINING RUNS:

    To bypass this extraction function entirely in future experiments, avoid passing raw 
    `nn.Module` objects or live parameter groups into `__init__`. Instead, pass primitive 
    datatypes (e.g., `learning_rates: list`) and build the model architecture inside
    the LightningModule constructor. Doing so will enable simple, one-line loading 
    via `Resnet18Lightning.load_from_checkpoint("path.ckpt")`.

    Args:
        model_dir (str): Root directory where experiments are stored.
        experiment_name (str): Specific experiment folder and weight name prefix.
        device (torch.device): Compute device target (CPU or GPU).

    Returns:
        tuple: (base_model,change_to_warmstart_scheduler)
    """
    ckpt_path = os.path.join(f"{model_dir}/{experiment_name}", f"{experiment_name}_weight.ckpt")
    
    # Load raw checkpoint safely to device memory
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract successfully logged primitive configurations
    saved_hparams = checkpoint.get("hyper_parameters", {})
    print("--- Extracting Configurations ---")
    
    try:
        layers = saved_hparams.get("layers",["layer4","fc"])
       
        change_to_warmstart_scheduler = saved_hparams.get("change_to_warmstart_scheduler", False)
        
        optimizer_states = checkpoint.get("optimizer_states", [])
        if optimizer_states:
            learning_rates = [group["lr"] for group in optimizer_states[0]["param_groups"]]
        else:
            raise KeyError("optimizer_states empty or missing in the checkpoint file.")

        print(f"Extracted learning rates from training: {learning_rates}")
        
    except (KeyError, IndexError, TypeError) as e:
        print(f"Extraction failed: Critical training parameters are missing from this checkpoint.")
        raise e
        
    # Rebuild the missing underlying PyTorch architecture manually from extracted configs
    base_model = resnet18_main_model_setup(device=device, trainable_layers_group=layers)

    return base_model,change_to_warmstart_scheduler,layers


def load_model_for_test(
    is_baseline:bool,
    model_dir:str,
    experiment_name:str,
    num_classes:int,
    device:torch.device
    ):
    
    """
    Load a ResNet18 model for testing.

    Args:
        is_baseline (bool): If True, load baseline weights (.pth).
                            If False, load Lightning checkpoint (.ckpt).
        model_dir (str): Directory containing model weights.
        experiment_name (str): Experiment name used for checkpoint file.
        num_classes (int): Number of output classes for classification.

    Returns:
        nn.Module: ResNet18 model with loaded weights.
    """
    #change the head
    model=resnet18(weights=None)
    in_features=model.fc.in_features
    model.fc=nn.Linear(in_features,num_classes)

    print(f"change the head of resnet18 to {model.fc.out_features}")

    #load the existed model weights
    if is_baseline:
      #weight dir
      baseline_weight_directory=os.path.join(f"{model_dir}/baseline","best_baseline_eval_model.pth")
      #load model
      state_dict=torch.load(baseline_weight_directory,map_location=device)
      model.load_state_dict(state_dict)
      
    else:
        base_model,change_to_warmstart_scheduler,layers=get_parameters_for_lightning_model_inference(
            model_dir=model_dir,
             experiment_name=experiment_name,
             device=device
             )     
        
        model=Resnet18Lightning.load_from_checkpoint(
            checkpoint_path=os.path.join(f"{model_dir}/{experiment_name}",f"{experiment_name}_weight.ckpt"),
            model=base_model,
            change_to_warmstart_scheduler=change_to_warmstart_scheduler,
            layers=layers
            )
        
    return model



def test_model(
    model:nn.Module,
    test_dataloader,
    num_classes:int,
    device:torch.device,
    test_result_dir:str
        ):

  """
  test the model on test dataset
  Apply torchmetrics for measuring the metrics

  Args: 
   test_dataloader (Callable): Dataloader for test dataset 
   num_classes: Cifar10 classes (10 classes)
   device(torch.device): CPU or GPU
   model (nn.Module): resnet18 model with 10 classes
  
  Returns:

   tuple - test_history (dictionary)
   
   test_accuracy_collection : test accuracy of the model
   test_loss_collection: test loss of the model    
  """

  os.makedirs(test_result_dir,exist_ok=True)

  test_acc=metrics.Accuracy(task="multiclass",num_classes=num_classes).to(device)
  cm_metric =metrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)
  test_f1score=metrics.F1Score(task="multiclass",num_classes=num_classes,average=None).to(device)
  test_precision=metrics.Precision(task="multiclass",num_classes=num_classes,average=None).to(device)
  test_recall=metrics.Recall(task="multiclass",num_classes=num_classes,average=None).to(device)
  test_acc_per_class=metrics.Recall(task="multiclass",num_classes=num_classes,average=None).to(device)

  # reset the metrics 
  test_acc.reset()
  test_f1score.reset()
  test_precision.reset()
  test_recall.reset()
  test_acc_per_class.reset()


  #total loss
  total_loss=0.0

  #pass dataloader into tqdm
  test_dataloader=tqdm(test_dataloader,desc="test model is in progress")


 #send model to the device and set the eval mode
  model.to(device)
  model.eval()

  #turn off the updating gradients
  with torch.no_grad():

    for images,labels in test_dataloader:
      img,label=images.to(device),labels.to(device)
      #predict
      outputs=model(img)
      #loss
      loss=F.cross_entropy(outputs,label)
      #accumulate the loss
      total_loss +=loss.item()
      #update the test acc in val accuracy metrics
      test_acc.update(outputs,label)
      #test acc per class 
      test_acc_per_class.update(outputs,label)
      #update the cm metric
      cm_metric.update(outputs,label)
      #update test precision 
      test_precision.update(outputs,label)
      #update the test f1 score 
      test_f1score.update(outputs,label)
      #update the test recall
      test_recall.update(outputs,label)

  #compute accuracy
  total_test_accuracy=f"{test_acc.compute().item()*100:.2f}"
  #compute per class test accuracy 
  total_test_per_class_acc=np.round(test_acc_per_class.compute().cpu().numpy()*100,2)
  #compute per class precision 
  total_test_precision=np.round(test_precision.compute().cpu().numpy()*100,2)
  #compute per class f1 score 
  total_test_f1_score=np.round(test_f1score.compute().cpu().numpy()*100,2)
  #compute per class recall 
  total_test_recall=np.round(test_recall.compute().cpu().numpy()*100,2)
  #compute confusion matrix
  conf_matrix = cm_metric.compute().cpu().numpy()
  #compute total loss
  total_test_loss=f"{total_loss/len(test_dataloader):.2f}"

  test_history={
    "test_acc":total_test_accuracy,
    "test_per_class_acc":total_test_per_class_acc,
    "test_loss":total_test_loss,
    "conf_matrix":conf_matrix,
    "precision":total_test_precision,
    "f1score":total_test_f1_score,
    "recall":total_test_recall
  }
  #saved with pickle because it is much cleaner than torch.save() due to numpy() arrays
  with open(f"{test_result_dir}/test_history.pkl", "wb") as f:
    pickle.dump(test_history, f)


  return test_history
