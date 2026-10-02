from src.main_model import Resnet18Lightning
from tqdm.auto import tqdm
from torchvision.models import resnet18
import torch.nn.functional as F
import torchmetrics.classification as metrics
import torch.nn as nn
import torch
import os



def load_model_for_test(
    is_baseline:bool,
    model_dir:str,
    experiment_name:str,
    num_classes:int
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
      baseline_weight_directory=os.path.join(model_dir,"best_baseline_eval_model.pth")
      #load model
      state_dict=torch.load(baseline_weight_directory,map_location="cpu")
      model.load_state_dict(state_dict)
      
    else:
        model=Resnet18Lightning.load_from_checkpoint(os.path.join(model_dir,f"{experiment_name}_weight.ckpt"))
        
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
  test_acc.reset()


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

  #compute accuracy
  total_test_accuracy=test_acc.compute().item()*100
  #compute total loss
  total_test_loss=total_loss/len(test_dataloader)

  test_history={
    "test_acc":total_test_accuracy,
    "test_loss":total_test_loss
  }

  torch.save(
    test_history,
    f"{test_result_dir}/test_history.pth"
  )


  return test_history

