from typing import Callable
from config.exp_config import (
    cifar10_classes,
    seed,
    val_fraction,
    cutmix_alpha,
    mixup_alpha,
    cutmix_mixup_probability
    )

from torchvision.datasets import CIFAR10
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.transforms import v2
from torch.utils.data import (
    DataLoader,
    Dataset,
    default_collate,
    random_split
    )

import torch
import tempfile
import time 
import os 

import torchvision.transforms as transform





class TransformSubset(Dataset):
     """
     TransformSubset: the class for splitting the val dataset and train dataset from raw dataset

     Args: 

     Subset (subset dataset): Subset for val dataset or train dataset
     transform (callable): Dataset transformation 

     """
     def __init__(self,Subset,transform=None):
        #splitted subset from raw whole dataset
        self.subset=Subset
        #apply transform
        self.transform=transform
     def __getitem__(self,idx):
        img,label=self.subset[idx]
        #transform  return transformed train dataset OR don't apply augmented transform
        if self.transform:
            #keep the img as img because if not validation to fall back to original img
            img=self.transform(img)
        return img,label

     def __len__(self):
         #get the len of the whole dataset
        return len(self.subset)


def load_resnet18(device:torch.device):

  """
  *load resnet18 model with ImageNet  pretrained weights
  *change the head of classification  from 1k to 10
  *Freeze the backbone of the resnet18
  *Unfreeze the classification layer only

  Args: 
   device (torch.device): cuda or cpu device for model experiments' run 

  Returns: 

   model (nn.Module): pretrained model with freezed backbone layers except fc layer

  """
  model=resnet18(weights=ResNet18_Weights.DEFAULT)

  for name,param in model.named_parameters():
    if "fc" not in name:
      param.requires_grad=False

  #change the head of classification layer
  in_features=model.fc.in_features
  #change the head 1k to 10
  model.fc=torch.nn.Linear(in_features,cifar10_classes)
  #send to device
  model.to(device)

  return model



def split_dataset(raw_dataset,train_transform,val_transform):
    """
    Split the dataset to val dataset and train dataset

    Args:

    raw_dataset (Dataset): the original downloaded raw dataset for train + val (no transformation applied yet)
    train_transform (callable): transform method for train dataset 
    val_transform (callable): transform method for val dataset
    


    Returns: 
      tuple: (train_data_subset,val_data_subset)

      train_data_subset (Subset):Subset of train dataset with applied transform for training 
      val_data_subset (Subset): Subset of val dataset with applied transform for validation

    """
    val_dataset_size=int(len(raw_dataset)*val_fraction)
    #train size
    train_dataset_size=int(len(raw_dataset)-val_dataset_size)

    #split val dataset
    train_subset,val_subset=random_split(
        raw_dataset,[train_dataset_size,val_dataset_size],
        generator=torch.Generator().manual_seed(seed) # seed for reproducibility in ML experiments 

    )

    #apply transformation to val and train based on their transformation respectively
    train_data_subset,val_data_subset=TransformSubset(train_subset,train_transform),TransformSubset(val_subset,val_transform)

    return train_data_subset,val_data_subset




def load_prepare_cifar10(dataset_path:str,build_transforms:Callable,image_size:int,standard_aug:bool):
    """
    This function is for downloading the cifar10 dataset 

    Args: 

    dataset_path (dir): the dataset path to place the dataset after download
    build_transforms (function): dataset transformation function 
    image_size (int): Image input size for feeding into the model
    standard_aug (bool): if True, apply standard augmentation

    Returns: 
    tuple: (train_dataset, val_dataset, test_dataset)

    train_dataset (Subset): Subset for training Model 
    val_dataset (Subset): Subset for Model validation 
    test_dataset (test Dataset): Dataset for testing Model




    """
    #make the dataset path if no exist
    os.makedirs(dataset_path,exist_ok=True)
    #Load transform function
    train_transform,val_transform=build_transforms(image_size=image_size,standardard_augmentation=standard_aug)

    try:
        #Download dataset if the data does not exist, is corrupted or is missing

        #Load train + val dataset
        raw_dataset=CIFAR10(
            root=dataset_path,
            train=True,
            download=True,
            transform=None
            )

        #Extract train and val dataset
        train_dataset,val_dataset=split_dataset(
            raw_dataset,
            train_transform,
            val_transform
            )

        #Load test dataset
        test_dataset=CIFAR10(
            root=dataset_path,
            train=False,
            download=True,
            transform=val_transform
            )


    except Exception as e:
        raise ValueError(f" Raise error as {e}")
    
    print("downloaded cifar10 dataset ...")

    return train_dataset,val_dataset,test_dataset


def build_transforms(image_size:int,standardard_augmentation:bool):

  """
  Apply augmentation to the train dataset
  for better learning. Convert to tensor and apply normalization to
  train dataset,val dataset and test dataset

  Args: 
  image_size (int): Image Input size for model
  standard_augmentation (bool, optional): Augmentation for train dataset

  Returns: 

  tuple - (train_transform,val_test_transform)
  train_transform (callable): transform for train dataset 
  val_test_transform (callable): transform for both val and test dataset


  """

  #train base transform
  train_pipeline=[
        transform.Resize(image_size),
        transform.ToTensor(),
        transform.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]

  if standardard_augmentation:

    #apply standard augmenation 
    train_pipeline=[
        transform.RandomHorizontalFlip(p=0.5),
        transform.RandomRotation(degrees=15),
        *train_pipeline
        ]
    
    
  train_transform=transform.Compose(train_pipeline)

  #val,test transform
  val_test_transform = transform.Compose([
        transform.Resize(image_size),
        transform.ToTensor(),
        transform.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
  return train_transform,val_test_transform


def cutmix_mixup_function(batch):
    """
     Apply Cutmix/Mixup to the train dataset for better learning

     Args:
     batch: dataset batchs 

     Returns: 
     
          a tuple - (images, labels)
          images: torch.Tensor (Batch_size,C,H,W)
          labels: torch.Tensor ( soft labels after cutmix/mixup)
    """
    #MixUp or Cutmix randmly choosing
    mixup=v2.MixUp(alpha=mixup_alpha,num_classes=cifar10_classes)
    cutmix=v2.CutMix(alpha=cutmix_alpha,num_classes=cifar10_classes)

    #cutmix_or_mixup
    cutmix_or_mixup=v2.RandomChoice([mixup,cutmix])
    #apply cutmix/mixup
    augmented_cutmix_or_mixup=v2.RandomApply([cutmix_or_mixup],p=cutmix_mixup_probability)

    # return augmented batch (images, labels)
    return augmented_cutmix_or_mixup(*default_collate(batch))


def build_dataloader(
    batch_size:int,
    num_workers:int,
    use_cutmix_mixup:bool,
    device:torch.device,
    train_dataset,
    val_dataset,
    test_dataset
    ):
 
  """
  Build the dataloader for train/val/test 
  
  Args: 

  batch_size (int): Batch size to be used for val/test/train processes
  num_workers (int): Number of workers to process data faster
  use_cutmix_mixup (bool): Use cutmix/mixup augmentation if set True 
  train_dataset (Subset): Subset from raw dataset 
  val_dataset (Subset): Subset from raw dataset 
  test_dataset (Dataset): Downloaded test set from Cifar10

  Returns:

  tuple - ( train_dataloader, val_dataloader,test_dataloader)

  train_dataloader(Callable): train dataloader for training model 
  val_dataloader(Callable): val dataloader for model validation 
  test_dataloader(Callable): test dataloader for testing model

  """

  active_collate_fn=cutmix_mixup_function if use_cutmix_mixup else None

  #train dataLoader
  train_dataloader=DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      persistent_workers=True if num_workers > 0 else False,
      pin_memory=(device=="cuda"), # set True if device is cuda else False
      collate_fn=active_collate_fn

  )

  #val dataloader
  val_dataloader=DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      persistent_workers=True if num_workers > 0 else False,
      pin_memory=(device=="cuda")  # set True if device is cuda else False
  )

  #test dataloader

  test_dataloader=DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      persistent_workers=True if num_workers > 0 else False,
      pin_memory=(device=="cuda")  # set True if device is cuda else False
  )

  return train_dataloader,val_dataloader,test_dataloader


def get_model_size(model):

    """
    Create the temporary file path to store model with tempfile
    Get model size in MB

    Args:
    model: Resnet18 Pretrained model

    Return:
    size_mb: convert the model size bytes to MB

    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tempFile:
        tempfile_path = tempFile.name

    try:
        #Save model dict
        torch.save(model.state_dict(), tempfile_path)

        #Get Resnet18 pretrained model size in bytes
        model_size_bytes = os.path.getsize(tempfile_path)

        #Get model size in Bytes to MB
        size_mb = model_size_bytes / (1024 * 1024)
    except Exception as e:
        print(f"Failed measuring the model size due to {e}")

    finally:
        #check if file path still exists
        if os.path.exists(tempfile_path):
            #Remove temporary file path
            os.remove(tempfile_path)
    return size_mb


def measure_inference_time(model,device,input_shape,warm_up_runs=10,num_runs=100):
    """
    Measure the inference time of baseline model and  Fine-tuned Resnet50

    Args: 
     model (nn.Module): pretrained resnet18 model 
     device (torch.device): cpu or gpu
     input_shape: the image input for model (eg. (1,3,224,224))
     warm_up_runs: number of warm-up iterations (default 10)
     num_runs: number of timed runs (default 100)

    Returns:
    avg_time: Average inference timing of the model
    """
    #Create sample input data
    sample_input=torch.rand(input_shape).to(device)
    #set eval mode
    model.eval()
    #Run warmup for initializing  system caches and operations
    with torch.no_grad():
        for _ in range(warm_up_runs):
            model(sample_input)

    timings=[]
    with torch.no_grad():
        for _ in range(num_runs):

            #let GPU finishes the work before time was recorded

            if device.type=="cuda":
                torch.cuda.synchronize()
            #Record start time
            time_start=time.perf_counter()
            #feed sample data to the model
            model(sample_input)
            #GPU is runing asynchronous, let the GPU finishes the work before the time was recorded
            if device.type=="cuda":
                torch.cuda.synchronize()

            time_end=time.perf_counter()
            #get the time in seconds and calculate the differences
            timings.append(time_end-time_start)
        #Get the total size and total of the timings
        timings_size=len(timings)
        total_timings=sum(timings)
        #Calculate avg time with average formula and then convert it into miilliseconds
        avg_time=total_timings/timings_size*1000
        return avg_time
