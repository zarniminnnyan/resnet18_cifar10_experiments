from torchvision import transforms
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd
import torch
import glob
import os


def apply_unnormalized(image): 
    """ 
    Apply unnormalization to the image before plotting

    Args:
       image: normalized image

    Returns: 
      image: unnormalized image 

    """
    mean=torch.tensor([0.485, 0.456, 0.406],device=image.device,dtype=image.dtype)
    std=torch.tensor([0.229, 0.224, 0.225],device=image.device,dtype=image.dtype)

    #apply broadcasting to std and mean to be able to multiply with normalized image 
    # ( eg. (3) to (3,1,1) to be able to multiply with (3,224,224))
    return image*std[:,None,None]+mean[:,None,None]
        

def sample_image(dataset,test_set:bool):

  """
  show 15 images for train dataset

  Args: 
   dataset (Subset): train dataset  from the raw dataset

  Returns: 
   - the plot images of train dataset
  
  """

  fig,axes=plt.subplots(3,5,figsize=(10,8))
  axes=axes.flatten()
  to_tensor=transforms.ToTensor()

  dataset_classes=dataset.subset.dataset.classes

  if test_set:
    print(f"test_set - True is  only for plotting the test dataset.")
    dataset_classes=dataset.classes


  for idx in range(15):
    img,label=dataset[idx] #to access normalized images

    #if the image is not tensor,convert it into tensor to apply unnormalization 
    if not isinstance(img,torch.Tensor): 
      img=to_tensor(img)

    # convert from CxHxW to HxW
    # xC as matplotlib expects H,W,C
    axes[idx].imshow(apply_unnormalized(img).permute(1, 2, 0).clamp(0,1).cpu().numpy())  

    axes[idx].set_title(dataset_classes[label]) #to access the classes names from the original subset
    axes[idx].axis("off")

  plt.show()


def plot_train_and_val_metrics(
    train_collection:list,
    val_collection:list,
    train_legend_name:str,
    val_legend_name:str,
    y_label:str,
    plot_title:str,
    epochs:int,
    experiment_name:str
    ):

    """ 
    This function is used for plotting the train accuracy and val accuracy, train loss and val loss 

    Args: 

       train_collection (list): a list for train accuracy/loss
        val_collection (list) : a list for val accuracy/loss
        train_legend_name (str): train loss/accuracy label name on plot
        val_legend_name (str):  val loss/accuracy label name on plt
        y_label (str): y axis label name (Accuracy % or Loss )
        plot_title (str): the Plot title
        epochs (int): number of epochs

    """


    plt.figure(figsize=(10, 6))
    plt.plot(range(1,epochs+1), train_collection, label=f'{train_legend_name}', color='#1f77b4', linewidth=2)
    plt.plot(range(1,epochs+1), val_collection, label=f'{val_legend_name}', color='#ff7f0e', linewidth=2)

    # Add titles and labels
    plt.title(f'{plot_title} for experiment- {experiment_name}')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(f'{y_label}', fontsize=12)


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')
    plt.tight_layout()
    plt.show()



def confusion_matrix_plot(final_matrix,classes_name_list:list, experiment_name:str):

  """
  The function is used for plotting confusion matrics on testset of baseline and exp5

  Args: 
  
    final_metrics: confusion matix 
    classes_name_list: list of class names from the dataset 
    experiment_name: experiment name of the plot

  Returns: 
    - the matrix for confusion matrix heatmap
  
  """

  plt.figure(figsize=(8,8))
  sns.heatmap(
    final_matrix,
    annot=True,
    fmt="d", 
    xticklabels=classes_name_list,
    yticklabels=classes_name_list
    )
  plt.xlabel("Predicted labels")
  plt.ylabel("True labels")
  plt.title(f"Confusion matrix for Test set for {experiment_name} ")
  plt.show()


def each_class_accuracy(final_matrix,classes_name_list:list,experiment_name:str): 

  """
   Per Class accuracy plotting bar 

   Args: 
      
      final_metrics (metrix): confusion matrix
      classes_name_list (list): list of class names 
      experiment_name (str): the experiment name of the plot

  Returns: 
      - the plot for per class accuracy
  """

  if isinstance(final_matrix, torch.Tensor):
    final_matrix = final_matrix.cpu().numpy()

  plt.figure(figsize=(12,8))
  #calculate the per class acurracy based on True positives/total samples in each class 
  per_class_acc=np.diag(final_matrix)/np.sum(final_matrix,axis=1) #adding samples accross columns  

  plt.bar(classes_name_list,per_class_acc,width=0.9,color="darkred")
  plt.xlabel("class names",fontsize=14)
  plt.ylabel("per class accuarcy",fontsize=14)
  plt.title(f"Per Class accuracy plot for {experiment_name}",fontsize=14)
  #adding text just above its accuracy 
  for i,v in enumerate(per_class_acc): 
    plt.text(i,v+0.03,f"{v*100:.2f}") 
  plt.ylim(0,1.1)
  plt.show()



def load_and_plot_metrics_history(
  train_legend_acc_name:str,
  val_legend_acc_name:str,
  train_legend_loss_name:str,
  val_legend_loss_name:str,
  epochs:int,
  experiment_name:str,
  baseline:bool,
  baseline_path:Optional[str]=None,
  main_model_history:Optional[dict]=None, 
  ): 

  if baseline:
    #if plotting for baseline load the metrics history from baseline_path
    print("baseline - True is only for baseline plots")
    history=torch.load(os.path.join(baseline_path,"baseline_metrics_history.pth"),map_location="cpu") 
  else: 
    #if not baseline, get the prepared dict metrics history for main model
    history=main_model_history

  plot_train_and_val_metrics(
    train_collection=history["train_acc"],
    val_collection=history["val_acc"],
    train_legend_name=train_legend_acc_name,
    val_legend_name=val_legend_acc_name,
    y_label="Accuracy",
    plot_title="Accuracy Plot",
    epochs=epochs,
    experiment_name=experiment_name
    )

  print("Accuracy plot has been plotted\n\n")
  print(f"Train Accuracy for {experiment_name}: {max(history["train_acc"]):.2f}")
  print(f"Val Accuracy for {experiment_name}: {max(history["val_acc"]):.2f}")

  plot_train_and_val_metrics(
    train_collection=history["train_loss"],
    val_collection=history["val_loss"],
    train_legend_name=train_legend_loss_name,
    val_legend_name=val_legend_loss_name,
    y_label="Loss",
    plot_title="Loss Plot",
    epochs=epochs,
    experiment_name=experiment_name
    )

  print("Loss plot has been plotted\n\n")
  print(f"Train Loss for {experiment_name}: {min(history["train_loss"]):.2f}")
  print(f"Val Loss for {experiment_name}: {min(history["val_loss"]):.2f}")



def load_csv_main_model_metrics_history(csv_file_path:str,experiment_name:str): 

  csv_dataframes=[]

  search_file_pattern=os.path.join(f"{csv_file_path}/{experiment_name}","metrics*.csv")
  gather_csv_files=glob.glob(search_file_pattern)

  if not gather_csv_files:
    raise FileNotFoundError(f"Any csv files not found at {csv_file_path}/{experiment_name}")

  gather_csv_files.sort() #for ensuring to stack the epochs in order

  print(f"csv_files (debugging): {gather_csv_files}")

  #read csv files 
  if len(gather_csv_files)!=1:

    for file_name in gather_csv_files: 
      #read csv files
      reading_csv_files=pd.read_csv(file_name) 
      #append it to dataframe 
      csv_dataframes.append(reading_csv_files)
      #concatenate csv dfs
    concated_csv=pd.concat(csv_dataframes,ignore_index=True)

    return concated_csv

  else: 
    csv_file=pd.read_csv(gather_csv_files[0])

    return csv_file


def extract_dataframe(df):

    extracted_df={}

    if "epoch" not in df.columns: 
        raise ValueError(f"Epoch Not found to plot") 

    if "train_acc" in df.columns and "train_loss"  in df.columns:

        #drop nan values and get the epoch,train_acc and train_loss 
        train_df=df[["epoch","train_acc","train_loss"]].dropna(subset=["train_acc","train_loss"])
        #append it to the dict
        extracted_df["epoch"]=train_df["epoch"].max()+1 
        extracted_df["train_acc"]=train_df["train_acc"].tolist()
        extracted_df["train_loss"]=train_df["train_loss"].tolist()

    if "val_acc" in df.columns and "val_loss" in df.columns:
        #drop nan values and get the epoch,val_acc and val_loss 
        val_df=df[["epoch","val_acc","val_loss"]].dropna(subset=["val_acc","val_loss"])

        #append it to the dictionary
        extracted_df["val_acc"]=val_df["val_acc"].tolist()
        extracted_df["val_loss"]=val_df["val_loss"].tolist()

    return extracted_df 

















