import torch 
import torch.nn.functional as F
import torchmetrics.classification as metrics
from tqdm.auto import tqdm
from pathlib import Path
import os 

def evaluate_baseline_resnet18(
    baseline_model,
    val_dataloader,
    num_classes:int,
    device:torch.device
    ):

  """
  Evaluate the baseline resnet18 model on val dataset
  Apply torchmetrics for measuring the metrics

  Args: 

   model (nn.Module): resnet18 model with unfreezed fc layer with 10 heads 
   val_dataloader (Callable): Dataloader for validation dataset 
   num_classes: Cifar10 classes (10 classes)
  
  Returns:

   tuple - (total_val_accuracy,total_val_loss,model)
   
   total_val_accuracy : validation accuracy of pretrained baseline model
   total_val_loss: validation loss of pretrained baseline model 
   baseline_model (nn.Module): pretrained baseline model (out of the box ) after validating with train dataset and val dataset 
   
  """
  val_acc=metrics.Accuracy(task="multiclass",num_classes=num_classes).to(device)
  val_acc.reset()


  #total loss
  total_loss=0.0

  #pass dataloader into tqdm
  validation_dataloader=tqdm(val_dataloader,desc="Validation baseline is in progress")

 #send model to the device and set the eval mode
  baseline_model.to(device)
  baseline_model.eval()

  #turn off the updating gradients
  with torch.no_grad():

    for images,labels in validation_dataloader:
      img,label=images.to(device),labels.to(device)
      #predict
      outputs=baseline_model(img)
      #loss
      loss=F.cross_entropy(outputs,label)
      #accumulate the loss
      total_loss +=loss.item()
      #update the val acc in val accuracy metrics
      val_acc.update(outputs,label)

  #compute accuracy
  total_val_accuracy=val_acc.compute().item()*100
  #compute total loss
  total_val_loss=total_loss/len(val_dataloader)

  print(f"total val accuracy {total_val_accuracy:.2f}\ntotal val loss {total_val_loss:.2f}")

  return total_val_accuracy,total_val_loss,baseline_model


def train_baseline_classifier(
    baseline_model,
    train_dataloader,
    val_dataloader,
    baseline_model_dir:str,
    num_classes:int,
    num_epochs: int,
    learning_rate: float,
    resume_dir: str,
    resume: bool,
    device: torch.device):
    """
    Train the baseline ResNet18 classifier and evaluate it on the validation set.

    Args:

        baseline_model: ResNet18 model to train.
        train_dataloader: Training data loader.
        val_dataloader: Validation data loader.
        baseline_model_dir: Directory to save model weights.
        num_classes: Number of classes.
        num_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        resume_dir: Directory containing checkpoints.
        resume: Whether to resume from the latest checkpoint.
        device: Device used for training.

    Returns:
        Trained model and lists of training/validation accuracy and loss.
    """
        
    optimizer = torch.optim.SGD(
        (p for p in baseline_model.parameters() if p.requires_grad),
        lr=learning_rate, momentum=0.9
    )

    #training accuracy 
    train_acc=metrics.Accuracy(task="multiclass",num_classes=num_classes).to(device)

    # Move model to device BEFORE defining the scheduler or loading state dicts
    baseline_model.to(device)

    os.makedirs(resume_dir, exist_ok=True)
    os.makedirs(baseline_model_dir,exist_ok=True)

    start_epoch = 0

    def get_latest_checkpoint(resume_dir):
        checkpoints_path = Path(resume_dir)
        get_all_ckpts = list(checkpoints_path.glob("**/checkpoints*.pth"))
        if not get_all_ckpts:
            return None
        # Use os.path.getmtime or path validation safely
        return max(get_all_ckpts, key=os.path.getmtime)

    if resume:
        try:
            latest_checkpoint_file = get_latest_checkpoint(resume_dir)
            if latest_checkpoint_file is None:
                print("No checkpoints found. Starting from scratch.")
            else:
                print(f"Resuming from checkpoint: {latest_checkpoint_file}")
                checkpoints = torch.load(latest_checkpoint_file, map_location=device)
                baseline_model.load_state_dict(checkpoints["model_state"])
                optimizer.load_state_dict(checkpoints["optimizer_state"])
                # lr_scheduler_technique.load_state_dict(checkpoints["lr_scheduler_state"])
                start_epoch = checkpoints["epoch"] + 1
        except Exception as e:
            raise ValueError(f"Failed loading saved model due to {e}")

    train_loss_collection=[]
    train_accuracy_collection=[]
    val_accuracy_collection=[]
    val_loss_collection=[]

    #For accumulating loss and accuracy
    final_train_loss =0.0
    #best accuracy
    best_accuracy = 0.0

    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training in progress"):
        baseline_model.train()
        train_acc.reset()
        total_loss = 0.0

        trainloader = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, labels in trainloader:
            img, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = baseline_model(img)

            # If labels are one-hot/soft (2D), target the index with the highest weight
            if labels.ndim > 1:
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels
            
            # Loss evaluation (PyTorch CrossEntropy accepts soft probabilities natively if 2D)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_acc.update(outputs,targets)

            trainloader.set_postfix(
                current_loss=f"{loss.item():.4f}",
                batch_acc=f"{100.0 *train_acc.compute().item():.2f}%"
            )

        
        final_train_loss = total_loss / len(train_dataloader)
        final_train_accuracy = train_acc.compute().item()*100

        #add train accuracy and loss into the list
        train_accuracy_collection.append(final_train_accuracy)
        train_loss_collection.append(final_train_loss)

        print(f"Epoch {epoch+1}: Loss={final_train_loss:.4f}, Accuracy={final_train_accuracy:.2f}%")

        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save({
                "epoch": epoch,
                "model_state": baseline_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": final_train_loss
            }, os.path.join(resume_dir, f"checkpoints{epoch+1}.pth"))


        #call evaluation for measuring overfitting or underfitting
        total_val_accuracy,total_val_loss,baseline_eval_model= evaluate_baseline_resnet18(
          baseline_model=baseline_model,
          val_dataloader=val_dataloader,
          num_classes=num_classes,
          device=device
          )


        if total_val_accuracy>best_accuracy:
            #find  the best accuracy and save it
            best_accuracy=total_val_accuracy
           #save the model into defined directory
            torch.save(
                baseline_eval_model.state_dict(),
               os.path.join(baseline_model_dir,"best_baseline_eval_model.pth")
      )
       #append val accuracies,losses in the list for measuring overfitting or uderfitting
        val_accuracy_collection.append(total_val_accuracy)
        val_loss_collection.append(total_val_loss)

        #saving final model
        torch.save(
           baseline_eval_model.state_dict(),
           os.path.join(baseline_model_dir,"baseline_eval_model_final.pth")
       )

        #save training history 
        history = {
            "train_accuracy": train_accuracy_collection,
            "train_loss": train_loss_collection,
            "val_accuracy": val_accuracy_collection,
            "val_loss": val_loss_collection
             }
        #saving the history with torch.save 
        torch.save(
            history,
            os.path.join(baseline_model_dir,"baseline_metrics_history.pth")
            )

        #Load the best saved model
        baseline_eval_model.load_state_dict(
            torch.load(
                 os.path.join(baseline_model_dir,"best_baseline_eval_model.pth"),
                 map_location=device
            )
        )

    print(f"\nFinal Training Results:\n Loss={final_train_loss:.4f}\n Accuracy={final_train_accuracy:.2f}%")
    print(f"\nFinal Val Results:\n Loss={total_val_loss:.4f}\n Accuracy={total_val_accuracy:.2f}%")


    return baseline_eval_model,train_accuracy_collection, train_loss_collection,val_accuracy_collection,val_loss_collection
