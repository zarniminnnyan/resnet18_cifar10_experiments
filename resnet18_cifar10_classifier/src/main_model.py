from utils.helper_utils import load_resnet18
from config.exp_config import logs_dir
from typing import List,Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics.classification as metrics
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import torch.nn as nn
import torch
import os 

def resnet18_main_model_setup(device:torch.device,trainable_layers_group:list):
    
    """
    Load ResNet-18 and make only layer4 + fc trainable.
    All other layers stay frozen to keep pretrained features fixed.

    Args:
       device (torch.device): device (cpu or gpu)
       trainable_layers_group (List): List of chosen layers to unfreeze

    Returns: 
       model (nn.Module): the pretrained model with unfreezed layer4 and fc layers

    """

    model =load_resnet18(device=device)

    for name, param in model.named_parameters():
        if any(layer in name for layer in trainable_layers_group):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


def choose_learning_rates_and_layers(model: nn.Module, learning_rates: list, layers: list):
    """
    Selects parameters from given layers and assigns respective learning rates.
    Args:
        model: PyTorch model (nn.Module) - pretrained resnet18 with unfreezed layer4 and fc
        learning_rates: list of learning rates (floats) for each layer respectively
        layers: list of unfreezed layers for getting params for optimizer

    Returns:
        trainable_params_group: list of dicts for optimizer parameter groups
    """

    # Safety check: make sure lists match in length
    if len(learning_rates) != len(layers):
        raise ValueError("The length of learning rates and layers must match")

    trainable_params_group = []
    # Iterate over each (learning rate, layer) pair
    for lr, layer in zip(learning_rates, layers):
        # Collect parameters whose names contain the layer substring and are trainable
        params = [p for n, p in model.named_parameters() if layer in n and p.requires_grad]

        # If no parameters found, raise error for clarity
        if not params:
            raise ValueError(f"Params for {layer} not found")

        # Append parameter group with its learning rate
        trainable_params_group.append({"params": params, "lr": lr})

    # Return list of parameter groups ready for optimizer
    return trainable_params_group

class Resnet18Lightning(pl.LightningModule):
    
    """Training the resnet18 model using pytorch lightning"""

    def __init__(
         self, model: torch.nn.Module,
         T_max: int, num_classes: int,
         add_scheduler:bool, 
         trainable_params_group:List[Dict]
         ):

        super().__init__()
        self.save_hyperparameters(ignore=['model']) #avoid saving the entire model

        self.model = model

        #define torchmetrics for measuring model
        self.train_acc = metrics.Accuracy(task="multiclass", num_classes=num_classes)*100
        self.val_acc = metrics.Accuracy(task="multiclass", num_classes=num_classes)*100


        # get the parameters if require_grad is True and set the lr for layer4 and fc specifically
        self.trainable_params_and_lr =trainable_params_group 

    def forward(self, x):
        """passing data through the model"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """passing batch through the training step"""
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        #handles soft cutmix/mixup soft probability vectors
        if labels.ndim>1:
            targets=torch.argmax(labels,dim=-1)
        else:
            targets=labels

        self.train_acc(outputs,targets)

        self.log("train_acc",self.train_acc, on_step=False,on_epoch=True,prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):
        """ validate the finetuned model """
        inputs, labels = batch
        outputs = self(inputs)
        self.val_acc(outputs, labels)

        self.log("val_loss", F.cross_entropy(outputs, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc",self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.trainable_params_and_lr, momentum=0.9,weight_decay=5e-4)

        #add scheduler if True
        if self.hparams.add_scheduler:
            
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.T_max)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return {"optimizer":optimizer}

def lightning_callbacks(model_checkpoint_path:str):

    """define the callbacks for saving model checkpoints and earlystopping if the training isn't improving"""
    callbacks = [
        EarlyStopping(
            monitor="val_acc",
            patience=10,
            mode="max",
        ),
        ModelCheckpoint(
            monitor="val_acc",
            dirpath=model_checkpoint_path,
            mode="max",
            save_top_k=1,
            save_last=True
        )
    ]
    return callbacks

def lightning_trainer(
    train_dataloader, 
    val_dataloader,
     resnet18_lightning, 
     num_epochs: int, 
     save_model_path: str, 
     resume: bool,
     logger_filename:str,
     last_ckpt_file:str
     ):
     
    """Call Lightning trainer to train ResNet18 finetuned model with correct resume handling."""

    print("Executing finetuning resnet18 code .... ")

    # TQDM bar
    bar = TQDMProgressBar(refresh_rate=200)

    #Set Logger for saving logs
    csv_Logger=CSVLogger(save_dir=logs_dir,name=logger_filename)

    # Make the directory if it doesn't exist
    os.makedirs(save_model_path, exist_ok=True)


    # Callbacks for midway saving and early stopping
    callbacks = lightning_callbacks(save_model_path)

    # Apply trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        precision="32-true",
        logger=csv_Logger,
        callbacks=[bar] + callbacks
    )

    # Resume Logic
    ckpt_path = None
    if resume:
        # Look for the specific filename set in lightning_callbacks function
        expected_ckpt_path = os.path.join(save_model_path,last_ckpt_file)

        if os.path.exists(expected_ckpt_path):
            print(f"Found existing checkpoint! Resuming training from: {expected_ckpt_path}")
            ckpt_path = expected_ckpt_path
        else:
            print(f"Resume is True, but no checkpoint found at {expected_ckpt_path}. Starting from scratch.")

    # Fit the model
    trainer.fit(
        resnet18_lightning,
        train_dataloader,
        val_dataloader,
        ckpt_path=ckpt_path
    )

    # Return trained model architecture
    return resnet18_lightning.model






