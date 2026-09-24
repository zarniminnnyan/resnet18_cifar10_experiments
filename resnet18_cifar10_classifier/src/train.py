from config.exp_config import (
    dataset_dir,
    cifar10_classes,
    resume_dir,
    baseline_model_dir,
    main_model_dir
    )
from utils.helper_utils import (
    load_resnet18,
    load_prepare_cifar10,
    build_dataloader,
    build_transforms
    )

from src.main_model import ( 
    resnet18_main_model_setup,
    choose_learning_rates_and_layers,
    lightning_trainer,
    Resnet18Lightning
    )
from src.baseline_model import train_baseline_classifier
import argparse 
import torch


def configurable_arguments():

    """ configure options for model experiments for easier to modify"""
    
    parser=argparse.ArgumentParser(description="Model Experiments with configurable options")

    parser.add_argument("--epochs",type=int,default=10,help="number of epochs for model experiment")
    parser.add_argument("--experiments",type=str,default="exp1",choices=["exp1","exp2","exp3","exp4","exp5"],help="ML experiments")
    
    # exp1 - baseline experiment 
    # exp2 - experiment with unfreeze layer4 + fc 
    # exp3 - experiment with exp2 + applying standard augmentation 
    # exp4 - experiment with exp3 + applying CosineAnealing  for decreasing learning rates following Cosine Curve
    # exp5 - experiment with exp4 + applying Cutmix/Mixup augmentation method 

    parser.add_argument("--num_worker",type=int,default=4,help="number of workers for model experiment")
    parser.add_argument("--image_size",type=int,default=224,help="Image input size to feed model")
    parser.add_argument("--baseline", action="store_true", help="Run baseline model")
    parser.add_argument("--baseline_lr",type=float,default=1e-3, help="baseline model learning rate configuration")
    parser.add_argument("--batch_size",type=int,default=128,help=" batch size for model experiment")
    parser.add_argument("--main_model_lr",type=float,nargs="+",default=[1e-4,1e-3], help="list of learning rates for main model")
    parser.add_argument("--main_model_layers",type=str,nargs="+",default=["layer4","fc"], help="list of layers to unfreeze for main model")


    args=parser.parse_args()

    print(
          f"The configured arguments are:\nepochs: {args.epochs}\n"
          f"baseline lr: {args.baseline_lr}\nbatch size: {args.batch_size}\n"
          f"main model lr: {args.main_model_lr}\n Image Input size: {args.image_size}\n"
          f"num worker: {args.num_worker}\n"
          f"main model unfreezed layers: {args.main_model_layers}\n"
          f"ML experiments number : {args.experiments}\n\n"
          )

    return args  

def main(): 

    """Main function for running baseline model and main model experiments"""

    #configure arguments for experiments
    args=configurable_arguments()
    #set True if baseline 
    baseline=args.baseline

    #device 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #load resnet18 pretrained model 
    model=load_resnet18(device=device)

    standard_aug_flag=True #standard augmentation 
    apply_cutmix_mixup=True #cutmix/mixup using logic
    add_lr_scheduler=True #adding learning rate scheduler logic


    if args.experiments in ["exp1","exp2"]:
        standard_aug_flag=False

    if args.experiments in ["exp1","exp2","exp3"]:
        add_lr_scheduler=False
    
    if args.experiments in ["exp1","exp2","exp3","exp4"]:
        apply_cutmix_mixup=False

    #download cifar10 dataset if it doesn't exist or it is corrupted
    train_dataset,val_dataset,test_dataset=load_prepare_cifar10(
        dataset_path=dataset_dir,
        build_transforms=build_transforms,
        image_size=args.image_size,
        standard_aug=standard_aug_flag #set False not to be applied standard augmentation or True
        )
    
    #call dataloaders function
    train_dataloader,val_dataloader,_= build_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        use_cutmix_mixup=apply_cutmix_mixup,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )   

    print(f"length of train dataset after splitting {len(train_dataset)}")
    print(f"length of val dataset after splitting {len(val_dataset)}")
    print(f"length of test dataset  {len(test_dataset)}")

    if baseline:
        #train baseline model 
        train_baseline_classifier(
            baseline_model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            baseline_model_dir=baseline_model_dir,
            num_classes=cifar10_classes,
            num_epochs=args.epochs,
            learning_rate=args.baseline_lr,
            resume_dir=resume_dir,
            resume=True,
            device=device
            )

    else: 
        #setup main model 
        main_model=resnet18_main_model_setup(
            device=device,
            trainable_layers_group=args.main_model_layers
            )

        #setup params for optimizers 
        optimizer_params=choose_learning_rates_and_layers(
            model=main_model,
            learning_rates=args.main_model_lr,
            layers=args.main_model_layers
            )

        #train main model (Pytorch lightning)
        Lightning_class=Resnet18Lightning(
            model=main_model,T_max=args.epochs,
            num_classes=cifar10_classes,
            add_scheduler=add_lr_scheduler,
            trainable_params_group=optimizer_params
            )

        #train Lightning 
        lightning_trainer(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            resnet18_lightning=Lightning_class,num_epochs=args.epochs,
            save_model_path=main_model_dir,
            resume=True,
            logger_filename=f"main_model_{args.experiments}_metrics",
            last_ckpt_file=f"{args.experiments}_last_saved.ckpt"
        )

if __name__=="__main__":
    main()    






