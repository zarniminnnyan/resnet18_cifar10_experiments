from sklearn.metrics import log_loss
from torch.utils.data import Subset,DataLoader
from tqdm.auto import tqdm
from typing import List,Dict 
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import numpy as np
import math
import torch



@torch.no_grad()
def calibrate_eval_model(models:dict, val_dataset, batch_size:int, device:torch.device, split_size:int, bins_size:int,initial_T:int):
    """
    Do Model Calibration with ECE, Log Loss to know the behavior of the actual accuracy and model's general accuracy 
    Args:
         models (dict): Dict of models {name: model}
         val_dataset: Validation dataset
         batch_size (int): Number of batch size 
         device (cpu or gpu): Device name for operation 
         split_size (int): Split size for Calibration dataset 
         bins_size (int): bins size for ECE measurement
         initial_T (int): initial T for temperature scaling (T>1 if model calibration is over confidence, T<1 for underconfidence, T=1 (original))

    Returns: 
        results (list of dicts): calibration metrics before and after temperature scaling
    """

    def calculate_ece(y_true, y_prob, n_bins=bins_size):
        """Calculate Expected Calibration Error (ECE) for multiclass classification."""
        y_pred = np.argmax(y_prob, axis=1)
        confidence = np.max(y_prob, axis=1)
        correct = (y_pred == y_true).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_conf_, bin_acc_ = [], []

        for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
            mask = (confidence >= bin_lower) & (confidence < bin_upper)
            if np.sum(mask) > 0:
                bin_conf = np.mean(confidence[mask])
                bin_acc = np.mean(correct[mask])
                bin_conf_.append(bin_conf)
                bin_acc_.append(bin_acc)
                ece += np.abs(bin_conf - bin_acc) * np.sum(mask)

        ece /= len(y_true)
        return ece, bin_conf_, bin_acc_

    results = []
    np.random.seed(42)
    calibrate_indices = np.random.choice(len(val_dataset), size=split_size, replace=False)
    calibrate_dataset = Subset(val_dataset, calibrate_indices)

    for model_name, model in models.items():
        model.eval()
        logits_list, labels_list = [], []
        calibrate_dataloader = tqdm(
            DataLoader(calibrate_dataset, batch_size=batch_size, shuffle=False),
            desc=f"Model {model_name} Calibration in process",
            leave=False
        )

        for img, targets in calibrate_dataloader:
            image, label = img.to(device), targets.to(device)
            outputs = model(image)  # logits
            logits_list.append(outputs)
            labels_list.append(label)

        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        nll_criterion = nn.CrossEntropyLoss().to(device)

        # Before temperature
        probs = torch.softmax(logits, dim=1)
        y_probs = probs.detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

        nll_before = nll_criterion(logits, labels).item()
        ece_before, bin_conf_before, bin_acc_before = calculate_ece(y_true, y_probs)
        logloss_before = log_loss(y_true, y_probs)

        # Temperature parameter
        temperature = nn.Parameter(torch.ones(1, device=device)*initial_T)

        # Optimize temperature w.r.t NLL
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=200)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # After temperature
        scaled_logits = logits / temperature
        probs_after = torch.softmax(scaled_logits, dim=1)
        y_probs_after = probs_after.detach().cpu().numpy()

        nll_after = nll_criterion(scaled_logits, labels).item()
        ece_after, bin_conf_after, bin_acc_after = calculate_ece(y_true, y_probs_after)
        logloss_after = log_loss(y_true, y_probs_after)

        results.append({
            "model_name": model_name,
            "before_temperature_log_loss": logloss_before,
            "before_temperature_nll": nll_before,
            "before_temperature_ece": ece_before,
            "before_temperature_bin_conf": bin_conf_before,
            "before_temperature_bin_acc": bin_acc_before,
            "after_temperature_log_loss": logloss_after,
            "after_temperature_nll": nll_after,
            "after_temperature_ece": ece_after,
            "after_temperature_bin_conf": bin_conf_after,
            "after_temperature_bin_acc": bin_acc_after,
            "optimized_temperature": temperature.item()
        })

    return results


def visualize_reliability_diagram(calibrated_result:List[Dict],model_names:List,row_expand:int,col_expand:int): 
    """
    plot reliability diagram

    Args: 
        calibrated result (list): calibrated result for ECE,Log Loss, y_probs,y_true

    Returns: 
        Plot- plot for model calibration measurement by ECE,Log Loss, Reliability Diagram
    """
    cols=math.ceil(math.sqrt(len(model_names)))
    rows=math.ceil(len(model_names)/cols)
   
    fig,axes=plt.subplots(rows,cols,figsize=(rows*row_expand,cols*col_expand))
    
    for idx,ax in enumerate(axes.flatten()[:len(calibrated_result)]):
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.plot(calibrated_result[idx]["before_temperature_bin_conf"],calibrated_result[idx]["before_temperature_bin_acc"], color='orange', marker='o', 
        label=f'Before T: Calibration curve of {calibrated_result[idx]["model_name"]}', linewidth=2, markersize=8
            )

        ax.plot(calibrated_result[idx]["after_temperature_bin_conf"],calibrated_result[idx]["after_temperature_bin_acc"], color='red', marker='o', 
        label=f'After T: Calibration curve of {calibrated_result[idx]["model_name"]}', linewidth=2, markersize=8
            )
        
        after_T_title=f'After T ({calibrated_result[idx]["model_name"]}): Log Loss: {calibrated_result[idx]["after_temperature_log_loss"]:.3f} | ECE: {calibrated_result[idx]["after_temperature_ece"]:.3f} | NLL: {calibrated_result[idx]["after_temperature_nll"]}| T: {calibrated_result[idx]["optimized_temperature"]}'
        title_text= f'Before T ({calibrated_result[idx]["model_name"]}): Log Loss: {calibrated_result[idx]["before_temperature_log_loss"]:.3f} | ECE: {calibrated_result[idx]["before_temperature_ece"]:.3f} | NLL: {calibrated_result[idx]["before_temperature_nll"]}\n\n{after_T_title}'
        ax.set_title(title_text, fontsize=11, pad=10)
        ax.grid(True, alpha=0.7)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.legend(fontsize=10, loc='lower right')

    for j in range(len(calibrated_result), len(axes.flatten())):
     fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


