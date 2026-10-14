import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def global_structured_prune(
    model: nn.Module,
    layers: list,
    device:torch.device,
    make_pruning_permanent:bool,
    amount=0.3, 
    norm=2
    ):
    
    """
    Prunes Conv2d filters globally across selected layers.

    Calculates the norm of each filter, finds a global threshold so that 
    about `amount` (e.g. 0.3 = 30%) of the weakest filters are removed, 
    and applies a mask to zero out entire output channels.

    Parameters:
        model (nn.Module): The PyTorch model.
        layers (list): Names of Conv2d layers to prune.
        device (torch.device): Target device (CPU/GPU).
        make_pruning_permanent(bool) : True/False for pruning permanently
        amount (float): Fraction of filters to prune.
        norm (int): Norm type (e.g. 2 for L2).

    Returns:
        None. The model is modified in place.
    """
  
    all_norms = []
    layer_norms = {}
    
    # Compute norms correctly for each filter (dim 0)
    for layer_name in layers:
        layer = dict(model.named_modules())[layer_name]
        
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.detach().cpu()
            # Flatten everything except out_channels (dim 0) to get per-filter norms
            flat_weight = weight.flatten(start_dim=1) 
            norms = torch.norm(flat_weight, p=norm, dim=1)
            
            all_norms.append(norms)
            layer_norms[layer_name] = norms

    if not all_norms:
        return 

    # Compute the global threshold
    all_norms_flat = torch.cat(all_norms)
    k = int(amount * all_norms_flat.numel())
    if k == 0:
        return #Nothing to prune
        
    threshold = torch.kthvalue(all_norms_flat, k).values.item()

    #Create and apply masks using custom_from_mask
    for layer_name in layers:
        layer = dict(model.named_modules())[layer_name]
        norms = layer_norms[layer_name]
        
        #Create a 1D mask for the channels (1 = keep, 0 = prune)
        channel_mask = (norms >= threshold).float()
        
        # Expand 1D mask to match Conv2d weight shape: (out_channels, 1, 1, 1)
        # PyTorch broadcasting handles the rest
        weight_mask =channel_mask[:, None, None, None].expand_as(layer.weight).to(device)
        
        #Apply the explicit mask structure safely
        prune.custom_from_mask(layer, name="weight", mask=weight_mask)
        #make pruning permanent
        if make_pruning_permanent: 
            print("Pruned permanent!")
            prune.remove(layer,"weight")

    print("Pruning succeeded!!")
    return model
