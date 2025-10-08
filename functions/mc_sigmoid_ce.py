import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class MultiClassSigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, class_mapping, num_masks=None):
        if num_masks is None:
            B, C = logits.shape[:2]
            num_masks = B*C
        
        logits = logits.contiguous().float()
        targets = targets.contiguous().to(torch.uint8)
        class_mapping = class_mapping.contiguous().long()
        ctx.save_for_backward(logits, targets, class_mapping)
        output = mask_loss.forward_mc_sigmoid_ce_loss(logits, targets, class_mapping, num_masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, class_mapping = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_mc_sigmoid_ce_loss(grad_output, logits, targets, class_mapping)
        return grad_weights, None, None

def multiclass_sigmoid_cross_entropy_loss_py(logits, targets, class_mapping, num_masks=None):
    """
    Naive approach: upsample logits (nearest) to high-res, build per-class one-hot targets,
    and compute BCEWithLogits per pixel then mean.
    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, 255], of type uint8
    class_mapping: (256,) a 1D tensor that maps the encoded values in
        targets to class indices from logits.
    
    Returns:
        torch.Tensor: A scalar tensor representing the mean loss.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"

    # Upsample logits to the spatial resolution of the targets
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')

    device = logits.device
    targets_long = targets.long().to(device)

    # Map the target labels to the correct class indices
    mapped_targets = class_mapping[targets_long]

    # Create one-hot encoded targets from the mapped labels
    onehot = F.one_hot(mapped_targets, num_classes=C).permute(0, 3, 1, 2).to(dtype=logits.dtype)

    # Manually compute the binary cross-entropy with logits loss
    L = logits_up
    y = onehot
    maxL = torch.clamp(L, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L)))
    bce_elem = maxL - L * y + logexp
    
    # Compute the mean loss over all elements
    if num_masks is None:
        loss = bce_elem.mean()
    else:
        loss = bce_elem.sum() / (num_masks * H_t * W_t)

    return loss