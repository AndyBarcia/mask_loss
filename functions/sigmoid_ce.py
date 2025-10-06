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

class SigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        ctx.save_for_backward(logits, targets)
        output = mask_loss.forward_sigmoid_ce_loss(logits, targets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_sigmoid_ce_loss(grad_output, logits, targets)
        return grad_weights, None

def sigmoid_cross_entropy_loss(logits, targets):
    """
    Computes sigmoid cross entropy loss for multi-class classification.
    
    Args:
        logits: Tensor of shape (B, C, H, W) - per-query logits
        targets: Tensor of shape (B, H_t, W_t) - ground truth with values [0, C-1]
                 Can have different spatial dimensions than logits
    
    Returns:
        loss: Scalar tensor representing the mean loss
    """
    B, C, H, W = logits.shape
    B_t, H_t, W_t = targets.shape
    
    # Convert targets to one-hot encoding: (B, H_t, W_t) -> (B, C, H_t, W_t)
    targets_one_hot = F.one_hot(targets, num_classes=C)  # (B, H_t, W_t, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H_t, W_t)
    
    # Interpolate targets to match logits spatial dimensions if needed
    if (H_t, W_t) != (H, W):
        targets_one_hot = F.interpolate(
            targets_one_hot,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )  # (B, C, H, W)
    
    # Compute sigmoid cross entropy loss
    # BCE loss formula: -[y*log(σ(x)) + (1-y)*log(1-σ(x))]
    # Using log-sum-exp trick for numerical stability:
    # BCE = max(x,0) - x*y + log(1 + exp(-|x|))
    loss = F.binary_cross_entropy_with_logits(
        logits, 
        targets_one_hot, 
        reduction='mean'
    )
    
    return loss