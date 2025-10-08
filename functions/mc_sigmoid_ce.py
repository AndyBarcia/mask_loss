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
        B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        assert class_mapping.shape[-1] == 256, "Class mapping must have shape 256"

        ctx.received_num_masks = num_masks is not None
        if num_masks is None:
            B, C = logits.shape[:2]
            num_masks = B*C
        ctx.num_masks = num_masks
        
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
        grad_weights = mask_loss.backward_mc_sigmoid_ce_loss(
            grad_output, 
            logits, 
            targets, 
            class_mapping, 
            ctx.num_masks
        )
        if ctx.received_num_masks:
            return grad_weights, None, None, None
        else:
            return grad_weights, None, None

def multiclass_sigmoid_cross_entropy_loss_py(logits, targets, class_mapping, num_masks=None):
    """
    Upsamples logits, builds per-class targets, and computes BCEWithLogits per pixel.
    
    This version correctly handles background pixels. For pixels where the class_mapping
    is out of the valid range [0, C-1], the target for all classes is set to 0. This
    encourages the logits for all classes to be negative at these background locations.

    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, 255], of type uint8
    class_mapping: (B,256,) a tensor that maps the encoded values in
        targets to class indices from logits. If the encoded value is mapped
        to a value <0 or >=C, then it is treated as background.
    
    Returns:
        torch.Tensor: A scalar tensor representing the mean loss.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"
    assert class_mapping.shape[-1] == 256, "Class mapping must have shape 256"

    # Upsample logits to the spatial resolution of the targets
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')

    device = logits.device
    targets_long = targets.long().to(device)

    # Map the target labels to the correct class indices
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H_t, W_t)
    mapped_targets = class_mapping[batch_idx, targets_long]

    # Create a target tensor 'y' for the loss function.
    # Initialize with all zeros. This correctly sets the target for background
    # pixels to be a vector of all zeros.
    onehot = torch.zeros_like(logits_up)

    # Create a mask for valid foreground pixels
    valid_mask = (mapped_targets >= 0) & (mapped_targets < C)

    # For the foreground pixels, we need to place a '1' at the correct class channel.
    # We use scatter_ for an efficient update.
    # First, clamp the mapped_targets to avoid out-of-bounds errors for the index.
    # The invalid values won't be used anyway because of how we construct the 'src' tensor.
    index = mapped_targets.clamp(0, C - 1).unsqueeze(1)
    
    # The source tensor for scatter should be 1.0 only where the mask is valid.
    src = valid_mask.unsqueeze(1).to(onehot.dtype)
    
    # Place 1.0 in the channel corresponding to the class index for valid pixels.
    onehot.scatter_(1, index, src)

    # Compute the binary cross-entropy
    bce_elem = F.binary_cross_entropy_with_logits(logits_up, onehot, reduction='sum')
    
    # Compute mean
    if num_masks is None:
        loss = bce_elem / (B * C * H_t * W_t)
    else:
        loss = bce_elem / (num_masks * H_t * W_t)

    return loss