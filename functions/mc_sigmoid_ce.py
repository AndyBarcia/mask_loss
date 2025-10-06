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
    def forward(ctx, logits, targets, class_mapping):
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        class_mapping = class_mapping.contiguous()
        ctx.save_for_backward(logits, targets, class_mapping)
        output = mask_loss.forward_mc_sigmoid_ce_loss(logits, targets, class_mapping)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, class_mapping = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_mc_sigmoid_ce_loss(grad_output, logits, targets, class_mapping)
        return grad_weights, None, None

def multiclass_sigmoid_cross_entropy_loss(logits, targets, class_mapping):
    """
    Computes the sigmoid cross-entropy loss for multi-class classification,
    where the ground truth can have multiple positive examples per pixel. The
    maximum possible number of unique ground truths in the image is 256, though 
    there can be many more detections.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, H, W) representing the
                               per-query logits.
        targets (torch.Tensor): A tensor of shape (B, H, W) with ground truth values.
                                The dtype of this tensor determines the number of
                                positive examples per pixel:
                                - int8 or uint8: 1 positive example
                                - int16: 2 positive examples
                                - int32: 4 positive examples
                                - int64: 8 positive examples
        class_mapping (torch.Tensor): A 1D tensor of shape (256,) that maps the
                                      encoded values in `targets` to class indices.

    Returns:
        torch.Tensor: A scalar tensor representing the mean loss.
    
    NOTE: targets is not implemented for uint16, uint32 and uint64 because they are
    not yet supported on pytorch.
    """
    B, C, H, W = logits.shape
    dtype_to_n_positives = {
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }
    
    n_positives = dtype_to_n_positives[targets.dtype]
    
    # Create the multi-hot encoded target tensor
    target_multi_hot = torch.zeros_like(logits)
    
    # Unpack the ground truth values
    for i in range(n_positives):
        # Extract the i-th positive example for each pixel
        shift = i * 8
        mask = 0xFF << shift
        indices = (targets.long() & mask) >> shift
        
        # Map the extracted indices to class indices
        mapped_indices = class_mapping[indices.long()]
        
        # Create a one-hot tensor for the current positive example
        target_one_hot = F.one_hot(mapped_indices.long(), num_classes=C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)
        
        # Ensure that the spatial dimensions of targets match logits
        if target_one_hot.shape[2:] != logits.shape[2:]:
            target_one_hot = F.interpolate(
                target_one_hot.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        # Accumulate the one-hot tensors to create the multi-hot target
        target_multi_hot += target_one_hot

    # Compute the binary cross-entropy loss with logits
    loss = F.binary_cross_entropy_with_logits(
        logits,
        target_multi_hot,
        reduction='mean'
    )
    
    return loss