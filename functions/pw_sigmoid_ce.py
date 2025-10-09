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

class PairwiseSigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.forward_pw_sigmoid_ce_loss(logits, targets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None

def pairwise_sigmoid_cross_entropy_loss_py(logits, targets):
    """
    Computes pairwise sigmoid cross-entropy loss and returns a tensor list.

    This function calculates the loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the binary cross-entropy with logits.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, h, w) representing the
                             predicted logits for each class. B is the batch size,
                             C is the number of classes, and h, w are the spatial
                             dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                                labels for the ground truth. H_t and W_t are the
                                spatial dimensions of the targets.

    Returns:
        torch.Tensor: A list of tensors where each element corresponds to a batch
                      item and contains a tensor of shape (C, num_present_gt_classes)
                      with the pairwise losses.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"

    # Upsample logits to match the spatial dimensions of the targets
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')

    device = logits.device
    targets_long = targets.long().to(device)
    
    # Determine the number of ground truth classes
    gt_max = targets_long.max().item()
    GT = gt_max + 1
    
    # A list to hold tensors for each item in the batch.
    batch_losses = [[] for _ in range(B)]

    # Iterate over each possible ground truth class
    for gt_class in range(GT):
        # Create a binary target mask for the current ground truth class
        y = (targets_long == gt_class).unsqueeze(1).to(dtype=logits.dtype)
        
        # Check which batch elements contain this GT class
        has_class = y.sum(dim=(2, 3)) > 0  # Shape: (B, 1)

        if not has_class.any():
            continue
        
        # Stable BCE-with-logits calculation
        maxL = torch.clamp(logits_up, min=0.0)
        logexp = torch.log1p(torch.exp(-torch.abs(logits_up)))
        bce_elem = maxL - logits_up * y + logexp
        
        # Sum the loss for each predicted class and normalize
        loss_per_class = bce_elem.sum(dim=(2, 3)) / (H_t * W_t)
        
        # Append the loss for batch items that have the current gt_class
        for b in range(B):
            if has_class[b]:
                batch_losses[b].append(loss_per_class[b])

    # Stack the losses for each batch item to get tensors of shape (C, num_gt)
    final_tensors = [
        torch.stack(tensors, dim=1) if tensors else torch.empty((C, 0), device=device) 
        for tensors in batch_losses
    ]

    return final_tensors


def pairwise_sigmoid_cross_entropy_loss_efficient_py(logits, targets):
    """
    Efficiently computes pairwise sigmoid cross-entropy loss using a count-based method
    and returns a tensor list.

    This function avoids high-resolution one-hot target tensors by using unfolding
    to count the occurrences of each ground truth class within the regions
    corresponding to each logit.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, h, w) with the predicted logits.
                             B is the batch size, C is the number of classes, and h, w
                             are the spatial dimensions.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer ground
                                truth labels. H_t and W_t must be integer multiples
                                of h and w.

    Returns:
        torch.Tensor: List of tensors where each element corresponds to a batch
                      item and contains a tensor of shape (C, num_present_gt_classes)
                      with the pairwise losses.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"
    assert H_t % h == 0 and W_t % w == 0, \
        "Target dimensions must be integer multiples of logit dimensions"
    
    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation requires equal scaling factors for height and width.")
    s = sH
    N2 = s * s

    device = logits.device
    targets_long = targets.long().to(device)

    gt_max = targets_long.max().item()
    GT = gt_max + 1

    # A list to hold tensors for each item in the batch.
    batch_losses = [[] for _ in range(B)]
    
    L_reshaped = logits.reshape(B, C, h * w)
    
    # Precompute parts of the stable BCE-with-logits formula
    maxL = torch.clamp(L_reshaped, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L_reshaped)))

    for gt_class in range(GT):
        # Create a binary mask for the current ground truth class
        onehot_gt = (targets_long == gt_class).unsqueeze(1).to(dtype=logits.dtype)
        
        # Check which batch elements contain this GT class
        has_class = onehot_gt.sum(dim=(2, 3)) > 0  # Shape: (B, 1)
        
        if not has_class.any():
            continue

        # Reshape for unfolding to count ground truth classes in each block
        Bc = onehot_gt.reshape(B, 1, H_t, W_t)
        unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))
        unf = unf.reshape(B, 1, s * s, h * w)
        n_k = unf.sum(dim=2)  # Shape: (B, 1, h*w)

        # Stable BCE-with-logits summed across the block for each predicted class
        loss_block = N2 * maxL - L_reshaped * n_k + N2 * logexp
        
        # Sum the loss over the spatial dimensions and normalize
        loss_sum = loss_block.sum(dim=2) / (H_t * W_t)
        
        # Append the loss for batch items that have the current gt_class
        for b in range(B):
            if has_class[b]:
                batch_losses[b].append(loss_sum[b])

    # Stack the losses for each batch item to get tensors of shape (C, num_gt)
    final_tensors = [
        torch.stack(tensors, dim=1) if tensors else torch.empty((C, 0), device=device) 
        for tensors in batch_losses
    ]
        
    return final_tensors