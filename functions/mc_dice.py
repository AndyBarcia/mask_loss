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

class MultiClassDiceLossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, class_mapping, smooth=1.0, num_masks=None):
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
        ctx.smooth = smooth
        output, int_sum, p_sum, t_sum = mask_loss.forward_mc_dice_loss(
            logits, 
            targets, 
            class_mapping, 
            smooth,
            num_masks
        )
        ctx.save_for_backward(logits, targets, class_mapping, int_sum, p_sum, t_sum)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, class_mapping, int_sum, p_sum, t_sum = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_mc_dice_loss(
            grad_output, 
            logits, 
            targets,
            class_mapping, 
            int_sum,
            p_sum,
            t_sum,
            ctx.smooth,
            ctx.num_masks
        )
        if ctx.received_num_masks:
            return grad_weights, None, None, None
        else:
            return grad_weights, None, None

def multiclass_dice_loss_py(logits, targets, class_mapping, smooth=1e-6, num_masks=None):
    """
    Naive approach: upsample logits (nearest) to high-res, build per-class one-hot targets,
    compute sigmoid probabilities, compute Dice per (B,C) across the whole high-res map,
    and return mean(1 - Dice) over B and C.

    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, 255], of type uint8
    class_mapping: (B,256,) a tensor that maps the encoded values in
        targets to class indices from logits. If the encoded value is mapped
        to a value <0 or >=C, then it is treated as background.
    smooth: small float to avoid division by zero
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert class_mapping.shape[-1] == 256, "Class mapping must have shape 256"

    # Upsample logits to high-res (nearest)
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')  # (B, C, H_t, W_t)

    device = logits.device
    targets_long = targets.long().to(device)

    # Map the target labels to the correct class indices
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H_t, W_t)
    mapped_targets = class_mapping[batch_idx, targets_long]

    # one-hot targets: (B, C, H_t, W_t)
    onehot = torch.zeros((B,C,H_t,W_t), device=logits.device, dtype=logits.dtype)
    
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

    # predicted probabilities per-pixel per-class
    probs = torch.sigmoid(logits_up)  # (B, C, H_t, W_t)

    # sums over high-res spatial dims
    intersection = (probs * onehot).sum(dim=(2, 3))        # (B, C)
    p_sum = probs.sum(dim=(2, 3))                         # (B, C)
    t_sum = onehot.sum(dim=(2, 3))                        # (B, C)

    dice = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)  # (B, C)
    loss = 1.0 - dice
    loss = loss.mean() if num_masks is None else loss.sum()/num_masks

    return loss

def multiclass_dice_loss_efficient_py(logits, targets, class_mapping, smooth=1e-6, num_masks=None):
    """
    Efficient count-based Dice loss consistent with nearest upsampling behavior.
    Assumes H_t and W_t are integer multiples of h and w respectively and that
    nearest upsampling repeats the same logit value across each s x s block.

    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, C-1]
    class_mapping: (B,256,) a tensor that maps the encoded values in
        targets to class indices from logits. If the encoded value is mapped
        to a value <0 or >=C, then it is treated as background.
    smooth: small float to avoid division by zero

    Returns: scalar tensor (mean Dice loss over B and C)
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"
    assert class_mapping.shape[-1] == 256, "Class mapping must have shape 256"

    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation assumes equal scale factor for height and width (square blocks)")
    s = sH
    N2 = s * s  # number of high-res pixels per low-res pixel

    device = logits.device
    targets_long = targets.long().to(device)

    # Map the target labels to the correct class indices
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H_t, W_t)
    mapped_targets = class_mapping[batch_idx, targets_long]

    # one-hot targets: (B, C, H_t, W_t)
    onehot = torch.zeros((B,C,H_t,W_t), device=logits.device, dtype=logits.dtype)
    
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

    # Unfold the one-hot high-res maps into sxs blocks to count positives per block.
    # Reshape to (B*C, 1, H_t, W_t) to use F.unfold conveniently.
    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    # After unfold: shape (B*C, s*s, h*w)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))
    # reshape -> (B, C, s*s, h*w)
    unf = unf.reshape(B, C, s * s, h * w)
    # counts of target-positive pixels per block: (B, C, h*w)
    n_k = unf.sum(dim=2)

    # logits at low-res positions: (B, C, h, w) -> (B, C, h*w)
    L = logits.reshape(B, C, h * w)

    # predicted probability per low-res position (same for all pixels in block)
    p = torch.sigmoid(L)  # (B, C, h*w)

    # Sum of predicted probabilities over the whole high-res map for each (B,C):
    # p_sum_total = sum_over_blocks (N2 * p) = N2 * p.sum(dim=2)
    p_sum_total = N2 * p.sum(dim=2)  # (B, C)

    # Sum of target positives over whole high-res map for each (B,C):
    # t_sum_total = n_k.sum(dim=2)
    t_sum_total = n_k.sum(dim=2)  # (B, C)

    # Intersection total = sum_over_blocks (p * n_k) because every pixel in block has same p
    # -> intersection_total = (p * n_k).sum(dim=2)
    intersection_total = (p * n_k).sum(dim=2)  # (B, C)

    dice = (2.0 * intersection_total + smooth) / (p_sum_total + t_sum_total + smooth)  # (B, C)
    loss = 1.0 - dice
    loss = loss.mean() if num_masks is None else loss.sum()/num_masks

    return loss