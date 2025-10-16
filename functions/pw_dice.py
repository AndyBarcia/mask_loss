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

class PairwiseDiceLossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, smooth=1.0, background_index=None):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()

        output = mask_loss.forward_pw_dice_loss(
            logits, 
            targets, 
            smooth,
            background_index if background_index is not None else -1
        )
        ctx.background_index = background_index
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.background_index is not None:
            return None, None, None
        else:
            return None, None

def pairwise_dice_loss_py(logits, targets, smooth=1.0, background_index=None):
    """
    Computes the pairwise Dice loss.

    This function calculates the Dice loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the Dice loss.

    Args:
        logits (torch.Tensor): A tensor of shape (L, B, C, h, w) representing the
                        predicted logits for each class. B is the batch size,
                        C is the number of classes, and h, w are the spatial
                        dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                        labels for the ground truth. H_t and W_t are the
                        spatial dimensions of the targets.
        smooth (float): A small value added to the numerator and denominator
                        for numerical stability.
        background_index (Optional[int]): The index that corresponds to the background
                        to be ignored. If not provided, all classses are
                        computed normally. If specified, the output tensor
                        has the column corresponding to the background removed.

    Returns:
        torch.Tensor: A tensor of shape (L, B, C, max_GT) where max_GT is the maximum value
                      in the targets tensor + 1 if no background index is provided, or
                      th maximum value in the targets tensor otherwise. It contains the 
                      pairwise loss for each class against each possible target. For target 
                      masks with 0 area, a value of infinity is returned.
    """
    L, B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"

    device = logits.device
    dtype = logits.dtype

    # Upsample logits to match the targets spatial size efficiently:
    # reshape to (L*B, C, h, w) -> interpolate -> reshape back to (L, B, C, H_t, W_t)
    logits_reshaped = logits.view(L * B, C, h, w)
    logits_up = F.interpolate(logits_reshaped, size=(H_t, W_t), mode='nearest')
    logits_up = logits_up.view(L, B, C, H_t, W_t)  # (L, B, C, H_t, W_t)
    probs = logits_up.sigmoid()

    targets_long = targets.long().to(device)

    # Determine the full range of GT classes present in the targets
    gt_max = targets_long.max().item()
    GT_all = gt_max + 1

    # Build list of ground-truth classes to evaluate, optionally excluding background
    if background_index is None:
        gt_classes = list(range(GT_all))
    else:
        # If background_index is outside the observed range, it has no effect
        gt_classes = [i for i in range(GT_all) if i != background_index]

    GT = len(gt_classes)

    # Initialize pairwise_loss tensor with infinity
    pairwise_loss = torch.full((L, B, C, GT), torch.inf, device=device, dtype=dtype)

    # Iterate over each ground truth class we will evaluate
    for out_idx, gt_class in enumerate(gt_classes):
        # Create a binary target mask for the current ground truth class
        y = (targets_long == gt_class).unsqueeze(1).to(dtype=dtype) # (B, 1, H_t, W_t)

        # Check which batch elements contain this GT class
        has_class = y.sum(dim=(1, 2, 3)) > 0  # Shape: (B,)

        if not has_class.any():
            # No batch element has this class — skip (pairwise_loss stays +inf)
            continue

        # Broadcast y_mask to match logits_up: (L, B, C, H_t, W_t)
        y = y.unsqueeze(0)  # (1, B, 1, H_t, W_t)

        # Calculate intersection, probability sum, and target sum
        intersection = (probs * y).sum(dim=(3, 4))
        p_sum = probs.sum(dim=(3, 4))
        t_sum = y.sum(dim=(3, 4)).squeeze(1) # Squeeze to match p_sum and intersection shape

        # Calculate Dice score and Dice loss
        dice_score = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
        dice_loss = 1.0 - dice_score

        # Build selection mask where GT exists: expand (B,) -> (L,B,C)
        has_class_expand = has_class.unsqueeze(0).unsqueeze(-1).expand(L, B, C)

        # Update loss only where the class exists in the batch element
        inf_tensor = torch.tensor(torch.inf, device=device, dtype=dtype)
        pairwise_loss[:, :, :, out_idx] = torch.where(
            has_class_expand,
            dice_loss,
            inf_tensor
        )

    return pairwise_loss