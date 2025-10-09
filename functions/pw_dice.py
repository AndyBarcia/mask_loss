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
    def forward(ctx, logits, targets, smooth=1.0):
        B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()

        output = mask_loss.forward_pw_dice_loss(logits, targets, smooth)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

def pairwise_dice_loss_py(logits, targets, smooth=1.0):
    """
    Computes the pairwise Dice loss and returns a list of tensors.

    This function calculates the Dice loss for each predicted class against every
    present ground truth class for each item in the batch. It upsamples the
    logits, creates one-hot representations for present target classes, and then
    computes the Dice loss.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, h, w) representing the
                             predicted logits for each class.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                                labels for the ground truth.
        smooth (float): A small value added to the numerator and denominator
                        for numerical stability.

    Returns:
        list[torch.Tensor]: A list of tensors where each element corresponds to a
                            batch item and contains a tensor of shape
                            (C, num_present_gt_classes) with the pairwise Dice losses.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"

    # Upsample logits to match the spatial dimensions of the targets
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')
    probs = torch.sigmoid(logits_up)

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
        has_class = y.sum(dim=(1, 2, 3)) > 0  # Shape: (B,)

        if not has_class.any():
            continue

        # Calculate intersection, probability sum, and target sum
        intersection = (probs * y).sum(dim=(2, 3))
        p_sum = probs.sum(dim=(2, 3))
        t_sum = y.sum(dim=(2, 3))

        # Calculate Dice score and Dice loss
        dice_score = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
        dice_loss = 1.0 - dice_score

        # Append the loss for batch items that have the current gt_class
        for b in range(B):
            if has_class[b]:
                batch_losses[b].append(dice_loss[b])

    # Stack the losses for each batch item to get tensors of shape (C, num_gt)
    final_tensors = [
        torch.stack(tensors, dim=1) if tensors else torch.empty((C, 0), device=device, dtype=logits.dtype)
        for tensors in batch_losses
    ]

    return final_tensors