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
    def forward(ctx, logits, targets, background_index=None):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.forward_pw_sigmoid_ce_loss(
            logits, 
            targets, 
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

def pairwise_sigmoid_cross_entropy_loss_py(logits, targets, background_index=None):
    """
    Computes pairwise sigmoid cross-entropy loss.

    This function calculates the loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the binary cross-entropy with logits.

    Args:
        logits (torch.Tensor): A tensor of shape (L, B, C, h, w) representing the
                            predicted logits for each class. L is the number of 
                            layers, B is the batch size, C is the number of classes, 
                            and h, w are the spatial dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                            labels for the ground truth. H_t and W_t are the
                            spatial dimensions of the targets.
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
    # Validate & extract shapes
    assert logits.dim() == 5, "logits must have shape (L, B, C, h, w)"
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

    targets_long = targets.long().to(device)

    # Determine GT classes present
    gt_max = targets_long.max().item()
    GT_all = gt_max + 1

    if background_index is None:
        gt_classes = list(range(GT_all))
    else:
        gt_classes = [i for i in range(GT_all) if i != background_index]

    GT = len(gt_classes)

    # Initialize output with +inf
    pairwise_loss = torch.full((L, B, C, GT), torch.inf, device=device, dtype=dtype)

    # Precompute parts used in stable BCE-with-logits
    # We'll compute per-ground-truth-class using broadcasting
    for out_idx, gt_class in enumerate(gt_classes):
        # Create binary target mask y: shape (B, H_t, W_t)
        y_mask = (targets_long == gt_class).to(dtype=dtype)  # (B, H_t, W_t)

        # Determine which batch elements contain this GT
        has_class = (y_mask.sum(dim=(1, 2)) > 0)  # (B,)

        if not has_class.any():
            # No batch element has this class — skip (pairwise_loss stays +inf)
            continue

        # Broadcast y_mask to match logits_up: (L, B, C, H_t, W_t)
        # First shape to (1, B, 1, H_t, W_t) and let broadcasting do the rest
        y_b = y_mask.unsqueeze(0).unsqueeze(2)  # (1, B, 1, H_t, W_t)

        # Stable BCE-with-logits (broadcasts y_b over L and C)
        maxL = torch.clamp(logits_up, min=0.0)
        logexp = torch.log1p(torch.exp(-torch.abs(logits_up)))
        bce_elem = maxL - logits_up * y_b + logexp  # (L, B, C, H_t, W_t)

        # Sum spatially and normalize by area
        loss_per_class = bce_elem.sum(dim=(3, 4)) / (H_t * W_t)  # (L, B, C)

        # Build mask to select where to write loss (expand has_class to (L,B,C))
        has_class_expand = has_class.unsqueeze(0).unsqueeze(-1).expand(L, B, C)  # (L,B,C)

        # Use torch.where to leave +inf where has_class is False
        inf_tensor = torch.tensor(torch.inf, device=device, dtype=dtype)
        pairwise_loss[:, :, :, out_idx] = torch.where(
            has_class_expand,
            loss_per_class,
            inf_tensor
        )

    return pairwise_loss


def pairwise_sigmoid_cross_entropy_loss_efficient_py(logits, targets, background_index=None):
    """
    Computes pairwise sigmoid cross-entropy loss in an efficient way.

    This function calculates the loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the binary cross-entropy with logits.

    Args:
        logits (torch.Tensor): A tensor of shape (L, B, C, h, w) representing the
                            predicted logits for each class. L is the number of 
                            layers, B is the batch size, C is the number of classes, 
                            and h, w are the spatial dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                            labels for the ground truth. H_t and W_t are the
                            spatial dimensions of the targets.
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
    assert logits.dim() == 5, "logits must have shape (L, B, C, h, w)"
    L, B, C, h, w = logits.shape
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
    dtype = logits.dtype
    targets_long = targets.long().to(device)

    # Determine the full range of GT classes present in the targets
    gt_max = targets_long.max().item()
    GT_all = gt_max + 1

    # Build list of ground-truth classes to evaluate, optionally excluding background
    if background_index is None:
        gt_classes = list(range(GT_all))
    else:
        gt_classes = [i for i in range(GT_all) if i != background_index]

    GT = len(gt_classes)

    # Initialize to infinity: (L, B, C, GT)
    pairwise_loss = torch.full((L, B, C, GT), torch.inf, device=device, dtype=dtype)

    # Reshape logits for efficient per-cell operations: (L, B, C, h*w)
    L_flat = logits.reshape(L, B, C, h * w)

    # Precompute parts of the stable BCE-with-logits formula
    maxL = torch.clamp(L_flat, min=0.0)  # (L,B,C,h*w)
    logexp = torch.log1p(torch.exp(-torch.abs(L_flat)))  # (L,B,C,h*w)

    # Iterate over ground-truth classes (usually a small loop)
    for out_idx, gt_class in enumerate(gt_classes):
        # Binary mask for current GT: (B, H_t, W_t) -> then (B,1,H_t,W_t)
        onehot_gt = (targets_long == gt_class).unsqueeze(1).to(dtype=dtype)  # (B,1,H_t,W_t)

        # Check which batch elements contain this GT
        has_class = (onehot_gt.sum(dim=(2, 3)) > 0).squeeze(1)  # (B,)

        if not has_class.any():
            # No examples in this batch have this GT — leave +inf for this column
            continue

        # Unfold to count number of GT pixels falling into each logit cell
        Bc = onehot_gt.reshape(B, 1, H_t, W_t)  # (B,1,H_t,W_t)
        unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))  # (B, s*s, h*w)
        # Note: F.unfold returns (B, C * kernelH * kernelW, L), but since C==1 here we get (B, s*s, h*w)
        unf = unf.reshape(B, s * s, h * w)  # (B, s*s, h*w)
        n_k = unf.sum(dim=1)  # (B, h*w)  -- counts per cell per batch

        # Expand n_k to match L and channel dims: (L, B, 1, h*w)
        n_k_exp = n_k.unsqueeze(0).unsqueeze(2).expand(L, B, 1, h * w)

        # Compute stable BCE-with-logits summed across the block for each predicted class
        # loss_block shape: (L, B, C, h*w)
        loss_block = N2 * maxL - L_flat * n_k_exp + N2 * logexp

        # Sum spatially and normalize by area -> (L, B, C)
        loss_sum = loss_block.sum(dim=3) / (H_t * W_t)

        # Build selection mask where GT exists: expand (B,) -> (L,B,C)
        has_class_expand = has_class.unsqueeze(0).unsqueeze(-1).expand(L, B, C)

        # Use torch.where to keep +inf where class absent
        inf_tensor = torch.tensor(torch.inf, device=device, dtype=dtype)
        pairwise_loss[:, :, :, out_idx] = torch.where(
            has_class_expand,
            loss_sum,
            inf_tensor
        )

    return pairwise_loss