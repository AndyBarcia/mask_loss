import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from .utils import point_sample

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseSigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, background_index, scale):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.forward_pw_sigmoid_ce_loss(
            logits, 
            targets, 
            background_index if background_index is not None else -1,
            scale if scale is not None else 1.0
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

def pairwise_sigmoid_cross_entropy_loss_inneficient_py(logits, targets, background_index=None, scale=1.0):
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

    return pairwise_loss * scale

@torch.no_grad()
def _counts_per_cell_per_class(targets_long, h, w, s, GT_all):
    """
    targets_long: (B, H_t, W_t) long
    Returns counts: (B, h*w, GT_all) with the number of pixels of each GT class
    inside every downsampled cell.
    """
    B, H_t, W_t = targets_long.shape
    N2 = s * s
    J = h * w  # number of coarse cells per image

    # Split target into s×s blocks matching each coarse logit cell: (B, N2, J)
    patches = rearrange(
        targets_long, 'b (h s1) (w s2) -> b (s1 s2) (h w)', s1=s, s2=s
    )  # (B, N2, J)

    # Global cell index (0..B*J-1) for every pixel inside each cell
    # shape (B, N2, J) -> flatten to (B*N2*J,)
    j_ids   = torch.arange(J, device=targets_long.device).view(1, 1, J).expand(B, N2, J)
    b_offs  = (torch.arange(B, device=targets_long.device) * J).view(B, 1, 1).expand(B, N2, J)
    global_cell = (b_offs + j_ids).reshape(-1)                           # (B*N2*J,)
    classes_flat = patches.reshape(-1).to(torch.long)                    # (B*N2*J,)

    # Bin by (global_cell, class) via a single bincount on packed keys
    keys = global_cell * GT_all + classes_flat                            # (B*N2*J,)
    counts = torch.bincount(
        keys, minlength=B * J * GT_all
    ).view(B * J, GT_all).view(B, J, GT_all)                             # (B, J, GT_all)
    return counts


def pairwise_sigmoid_cross_entropy_loss_py(logits, targets, background_index=None, scale=1.0):
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
    assert H_t % h == 0 and W_t % w == 0, "Target dims must be multiples of logit dims"
    sH, sW = H_t // h, W_t // w
    if sH != sW:
        raise ValueError("Height/width downsample factors must be equal.")
    s = sH
    N2 = s * s
    J = h * w

    device = logits.device
    dtype  = logits.dtype

    # Prepare logits (L,B,C,J) and stable BCE pieces
    z = logits.reshape(L, B, C, J)

    # neg(z) = BCEWithLogits(z, 0) = max(z,0) + log1p(exp(-|z|))
    maxL = torch.clamp(z, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(z)))
    neg = maxL + logexp # (L,B,C,J)

    # Sum of the "all negatives" part across the N2 pixels per cell
    # (same for every GT class):  sum_j N2 * neg(z_j)
    base_neg = (N2 * neg).sum(dim=3) # (L,B,C)

    # Per-cell per-class positive counts k (how many pixels equal that class) 
    targets_long = targets.to(device=device, dtype=torch.long, non_blocking=True)
    gt_max = int(targets_long.max().item()) if targets_long.numel() else -1
    GT_all = gt_max + 1
    if GT_all <= 0:
        # No valid labels: return an empty last dim
        return logits.new_empty((L, B, C, 0))

    counts_all = _counts_per_cell_per_class(targets_long, h, w, s, GT_all)  # (B, J, GT_all)

    # Select GT columns (optionally drop background)
    if background_index is None:
        idxs = torch.arange(GT_all, device=device)
    else:
        idxs = torch.tensor(
            [i for i in range(GT_all) if i != background_index],
            device=device, dtype=torch.long
        )

    G = int(idxs.numel())
    if G == 0:
        return logits.new_empty((L, B, C, 0))

    counts = counts_all.index_select(2, idxs) # (B, J, G)
    counts_f = counts.to(dtype) # float for einsum

    # sum_j [-z * k] + sum_j [N2 * neg]
    # term1 = -einsum over cells J: (L,B,C,J) x (B,J,G) -> (L,B,C,G)
    term1 = -torch.einsum('lbcj,bjg->lbcg', z, counts_f)
    loss = (term1 + base_neg.unsqueeze(-1)) / (H_t * W_t)  # (L,B,C,G)

    # Set +inf where the GT class doesn't appear in that image
    present = (counts.sum(dim=1) > 0) # (B,G) bool
    mask = present.unsqueeze(0).unsqueeze(2).expand(L, B, C, G)
    out = torch.full_like(loss, float('inf'))
    out[mask] = loss[mask]

    return out * scale

