import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.utils.checkpoint import checkpoint

from ..utils import (
    calculate_uncertainty,
    point_sample,
    get_uncertain_point_coords_with_randomness,
)

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class SigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, num_masks, scale, focal_gamma=None, focal_alpha=None):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        if num_masks is None:
            num_masks = float(L * B * C)
        ctx.num_masks = num_masks
        ctx.scale = 1.0 if scale is None else float(scale)
        ctx.focal_gamma = 0.0 if focal_gamma is None else float(focal_gamma)
        if ctx.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")

        if focal_alpha is None:
            ctx.focal_alpha = -1.0
        else:
            ctx.focal_alpha = float(focal_alpha)
            if not (0.0 <= ctx.focal_alpha <= 1.0):
                raise ValueError("focal_alpha must be in [0, 1]")

        logits = logits.contiguous().float()
        targets = targets.contiguous()
        ctx.save_for_backward(logits, targets)
        output = mask_loss.forward_sigmoid_ce_loss(
            logits,
            targets,
            num_masks,
            ctx.scale,
            ctx.focal_gamma,
            ctx.focal_alpha,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_sigmoid_ce_loss(
            grad_output,
            logits,
            targets,
            ctx.num_masks,
            ctx.scale,
            ctx.focal_gamma,
            ctx.focal_alpha,
        )
        return grad_weights, None, None, None, None, None


def sigmoid_cross_entropy_loss_inefficient_py(logits, targets, num_masks=None, scale=1.0):
    """Naive BCE-with-logits computed by explicitly upsampling the logits.

    Args:
        logits: Tensor of shape (L, B, C, h, w)
        targets: Tensor of shape (B, H_t, W_t) with integer labels in [0, C-1]
    
    Returns:
        Tensor of shape (L,) containing the mean BCE loss for each level
    """

    L, B, C, h, w = logits.shape
    if L == 0:
        return logits.new_zeros((0,))
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"

    if num_masks is None:
        num_masks = float(L * B * C)

    logits_up = F.interpolate(
        logits.reshape(L * B, C, h, w), size=(H_t, W_t), mode="nearest"
    ).reshape(L, B, C, H_t, W_t)

    device = logits.device
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(
        dtype=logits.dtype
    )

    L_up = logits_up
    y = onehot.unsqueeze(0)
    maxL = torch.clamp(L_up, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L_up)))
    bce_elem = maxL - L_up * y + logexp

    return bce_elem.sum(dim=(1, 2, 3, 4)) / (num_masks * H_t * W_t) * scale


def sigmoid_cross_entropy_loss_py(logits, targets, num_masks=None, scale=1.0):
    """Efficient count-based BCE-with-logits for non-mutually-exclusive classes.

    Args:
        logits: Tensor of shape (L, B, C, h, w)
        targets: Tensor of shape (B, H_t, W_t) with integer labels in [0, C-1]

    Returns:
        Tensor of shape (L,) representing the mean BCE-with-logits for each level
    """

    L, B, C, h, w = logits.shape
    if L == 0:
        return logits.new_zeros((0,))
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"

    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation assumes equal scale factor for height and width (square blocks)")
    s = sH
    N2 = s * s

    if num_masks is None:
        num_masks = float(B * C)

    device = logits.device
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(
        dtype=logits.dtype
    )

    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))
    unf = unf.reshape(B, C, s * s, h * w)
    n_k = unf.sum(dim=2)  # (B, C, h*w)

    L_flat = logits.reshape(L, B, C, h * w)
    maxL = torch.clamp(L_flat, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L_flat)))

    loss_block = N2 * maxL - L_flat * n_k.unsqueeze(0) + N2 * logexp

    return loss_block.sum(dim=(1, 2, 3)) / (num_masks * H_t * W_t) * scale


def sigmoid_cross_entropy_loss_sampling_py(
    logits,  # (L, B, C, h, w) logits
    targets,  # (B, H_t, W_t) integer labels in [0, C-1]
    num_masks=None,  # normalization (defaults to L*B*C)
    num_points=5000,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
    scale=1.0,
):
    """Point-sampled BCE-with-logits performed per (level, image, class).

    Args:
        logits: Tensor of shape (L, B, C, h, w)
        targets: Tensor of shape (B, H_t, W_t) with integer labels in [0, C-1]

    Returns:
        Tensor of shape (L,) containing the sampled BCE loss for each level
    """

    L, B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"

    if num_masks is None:
        num_masks = float(B * C)

    device = logits.device
    dtype = logits.dtype

    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=dtype)

    per_class_logits = logits.reshape(L * B * C, 1, h, w)
    per_class_targets = onehot.unsqueeze(0).repeat(L, 1, 1, 1, 1).reshape(L * B * C, H_t, W_t)

    point_coords = get_uncertain_point_coords_with_randomness(
        per_class_logits,
        lambda logits: calculate_uncertainty(logits),
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    )  # -> (L*B*C, P, 2)

    sampled_logits = point_sample(per_class_logits, point_coords, align_corners=False)
    if sampled_logits.dim() == 4 and sampled_logits.size(-1) == 1:
        sampled_logits = sampled_logits.squeeze(-1)
    sampled_logits = sampled_logits.squeeze(1)  # (L*B*C, P)

    per_class_targets = per_class_targets.unsqueeze(1)
    sampled_labels = point_sample(per_class_targets, point_coords, align_corners=False)
    if sampled_labels.dim() == 4 and sampled_labels.size(-1) == 1:
        sampled_labels = sampled_labels.squeeze(-1)
    sampled_labels = sampled_labels.squeeze(1)  # (L*B*C, P)

    per_point_loss = F.binary_cross_entropy_with_logits(
        sampled_logits, sampled_labels, reduction="none"
    )

    P = sampled_logits.shape[-1]
    per_class_loss = per_point_loss.sum(dim=1).reshape(L, B, C)
    return per_class_loss.sum(dim=(1, 2)) / (num_masks * float(P)) * scale


def focal_cross_entropy_loss_py(
    logits, targets, num_masks=None, scale=1.0, gamma: float = 2.0
):
    """Efficient count-based sigmoid focal loss with per-(B,C) inverse-frequency weighting.

    For each (b, c) mask:
        alpha_pos[b,c] = M_bc / (M_bc + N_bc)   # M_bc = # negatives for that mask
        alpha_neg[b,c] = N_bc / (M_bc + N_bc)   # N_bc = # positives for that mask

    This makes small-area masks emphasize positives; large-area masks emphasize negatives.

    Args:
        logits: (L, B, C, h, w)
        targets: (B, H_t, W_t) with int labels in [0, C-1]
        num_masks: normalization factor over (L * B * C); defaults to L*B*C
        scale: extra scalar multiplier
        gamma: focal modulation exponent

    Returns:
        (L,) tensor: mean focal loss for each level
    """
    L, B, C, h, w = logits.shape
    if L == 0:
        return logits.new_zeros((0,))
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"

    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation assumes equal scale factor for height and width (square blocks)")
    s = sH
    N2 = s * s  # high-res pixels per low-res location

    if num_masks is None:
        num_masks = float(L * B * C)

    device, dtype = logits.device, logits.dtype

    # One-hot targets at high resolution: (B, C, H_t, W_t)
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=dtype)

    # Count positives per (B, C, h*w) low-res block
    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))  # (B*C, s*s, h*w)
    unf = unf.reshape(B, C, s * s, h * w)
    pos_counts = unf.sum(dim=2)                 # (B, C, h*w)
    neg_counts = N2 - pos_counts                # (B, C, h*w)

    # Per-(B,C) totals across spatial locations
    pos_tot_bc = pos_counts.sum(dim=-1)         # (B, C)
    neg_tot_bc = neg_counts.sum(dim=-1)         # (B, C)
    denom_bc = (pos_tot_bc + neg_tot_bc).clamp(min=1.0)  # equals H_t*W_t

    # Per-(B,C) inverse-frequency alphas
    alpha_pos_bc = (neg_tot_bc / denom_bc).to(dtype=dtype)  # M/(M+N)
    alpha_neg_bc = (pos_tot_bc / denom_bc).to(dtype=dtype)  # N/(M+N)

    alpha_pos_bc = alpha_pos_bc.clamp(max=0.5)
    alpha_neg_bc = alpha_neg_bc.clamp(min=0.5)

    # Broadcast alphas to (1, B, C, 1) to apply across L and spatial
    alpha_pos_bc = alpha_pos_bc.unsqueeze(0).unsqueeze(-1)  # (1, B, C, 1)
    alpha_neg_bc = alpha_neg_bc.unsqueeze(0).unsqueeze(-1)  # (1, B, C, 1)

    # Flatten logits over spatial
    L_flat = logits.reshape(L, B, C, h * w)

    # Stable BCE pieces
    ce_pos = F.softplus(-L_flat)   # -log(sigmoid(z))
    ce_neg = F.softplus(L_flat)    # -log(1 - sigmoid(z))

    # Focal modulation
    p = torch.sigmoid(L_flat)
    mod_pos = (1.0 - p) ** gamma
    mod_neg = p ** gamma

    # Apply counts, per-(B,C) alphas, and focal modulation
    loss_pos = alpha_pos_bc * mod_pos * ce_pos * pos_counts.unsqueeze(0)
    loss_neg = alpha_neg_bc * mod_neg * ce_neg * neg_counts.unsqueeze(0)
    loss_block = loss_pos + loss_neg  # (L, B, C, h*w)

    # Same normalization as original
    return loss_block.sum(dim=(1, 2, 3)) / (num_masks * H_t * W_t) * scale
