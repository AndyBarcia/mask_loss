import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.utils.checkpoint import checkpoint

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class DiceLossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, smooth, num_masks):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        if num_masks is None:
            num_masks = float(L * B * C)
        ctx.num_masks = num_masks

        logits = logits.contiguous().float()
        targets = targets.contiguous().long()
        ctx.smooth = smooth
        output, int_sum, p_sum, t_sum = mask_loss.forward_dice_loss(
            logits, targets, smooth, num_masks
        )
        ctx.save_for_backward(logits, targets, int_sum, p_sum, t_sum)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, int_sum, p_sum, t_sum = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_dice_loss(
            grad_output,
            logits,
            targets,
            int_sum,
            p_sum,
            t_sum,
            ctx.smooth,
            ctx.num_masks,
        )
        return grad_weights, None, None, None


def dice_loss_inefficient_py(logits, targets, smooth=1e-6, num_masks=None):
    """Naive Dice loss that explicitly upsamples logits to the target resolution.

    Args:
        logits: Tensor of shape (L, B, C, h, w)
        targets: Tensor of shape (B, H_t, W_t) with integer labels in [0, C-1]
        smooth: Small float to avoid division by zero
    Returns:
        Tensor of shape (L,) containing the mean Dice loss for each level over (B, C)
    """

    L, B, C, h, w = logits.shape
    if L == 0:
        return logits.new_zeros((0,))
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"

    if num_masks is None:
        num_masks = float(L * B * C)

    # Upsample logits to high-res (nearest)
    logits_up = F.interpolate(
        logits.reshape(L * B, C, h, w), size=(H_t, W_t), mode="nearest"
    ).reshape(L, B, C, H_t, W_t)

    device = logits.device
    targets_long = targets.long().to(device)

    # one-hot targets: (B, C, H_t, W_t)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(
        dtype=logits.dtype
    )

    # predicted probabilities per-pixel per-class
    probs = torch.sigmoid(logits_up)  # (L, B, C, H_t, W_t)

    # sums over high-res spatial dims
    intersection = (probs * onehot.unsqueeze(0)).sum(dim=(3, 4))  # (L, B, C)
    p_sum = probs.sum(dim=(3, 4))  # (L, B, C)
    t_sum = onehot.sum(dim=(2, 3)).unsqueeze(0)  # (1, B, C)

    dice = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
    loss = 1.0 - dice

    norm = num_masks / float(L)
    return loss.sum(dim=(1, 2)) / norm


def dice_loss_py(logits, targets, smooth=1e-6, num_masks=None):
    """Efficient count-based Dice loss consistent with nearest upsampling behavior.

    Assumes H_t and W_t are integer multiples of h and w respectively and that
    nearest upsampling repeats the same logit value across each s x s block.

    Args:
        logits: Tensor of shape (L, B, C, h, w)
        targets: Tensor of shape (B, H_t, W_t) with integer labels in [0, C-1]
        smooth: Small float to avoid division by zero

    Returns:
        Tensor of shape (L,) containing the mean Dice loss for each level over (B, C)
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
    N2 = s * s  # number of high-res pixels per low-res pixel

    if num_masks is None:
        num_masks = float(L * B * C)

    device = logits.device
    targets_long = targets.long().to(device)

    # One-hot encode targets at high resolution (B, C, H_t, W_t)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(
        dtype=logits.dtype
    )

    # Unfold the one-hot high-res maps into sxs blocks to count positives per block.
    # Reshape to (B*C, 1, H_t, W_t) to use F.unfold conveniently.
    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    # After unfold: shape (B*C, s*s, h*w)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))
    # reshape -> (B, C, s*s, h*w)
    unf = unf.reshape(B, C, s * s, h * w)
    # counts of target-positive pixels per block: (B, C, h*w)
    n_k = unf.sum(dim=2)

    # logits at low-res positions: (L, B, C, h, w) -> (L, B, C, h*w)
    L_flat = logits.reshape(L, B, C, h * w)

    # predicted probability per low-res position (same for all pixels in block)
    p = torch.sigmoid(L_flat)  # (L, B, C, h*w)

    # Sum of predicted probabilities over the whole high-res map for each (L,B,C)
    p_sum_total = N2 * p.sum(dim=3)  # (L, B, C)

    # Sum of target positives over whole high-res map for each (B,C)
    t_sum_total = n_k.sum(dim=2).unsqueeze(0)  # (1, B, C)

    # Intersection total per (L,B,C)
    intersection_total = (p * n_k.unsqueeze(0)).sum(dim=3)  # (L, B, C)

    dice = (2.0 * intersection_total + smooth) / (p_sum_total + t_sum_total + smooth)
    loss = 1.0 - dice

    norm = num_masks / float(L)
    return loss.sum(dim=(1, 2)) / norm
