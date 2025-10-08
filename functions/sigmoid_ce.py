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

class SigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, num_masks=None):
        if num_masks is None:
            B, C = logits.shape[:2]
            num_masks = B*C

        logits = logits.contiguous().float()
        targets = targets.contiguous()
        ctx.save_for_backward(logits, targets)
        output = mask_loss.forward_sigmoid_ce_loss(logits, targets, num_masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_sigmoid_ce_loss(grad_output, logits, targets)
        return grad_weights, None

def sigmoid_cross_entropy_loss_py(logits, targets, num_masks=None):
    """
    Naive approach: upsample logits (nearest) to high-res, build per-class one-hot targets,
    and compute BCEWithLogits per pixel then mean.
    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, C-1]
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')
    device = logits.device
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=logits.dtype)

    L = logits_up
    y = onehot
    maxL = torch.clamp(L, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L)))
    bce_elem = maxL - L * y + logexp

    num_masks = B*C if num_masks is None else num_masks
    loss = bce_elem.sum() / (num_masks * H_t * W_t)
    return loss

def sigmoid_cross_entropy_loss_efficient_py(logits, targets, num_masks=None):
    """
    Efficient count-based BCE-with-logits for non-mutually-exclusive multi-class case.
    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, C-1] (interpreted as one-hot per-class)
             H_t and W_t must be integer multiples of h and w respectively.
    Returns: scalar tensor (mean over all elements: B*C*H_t*W_t)
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"
    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation assumes equal scale factor for height and width (square blocks)")
    s = sH
    N2 = s * s

    device = logits.device
    targets_long = targets.long().to(device)

    # One-hot encode targets.
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=logits.dtype)  # (B,C,H_t,W_t)

    # Reshape for unfolding into high-res blocks. This will allows us
    # to count the number of ground truth classes in each block.
    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))  # (B*C, s*s, h*w)
    unf = unf.reshape(B, C, s * s, h * w)
    n_k = unf.sum(dim=2)  # (B, C, h*w)

    # Stable BCE-with-logits summed across block
    L = logits.reshape(B, C, h * w)
    maxL = torch.clamp(L, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L)))
    loss_block = N2 * maxL - L * n_k + N2 * logexp  # (B, C, h*w)

    num_masks = B*C if num_masks is None else num_masks
    loss = loss_block.sum() / (num_masks * H_t * W_t)
    return loss