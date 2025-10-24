from typing import Optional

import torch
from torch.autograd import Function

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseLabelLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        targets,
        background_index,
        scale,
        focal_gamma=None,
        focal_alpha=None,
    ):
        L, B, Q, C = logits.shape
        B_t, GT = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()
        fg = 0.0 if focal_gamma is None else float(focal_gamma)
        if fg < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if focal_alpha is None:
            fa = -1.0
        else:
            fa = float(focal_alpha)
            if not (0.0 <= fa <= 1.0):
                raise ValueError("focal_alpha must be in [0, 1]")
        output = mask_loss.forward_pw_sigmoid_label_loss(
            logits,
            targets,
            background_index if background_index is not None else -1,
            scale if scale is not None else 1.0,
            fg,
            fa,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None

def pairwise_label_loss_py(
    logits,
    targets,
    background_index=None,
    scale=1.0,
    focal_gamma: float = 0.0,
    focal_alpha: Optional[float] = None,
):
    """
    Computes pairwise binary cross-entropy (with logits) between each prediction
    vector and each ground-truth label slot.

    Args:
        logits (torch.Tensor): (L, B, Q, C) prediction logits.
        targets (torch.Tensor): (B, GT) integer class labels per image.
                                Padded entries must be -1.
        background_index (Optional[int]): If provided, DROP the column
                                targets[:, background_index] across the whole batch
                                (i.e., remove that GT slot for all images).
        scale (float): Scalar multiplier for the output.
        focal_gamma (float): Focal loss exponent. Use 0.0 for vanilla BCE.
        focal_alpha (Optional[float]): Weight for positives in [0, 1]. If ``None``
                            the standard BCE weighting is used.

    Returns:
        torch.Tensor: (L, B, Q, GT) if `background_index` is None, else
                      (L, B, Q, GT-1). For GT entries that are -1 (padding),
                      the corresponding loss is +inf.

    Notes:
        Stable BCE-with-logits decomposition:
          neg(z) = max(z,0) + log1p(exp(-|z|)) = BCEWithLogits(z, 0)
          BCE(z, one_hot(y)) = sum_c neg(z_c) - z_y
        We return the mean over classes (divide by C), analogous to spatial
        averaging in the pixelwise variant.
    """
    assert logits.dim() == 4, "logits must have shape (L, B, Q, C)"
    L, B, Q, C = logits.shape
    assert targets.dim() == 2 and targets.shape[0] == B, \
        "targets must have shape (B, GT) with same batch size as logits"

    device = logits.device
    _, GT = targets.shape

    # Optionally drop a fixed GT column across the batch
    if background_index is not None and background_index >= 0 and background_index < GT:
        assert GT > 0, "Cannot drop a column when GT == 0."
        bg = int(background_index)
        keep = torch.cat([
            torch.arange(0, bg, device=device),
            torch.arange(bg + 1, GT, device=device)
        ], dim=0)
        T = targets.index_select(1, keep)  # (B, GT-1)
    else:
        T = targets  # (B, GT)

    GT_out = T.shape[1]
    if GT_out == 0:
        return logits.new_empty((L, B, Q, 0))

    # Stable BCE pieces over class dim
    if focal_gamma is None:
        focal_gamma = 0.0
    focal_gamma = float(focal_gamma)
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")

    z = logits  # (L,B,Q,C)
    maxL  = torch.clamp(z, min=0.0)
    abs_z = torch.abs(z)
    logexp = torch.log1p(torch.exp(-abs_z))
    ce_neg = maxL + logexp              # (L,B,Q,C)
    max_neg = torch.clamp(-z, min=0.0)
    ce_pos = logexp + max_neg           # (L,B,Q,C)

    if focal_alpha is None:
        alpha_pos = logits.new_tensor(1.0)
        alpha_neg = logits.new_tensor(1.0)
    else:
        focal_alpha = float(focal_alpha)
        if not (0.0 <= focal_alpha <= 1.0):
            raise ValueError("focal_alpha must be in [0, 1]")
        alpha_pos = logits.new_tensor(focal_alpha)
        alpha_neg = logits.new_tensor(1.0 - focal_alpha)

    if focal_gamma == 0.0:
        mod_neg = logits.new_ones(z.shape)  # (L,B,Q,C)
        mod_pos = mod_neg
    else:
        probs = torch.sigmoid(z)
        mod_neg = probs.pow(focal_gamma)              # (L,B,Q,C)
        mod_pos = (1.0 - probs).pow(focal_gamma)      # (L,B,Q,C)

    base = (alpha_neg * mod_neg * ce_neg).sum(dim=3)  # (L,B,Q)

    # Gather z at the positive class index for each GT slot
    T_long = T.to(device=device, dtype=torch.long, non_blocking=True)
    # treat non-negative >= C as invalid too (set to +inf via mask)
    is_valid = (T_long >= 0) & (T_long < C)           # (B,GT_out)
    T_clip = T_long.clamp(min=0, max=max(C - 1, 0))   # safe for gather

    gather_idx = T_clip.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)
    ce_neg_pos = torch.gather(ce_neg, dim=3, index=gather_idx)  # (L,B,Q,GT_out)
    ce_pos_pos = torch.gather(ce_pos, dim=3, index=gather_idx)  # (L,B,Q,GT_out)
    mod_neg_pos = torch.gather(mod_neg, dim=3, index=gather_idx)  # (L,B,Q,GT_out)
    mod_pos_pos = torch.gather(mod_pos, dim=3, index=gather_idx)  # (L,B,Q,GT_out)

    # BCE(one-hot) per (l,b,q,g): mean over classes with focal modulation
    base = base.unsqueeze(-1)  # (L,B,Q,1)
    loss = (
        base
        + alpha_pos * mod_pos_pos * ce_pos_pos
        - alpha_neg * mod_neg_pos * ce_neg_pos
    ) / float(max(C, 1))

    # Apply +inf to invalid GT slots (-1 padding or out-of-range labels)
    mask = is_valid.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)
    out = torch.full_like(loss, float('inf'))
    out[mask] = loss[mask]

    return out * scale
