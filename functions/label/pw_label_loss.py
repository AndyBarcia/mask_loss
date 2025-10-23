import torch
import torch.nn.functional as F
from torch.autograd import Function

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class PairwiseLabelLossType:
    BCE = 0
    BCE_FOCAL = 1
    CE = 2
    CE_FOCAL = 3


class PairwiseSigmoidCELossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        targets,
        background_index,
        scale,
        loss_type=PairwiseLabelLossType.BCE,
        focal_alpha=None,
        focal_gamma=2.0,
    ):
        L, B, Q, C = logits.shape
        B_t, GT = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.forward_pw_label_loss(
            logits,
            targets,
            background_index if background_index is not None else -1,
            scale if scale is not None else 1.0,
            int(loss_type),
            float(focal_alpha) if focal_alpha is not None else -1.0,
            float(focal_gamma),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 7


def pairwise_label_loss_py(
    logits,
    targets,
    background_index=None,
    scale=1.0,
    loss_type=PairwiseLabelLossType.BCE,
    focal_alpha=None,
    focal_gamma=2.0,
):
    """Compute pairwise classification costs between predictions and labels.

    ``loss_type`` selects among sigmoid BCE, sigmoid focal BCE, softmax cross
    entropy, and softmax focal loss.  When a focal variant is used,
    ``focal_gamma`` controls the focusing factor and ``focal_alpha`` (if not
    ``None``) scales the positive term while negatives keep unit weight.

    Args:
        logits (torch.Tensor): (L, B, Q, C) prediction logits.
        targets (torch.Tensor): (B, GT) integer class labels per image.
                                Padded entries must be -1.
        background_index (Optional[int]): If provided, DROP the column
                                targets[:, background_index] across the whole batch
                                (i.e., remove that GT slot for all images).
        scale (float): Scalar multiplier for the output.

    Returns:
        torch.Tensor: (L, B, Q, GT) if `background_index` is None, else
                      (L, B, Q, GT-1). For GT entries that are -1 (padding),
                      the corresponding loss is +inf.
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

    z = logits  # (L,B,Q,C)
    T_long = T.to(device=device, dtype=torch.long, non_blocking=True)
    is_valid = (T_long >= 0) & (T_long < C)           # (B,GT_out)
    T_clip = T_long.clamp(min=0, max=max(C - 1, 0))   # (B,GT_out)
    gather_idx = T_clip.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)
    valid_mask = is_valid.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)

    loss_type = int(loss_type)

    if loss_type == PairwiseLabelLossType.BCE:
        maxL  = torch.clamp(z, min=0.0)
        logexp = torch.log1p(torch.exp(-torch.abs(z)))
        neg = maxL + logexp                 # (L,B,Q,C)
        sum_neg = neg.sum(dim=3, keepdim=True)  # (L,B,Q,1)

        z_pos = torch.gather(z, dim=3, index=gather_idx)
        loss = (sum_neg - z_pos) / float(max(C, 1))

    elif loss_type == PairwiseLabelLossType.BCE_FOCAL:
        gamma = float(focal_gamma)
        if focal_alpha is None:
            alpha_pos = 1.0
            alpha_neg = 1.0
        else:
            alpha = max(0.0, min(float(focal_alpha), 1.0))
            alpha_pos = alpha
            alpha_neg = 1.0 - alpha

        neg = F.softplus(z)              # (L,B,Q,C)
        sig = torch.sigmoid(z)           # (L,B,Q,C)

        mod_neg = sig.pow(gamma) * neg  # (L,B,Q,C)
        sum_neg = mod_neg.sum(dim=3, keepdim=True)

        z_pos = torch.gather(z, dim=3, index=gather_idx)
        sig_pos = torch.sigmoid(z_pos)
        softplus_pos = F.softplus(z_pos)
        softplus_neg = F.softplus(-z_pos)

        neg_except = sum_neg - (sig_pos.pow(gamma) * softplus_pos)
        pos_term = softplus_neg * torch.pow(1.0 - sig_pos, gamma)
        loss = (alpha_pos * pos_term + alpha_neg * neg_except) / float(max(C, 1))

    elif loss_type == PairwiseLabelLossType.CE:
        logsumexp = torch.logsumexp(z, dim=3, keepdim=True)  # (L,B,Q,1)
        z_pos = torch.gather(z, dim=3, index=gather_idx)
        loss = logsumexp - z_pos

    elif loss_type == PairwiseLabelLossType.CE_FOCAL:
        gamma = float(focal_gamma)
        alpha = 1.0 if focal_alpha is None else float(focal_alpha)
        logsumexp = torch.logsumexp(z, dim=3, keepdim=True)  # (L,B,Q,1)
        z_pos = torch.gather(z, dim=3, index=gather_idx)
        log_p = z_pos - logsumexp           # (L,B,Q,GT_out)
        p = torch.exp(log_p)                # (L,B,Q,GT_out)
        focal = torch.pow(1.0 - p, gamma)   # (L,B,Q,GT_out)
        loss = -alpha * focal * log_p

    else:
        raise ValueError(f"Unsupported pairwise label loss type: {loss_type}")

    out = torch.full_like(loss, float('inf'))
    out[valid_mask] = loss[valid_mask]
    return out * scale
