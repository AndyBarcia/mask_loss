from typing import Optional

import torch
from torch.autograd import Function

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")


class PairwiseSoftmaxLabelLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        background_index: Optional[int],
        scale: Optional[float],
    ) -> torch.Tensor:
        L, B, Q, C = logits.shape
        B_t, GT = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()

        output = mask_loss.forward_pw_softmax_label_loss(
            logits,
            targets,
            background_index if background_index is not None else -1,
            scale if scale is not None else 1.0,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None


def pairwise_softmax_label_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    background_index: Optional[int] = None,
    scale: Optional[float] = 1.0,
) -> torch.Tensor:
    """CUDA pairwise softmax cross-entropy loss."""

    return PairwiseSoftmaxLabelLossFunction.apply(
        logits,
        targets,
        background_index,
        scale,
    )


def pairwise_softmax_label_loss_py(
    logits: torch.Tensor,
    targets: torch.Tensor,
    background_index: Optional[int] = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute pairwise softmax cross-entropy between predictions and GT labels.

    Args:
        logits (torch.Tensor): (L,B,Q,C) prediction logits.
        targets (torch.Tensor): (B,GT) integer class labels per image.
                                Padded entries must be -1.
        background_index (Optional[int]): If provided, DROP the column
                                ``targets[:, background_index]`` across the
                                whole batch (i.e., remove that GT slot for all
                                images).
        scale (float): Scalar multiplier for the output.

    Returns:
        torch.Tensor: (L,B,Q,GT_out) softmax cross-entropy loss where GT_out
                      equals ``GT`` after an optional background removal. The
                      loss is +inf for invalid GT entries (padding or out of
                      range labels).
    """
    assert logits.dim() == 4, "logits must have shape (L, B, Q, C)"
    L, B, Q, C = logits.shape
    assert targets.dim() == 2 and targets.shape[0] == B, (
        "targets must have shape (B, GT) with same batch size as logits"
    )

    device = logits.device
    _, GT = targets.shape

    # Optionally drop a fixed GT column across the batch
    if background_index is not None and 0 <= background_index < GT:
        assert GT > 0, "Cannot drop a column when GT == 0."
        bg = int(background_index)
        keep = torch.cat([
            torch.arange(0, bg, device=device),
            torch.arange(bg + 1, GT, device=device),
        ], dim=0)
        T = targets.index_select(1, keep)  # (B,GT-1)
    else:
        T = targets  # (B,GT)

    GT_out = T.shape[1]
    if GT_out == 0:
        return logits.new_empty((L, B, Q, 0))

    log_probs = torch.log_softmax(logits, dim=-1)  # (L,B,Q,C)

    T_long = T.to(device=device, dtype=torch.long, non_blocking=True)
    is_valid = (T_long >= 0) & (T_long < C)  # (B,GT_out)
    T_clip = T_long.clamp(min=0, max=max(C - 1, 0))

    gather_idx = T_clip.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)
    losses = -torch.gather(log_probs, dim=3, index=gather_idx)  # (L,B,Q,GT_out)

    mask = is_valid.view(1, B, 1, GT_out).expand(L, B, Q, GT_out)
    out = torch.full_like(losses, float("inf"))
    out[mask] = losses[mask]
    return out * float(scale)
