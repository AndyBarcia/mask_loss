import math
import torch
from torch.autograd import Function
from einops import rearrange, einsum
from torch.utils.checkpoint import checkpoint

from ..utils import point_sample, counts_per_cell_per_class

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseDiceLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        targets,
        smooth,
        background_index,
        scale,
        uncertainty_gamma=None,
        uncertainty_gamma_min=None,
    ):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        logits = logits.contiguous().float()
        targets = targets.contiguous()

        ug = 1.0 if uncertainty_gamma is None else float(uncertainty_gamma)
        if ug < 0.0:
            raise ValueError("uncertainty_gamma must be non-negative")
        if uncertainty_gamma_min is None:
            ug_min = 0.05
        else:
            ug_min = float(uncertainty_gamma_min)
            if not (0.0 <= ug_min <= 1.0):
                raise ValueError("uncertainty_gamma_min must be in [0, 1]")

        output = mask_loss.forward_pw_dice_loss(
            logits,
            targets,
            smooth if smooth is not None else 1.0,
            background_index if background_index is not None else -1,
            scale if scale is not None else 1.0,
            ug,
            ug_min,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None

def pairwise_dice_loss_py(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    background_index: int = None,
    scale: float = 1.0,
    uncertainty_gamma: float = 1.0,
    uncertainty_gamma_min: float = 0.05,
):
    """
    Pairwise Dice loss without upsampling or per-class loops.

    Shapes:
      logits:  (L, B, C, h, w)
      targets: (B, H_t, W_t) integer labels

    Returns:
      (L, B, C, G) where G = #classes considered (optionally excluding background).
      Entries are +inf where that class is absent in the image.

    Args:
      uncertainty_gamma: Exponent applied to the normalized Bernoulli entropy to
        build per-pixel weights (higher -> sharper focus on uncertain pixels).
      uncertainty_gamma_min: Minimum allowed weight to avoid zeroing confident
        regions entirely.
    """
    assert logits.dim() == 5, "logits must have shape (L, B, C, h, w)"
    L, B, C, h, w = logits.shape
    Bt, Ht, Wt = targets.shape
    assert B == Bt, "Batch size mismatch between logits and targets"
    assert Ht % h == 0 and Wt % w == 0, "Target dims must be integer multiples of logit dims"
    sH, sW = Ht // h, Wt // w
    if sH != sW:
        raise ValueError("Height/width scale factors must be equal.")
    s = sH
    N2 = s * s
    J = h * w

    device = logits.device
    dtype  = logits.dtype

    if uncertainty_gamma < 0.0:
        raise ValueError("uncertainty_gamma must be non-negative")
    if not (0.0 <= uncertainty_gamma_min <= 1.0):
        raise ValueError("uncertainty_gamma_min must be in [0, 1]")

    # Flatten spatial and get probabilities at coarse grid
    z = logits.reshape(L, B, C, J)
    p = torch.sigmoid(z)                                     # (L,B,C,J)

    # Bernoulli entropy based weights (treated as constants during backward)
    eps = 1e-12
    probs_detached = p.detach()
    probs_clamped = probs_detached.clamp(min=eps, max=1.0 - eps)
    entropy = -(
        probs_clamped * torch.log(probs_clamped)
        + (1.0 - probs_clamped) * torch.log(1.0 - probs_clamped)
    )
    entropy = entropy / math.log(2.0)
    weights = torch.clamp(
        entropy.pow(uncertainty_gamma),
        min=uncertainty_gamma_min,
        max=1.0,
    ).detach()                                              # (L,B,C,J)

    weighted_p = weights * p

    targets_long = targets.to(device=device, dtype=torch.long, non_blocking=True)
    gt_max = int(targets_long.max().item()) if targets_long.numel() else -1
    GT_all = gt_max + 1
    if GT_all <= 0:
        return logits.new_empty((L, B, C, 0))

    # Per-cell per-class positive counts k_{b, j, g}
    counts_all = counts_per_cell_per_class(targets_long, h, w, s, GT_all)  # (B, J, GT_all)

    # Select GT columns (optionally drop background)
    if background_index is None:
        idxs = torch.arange(GT_all, device=device)
    else:
        idxs = torch.tensor([i for i in range(GT_all) if i != background_index],
                            device=device, dtype=torch.long)

    G = int(idxs.numel())
    if G == 0:
        return logits.new_empty((L, B, C, 0))

    counts = counts_all.index_select(2, idxs)                # (B, J, G)
    counts_f = counts.to(dtype)                              # float for einsum
    t_sum = counts.sum(dim=1)                                # (B, G) for presence mask

    # Dice pieces:
    # intersection = sum_j p_{lbcj} * k_{bjg}
    inter = torch.einsum('lbcj,bjg->lbcg', weighted_p, counts_f)  # (L,B,C,G)

    # p_sum over full-resolution = sum_j N2 * p_{lbcj}
    p_sum = (N2 * weighted_p).sum(dim=3)                     # (L,B,C)

    # t_sum over full-resolution = sum_j k_{bjg}
    weighted_targets = torch.einsum('lbcj,bjg->lbcg', weights, counts_f)

    # Dice score and loss
    denom = p_sum[..., None] + weighted_targets + smooth
    dice_score = (2.0 * inter + smooth) / denom
    dice_loss  = 1.0 - dice_score                            # (L,B,C,G)

    # +inf where class absent in that image
    present = (t_sum > 0)                                    # (B,G)
    mask = present.unsqueeze(0).unsqueeze(2).expand(L, B, C, G)
    out = torch.full_like(dice_loss, float('inf'))
    out[mask] = dice_loss[mask]

    return out * scale


def pairwise_dice_loss_sampling_py(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
    background_index: int = None,
    scale: float = 1.0,
    num_points: int = 5000,
    align_corners: bool = False,
):
    """
    Pairwise Dice loss with *point sampling*.

    For each image we draw one shared set of random points and evaluate Dice
    between every (L,C) prediction and each GT class using only those points.

    Shapes:
      logits:  (L, B, C, h, w)
      targets: (B, H_t, W_t) integer labels

    Returns:
      (L, B, C, G) with +inf for classes absent in that image.
    """
    assert logits.dim() == 5, "logits must have shape (L, B, C, h, w)"
    L, B, C, h, w = logits.shape
    Bt, Ht, Wt = targets.shape
    assert B == Bt, "Batch size mismatch"
    assert num_points > 0, "num_points must be > 0"

    device = logits.device
    dtype  = logits.dtype

    targets_long = targets.to(device=device, dtype=torch.long, non_blocking=True)
    gt_max = int(targets_long.max().item()) if targets_long.numel() else -1
    GT_all = gt_max + 1
    if GT_all <= 0:
        return logits.new_empty((L, B, C, 0))

    # Global class list (columns of output)
    if background_index is None:
        gt_classes: List[int] = list(range(GT_all))
    else:
        gt_classes = [k for k in range(GT_all) if k != background_index]
    G = len(gt_classes)
    if G == 0:
        return logits.new_empty((L, B, C, 0))

    id2col = {cls_id: i for i, cls_id in enumerate(gt_classes)}
    out = torch.full((L, B, C, G), float('inf'), device=device, dtype=dtype)

    for b in range(B):
        labels_b = targets_long[b]  # (Ht, Wt)

        # Which classes appear in this image?
        present_ids = torch.unique(labels_b).tolist()
        if background_index is not None:
            present_ids = [k for k in present_ids if k != background_index]
        if len(present_ids) == 0:
            continue

        cols = torch.tensor([id2col[k] for k in present_ids], device=device, dtype=torch.long)
        Gb = len(present_ids)

        # Shared random points in normalized coords [0,1]^2
        coords = torch.rand(1, num_points, 2, device=device)  # (1, P, 2)

        # Sample predictions (differentiable)
        z_b = logits[:, b]                         # (L, C, h, w)
        z_flat = z_b.reshape(L * C, 1, h, w)       # (N, 1, h, w)
        preds_pts = point_sample(
            z_flat, coords.repeat(z_flat.shape[0], 1, 1), align_corners=align_corners
        ).squeeze(1)                               # (N, P)
        p_pts = torch.sigmoid(preds_pts)           # (N, P)

        # Sample labels at the same points (nearest indexing; constants -> no grad)
        xy = coords[0]                             # (P, 2)
        ix = torch.clamp((xy[:, 0] * (Wt - 1)).long(), 0, Wt - 1)
        iy = torch.clamp((xy[:, 1] * (Ht - 1)).long(), 0, Ht - 1)
        labels_pts = labels_b[iy, ix]              # (P,)

        class_ids_b = torch.tensor(present_ids, device=device, dtype=torch.long)
        tgt_pts = (labels_pts[None, :] == class_ids_b[:, None]).to(p_pts.dtype)  # (Gb, P)

        # Dice over points
        inter = torch.einsum("np,mp->nm", p_pts, tgt_pts)                 # (N, Gb)
        p_sum = p_pts.sum(dim=1, keepdim=True)                            # (N, 1)
        t_sum = tgt_pts.sum(dim=1, keepdim=True).transpose(0, 1)          # (1, Gb)

        dice = (2.0 * inter + smooth) / (p_sum + t_sum + smooth)          # (N, Gb)
        dice_loss = 1.0 - dice                                            # (N, Gb)

        # Write back -> (L, C, Gb) -> (L, B, C, G)
        loss_l_c_gb = dice_loss.view(L, C, Gb)
        out[:, b, :, cols] = loss_l_c_gb

    return out * scale