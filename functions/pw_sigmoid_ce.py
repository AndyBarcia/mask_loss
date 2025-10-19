import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from .utils import point_sample, counts_per_cell_per_class

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

    counts_all = counts_per_cell_per_class(targets_long, h, w, s, GT_all)  # (B, J, GT_all)

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

def pairwise_sigmoid_cross_entropy_loss_sampling_py(
    logits: torch.Tensor,
    targets: torch.Tensor,
    background_index: int = None,
    scale: float = 1.0,
    num_points: int = 5000,
    align_corners: bool = False,
):
    """
    Pairwise sigmoid CE using *point sampling* (like Mask2Former matching).

    We evaluate BCE-with-logits between every predicted class map and each *ground-truth class*
    using a shared set of random points per image — no full upsampling, no full one-hot GT.

    Args:
        logits:  (L, B, C, h, w)  predicted logits
        targets: (B, H_t, W_t)    integer label map per image
        background_index: int or None
        scale:   multiply the final loss
        num_points: number of random points per image to sample (default: 5000)
        align_corners: forwarded to `point_sample` for the prediction sampling

    Returns:
        (L, B, C, G) where G is the number of evaluated ground-truth classes
        (optionally excluding background). Entries are +inf where a class is
        absent in that image.
    """
    assert logits.dim() == 5, "logits must have shape (L, B, C, h, w)"
    L, B, C, h, w = logits.shape
    Bt, Ht, Wt = targets.shape
    assert B == Bt, "Batch size mismatch between logits and targets"
    assert num_points > 0, "num_points must be > 0"

    device = logits.device
    dtype  = logits.dtype

    # Determine global set of GT classes to report (consistent last dim across batch)
    targets_long = targets.to(device=device, dtype=torch.long, non_blocking=True)
    gt_max = int(targets_long.max().item()) if targets_long.numel() else -1
    GT_all = gt_max + 1

    if GT_all <= 0:
        return logits.new_empty((L, B, C, 0))

    if background_index is None:
        gt_classes: List[int] = list(range(GT_all))
    else:
        gt_classes = [k for k in range(GT_all) if k != background_index]

    G = len(gt_classes)
    if G == 0:
        return logits.new_empty((L, B, C, 0))

    # Map real class id -> column in the output tensor
    id2col = {cls_id: i for i, cls_id in enumerate(gt_classes)}

    # Output initialized to +inf; we overwrite positions for present classes
    out = torch.full((L, B, C, G), float('inf'), device=device, dtype=dtype)

    # Work per image for locality and to share a single point set per image (like matcher)
    for b in range(B):
        labels_b = targets_long[b]  # (Ht, Wt)

        # Which of the global classes appear in this image?
        present_ids = torch.unique(labels_b).tolist()
        if background_index is not None:
            present_ids = [k for k in present_ids if k != background_index]
        if len(present_ids) == 0:
            # nothing to write for this image; keep +inf
            continue

        # Indices in the global G dimension we will write to for this image
        cols = torch.tensor([id2col[k] for k in present_ids], device=device, dtype=torch.long)
        Gb = len(present_ids)

        # Sample a shared set of random points in [0, 1] x [0, 1] ----
        coords = torch.rand(1, num_points, 2, device=device)  # (1, P, 2), shared across all maps for this image

        # Sample predictions at those points 
        # Group (L, C) as batch to avoid class loops: N = L*C
        z_b = logits[:, b]                              # (L, C, h, w)
        z_flat = z_b.reshape(L * C, 1, h, w)            # (N, 1, h, w)
        coords_rep = coords.repeat(z_flat.shape[0], 1, 1)  # (N, P, 2)
        # (N, 1, P) -> (N, P)
        pred_pts = point_sample(z_flat, coords_rep, align_corners=align_corners).squeeze(1)

        # Convert [0,1] coords to integer pixel indices (nearest); clamp to valid range.
        # This keeps memory low and avoids building (Gb, H, W) binary stacks.
        # Note: indices follow a simple floor mapping which is a good approximation for point sampling.
        xy = coords[0]                                   # (P, 2), x in [:,0], y in [:,1]
        ix = torch.clamp((xy[:, 0] * (Wt - 1)).long(), 0, Wt - 1)
        iy = torch.clamp((xy[:, 1] * (Ht - 1)).long(), 0, Ht - 1)
        labels_pts = labels_b[iy, ix]                    # (P,)

        # For just the classes present in this image, build binary labels at points: (Gb, P)
        class_ids_b = torch.tensor(present_ids, device=device, dtype=torch.long)
        tgt_pts = (labels_pts[None, :] == class_ids_b[:, None]).to(dtype)  # (Gb, P)

        # BCE-with-logits via einsum
        # pred_pts: (N, P) with N=L*C
        pos = F.binary_cross_entropy_with_logits(pred_pts, torch.ones_like(pred_pts), reduction="none")  # (N, P)
        neg = F.binary_cross_entropy_with_logits(pred_pts, torch.zeros_like(pred_pts), reduction="none")  # (N, P)

        # loss = ⟨pos, tgt⟩ + ⟨neg, 1 - tgt⟩ over the points dimension
        term_pos = torch.einsum("np,mp->nm", pos, tgt_pts)                      # (N, Gb)
        term_neg = torch.einsum("np,mp->nm", neg, (1.0 - tgt_pts))              # (N, Gb)
        loss_lc_gb = (term_pos + term_neg) / float(num_points)                  # (N, Gb)
        loss_l_c_gb = loss_lc_gb.view(L, C, Gb)                                 # (L, C, Gb)

        out[:, b, :, cols] = loss_l_c_gb

    return out * scale