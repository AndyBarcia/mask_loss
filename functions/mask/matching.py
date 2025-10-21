import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from .pw_mask import pairwise_mask_loss_py
from scipy.optimize import linear_sum_assignment

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class MaskMatchingFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        mask_logits,         # (L,B,Q,H,W)
        mask_targets,        # (B,H_t,W_t)
        cls_logits,          # (L,B,Q,C)
        cls_targets,         # (B,GT)
        smooth,
        sigmoid_scale,
        dice_scale,
        cls_scale,
        background_index, 
        inf_thresh,
        num_masks
    ):
        L, B, C, h, w = mask_logits.shape
        B_t, H_t, W_t = mask_targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        mask_logits = mask_logits.contiguous().float()
        mask_targets = mask_targets.contiguous()
        cls_logits = cls_logits.contiguous().float()
        cls_targets = cls_targets.contiguous()

        smooth_val = float(smooth if smooth is not None else 1.0)
        sigmoid_scale_val = float(sigmoid_scale if sigmoid_scale is not None else 1.0)
        dice_scale_val = float(dice_scale if dice_scale is not None else 1.0)
        cls_scale = float(cls_scale if cls_scale is not None else 1.0)
        background_index_val = int(background_index if background_index is not None else -1)
        inf_thresh_val = float(inf_thresh if inf_thresh is not None else 1e30)
        num_masks_val = int(num_masks if num_masks is not None else -1)

        matches, layer_mask_mean, layer_dice_mean, layer_cls_mean, matched = mask_loss.mask_matching(
            mask_logits,
            mask_targets,
            cls_logits,
            cls_targets,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale,
            background_index_val,
            inf_thresh_val,
            num_masks_val,
        )

        ctx.save_for_backward(
            mask_logits.detach(), 
            mask_targets.detach(),
            cls_logits.detach(),
            cls_targets.detach(),
            matches.detach()
        )
        ctx.logits_dtype = mask_logits.dtype
        ctx.smooth = smooth_val
        ctx.sigmoid_scale = sigmoid_scale_val
        ctx.dice_scale = dice_scale_val
        ctx.cls_scale = cls_scale
        ctx.background_index = background_index_val
        ctx.num_masks = num_masks_val
        ctx.matched = int(matched.item()) if matched.numel() > 0 else 0

        return matches, layer_mask_mean, layer_dice_mean, layer_cls_mean, matched

    @staticmethod
    def backward(ctx, grad_matches, grad_layer_mask_mean, grad_layer_dice_mean, grad_layer_cls_mean, grad_matched):
        (
            mask_logits, 
            mask_targets,
            cls_logits,
            cls_targets, 
            matches
        ) = ctx.saved_tensors
        logits_dtype = ctx.logits_dtype
        smooth = ctx.smooth
        sigmoid_scale = ctx.sigmoid_scale
        dice_scale = ctx.dice_scale
        cls_scale = ctx.cls_scale
        background_index = ctx.background_index
        num_masks = ctx.num_masks

        grad_layer_mask_mean = grad_layer_mask_mean.contiguous()
        grad_layer_dice_mean = grad_layer_dice_mean.contiguous()
        grad_layer_cls_mean = grad_layer_cls_mean.contiguous()

        grad_mask_logits, grad_cls_logits = mask_loss.mask_matching_backward(
            grad_layer_mask_mean,
            grad_layer_dice_mean,
            grad_layer_cls_mean,
            mask_logits,
            mask_targets,
            cls_logits,
            cls_targets,
            matches,
            smooth,
            sigmoid_scale,
            dice_scale,
            cls_scale,
            background_index,
            num_masks,
            ctx.matched,
        )

        return grad_mask_logits, None, grad_cls_logits, None, None, None, None, None, None, None, None

def mask_matching_py(
    mask_logits,         # (L,B,Q,H,W)
    mask_targets,        # (B,H_t,W_t)
    cls_logits,          # (L,B,Q,C)
    cls_targets,         # (B,GT)
    smooth,
    sigmoid_scale   = 1.0,
    dice_scale      = 1.0,
    cls_scale       = 1.0,
    background_index= -1,
    inf_thresh      = 1e30,
    num_masks       = None,
):
    L, B, C, H, W = mask_logits.shape

    # Pairwise costs: (3,L,B,C,GT_out)
    costs = pairwise_mask_loss_py(
        mask_logits,         # (L,B,Q,H,W)
        mask_targets,        # (B,H_t,W_t)
        cls_logits,          # (L,B,Q,C)
        cls_targets,         # (B,GT)
        smooth,
        sigmoid_scale,
        dice_scale,
        cls_scale,
        background_index,
    )
    sigmoid_cost = costs[0]
    dice_cost = costs[1]
    cls_cost = costs[2]
    costs = costs.sum(dim=0)  # (L,B,C,GT_out)

    # Output matches (L,B,GT_out), default -1 for invalid/ignored GTs
    matches = torch.full(
        (L, B, costs.shape[-1]),
        -1,
        dtype=torch.int64,
        device=mask_logits.device,
    )

    # Large finite fallback for masked pairs
    if math.isfinite(inf_thresh) and inf_thresh > 0:
        BIG = max(1e6, min(1e290, inf_thresh * 0.5))
    else:
        BIG = 1e15

    for l in range(L):
        for b in range(B):
            # C x GT
            cm = costs[l, b]  # stays on CUDA

            # Columns (GTs) that have at least one finite, < inf_thresh entry
            finite = torch.isfinite(cm) & (cm < inf_thresh)  # C x GT (bool)
            valid_cols = torch.nonzero(finite.any(dim=0), as_tuple=False).squeeze(1)  # (M,)
            M = int(valid_cols.numel())
            if M == 0:
                continue  # no valid GTs here

            if C < M:
                raise RuntimeError(
                    f"Hungarian requires #preds >= #valid GTs, got C={C}, M={M}"
                )

            # Build (C x M) submatrix with BIG where invalid
            sub = cm[:, valid_cols]  # C x M
            sub = torch.where(
                (torch.isfinite(sub) & (sub < inf_thresh)),
                sub,
                torch.as_tensor(BIG, dtype=sub.dtype, device=sub.device),
            )

            # Solve on (M x C): rows = valid GTs, cols = predictions
            sub_np = sub.transpose(0, 1).to(torch.float64).detach().cpu().numpy()  # (M,C)
            row_ind, col_ind = linear_sum_assignment(sub_np, maximize=False)  # len = M

            # Write back: GT_out index -> predicted row index
            # row_ind indexes valid GT rows (0..M-1) -> map back to original GT column ids
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                gt_col = int(valid_cols[r].item())  # original GT column
                pred_row = int(c)                   # prediction row (0..C-1)
                matches[l, b, gt_col] = pred_row

    # Aggregate losses using assignments
    assigned = matches.ge(0)                          # (L,B,GT)
    matched = int(assigned.sum().item())              # local matched GTs

    layer_mask_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_dice_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_cls_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)

    if matched > 0:
        idx = assigned.nonzero(as_tuple=False)        # (N,3): [l,b,g]
        l_idx = idx[:, 0]
        b_idx = idx[:, 1]
        g_idx = idx[:, 2]
        p_idx = matches[assigned].to(torch.long)      # (N,)

        sig_vals  = sigmoid_cost[l_idx, b_idx, p_idx, g_idx]
        dice_vals = dice_cost[l_idx, b_idx, p_idx, g_idx]
        cls_vals = cls_cost[l_idx, b_idx, p_idx, g_idx]

        layer_mask_sum.index_add_(0, l_idx, sig_vals)
        layer_dice_sum.index_add_(0, l_idx, dice_vals)
        layer_cls_sum.index_add_(0, l_idx, cls_vals)

    denom = float(num_masks) if (num_masks is not None and num_masks > 0) else float(matched)
    if denom <= 0:
        denom = 1.0

    layer_mask_mean = layer_mask_sum / denom          # (L,)
    layer_dice_mean = layer_dice_sum / denom          # (L,)
    layer_cls_mean = layer_cls_sum / denom            # (L,)

    # Return exactly like the C++ ext now does
    matched_tensor = torch.tensor(matched, device=mask_logits.device, dtype=torch.long)
    return matches, layer_mask_mean, layer_dice_mean, layer_cls_mean, matched_tensor

def mask_matching_sampling_py(
    mask_logits,         # (L,B,Q,H,W)
    mask_targets,        # (B,H_t,W_t)
    cls_logits,          # (L,B,Q,C)
    cls_targets,         # (B,GT)
    smooth,
    sigmoid_scale   = 1.0,
    dice_scale      = 1.0,
    cls_scale       = 1.0,
    background_index= -1,
    inf_thresh      = 1e30,
    num_masks       = None,
):
    L, B, C, H, W = mask_logits.shape

    # Pairwise costs: (3,L,B,C,GT_out)
    costs = pairwise_mask_loss_py(
        mask_logits,         # (L,B,Q,H,W)
        mask_targets,        # (B,H_t,W_t)
        cls_logits,          # (L,B,Q,C)
        cls_targets,         # (B,GT)
        smooth,
        sigmoid_scale,
        dice_scale,
        cls_scale,
        background_index,
    )
    sigmoid_cost = costs[0]
    dice_cost = costs[2]
    cls_cost = costs[3]
    costs = costs.sum(dim=0)  # (L,B,C,GT_out)

    # Output matches (L,B,GT_out), default -1 for invalid/ignored GTs
    matches = torch.full(
        (L, B, costs.shape[-1]),
        -1,
        dtype=torch.int64,
        device=mask_logits.device,
    )

    # Large finite fallback for masked pairs
    if math.isfinite(inf_thresh) and inf_thresh > 0:
        BIG = max(1e6, min(1e290, inf_thresh * 0.5))
    else:
        BIG = 1e15

    for l in range(L):
        for b in range(B):
            # C x GT
            cm = costs[l, b]  # stays on CUDA

            # Columns (GTs) that have at least one finite, < inf_thresh entry
            finite = torch.isfinite(cm) & (cm < inf_thresh)  # C x GT (bool)
            valid_cols = torch.nonzero(finite.any(dim=0), as_tuple=False).squeeze(1)  # (M,)
            M = int(valid_cols.numel())
            if M == 0:
                continue  # no valid GTs here

            if C < M:
                raise RuntimeError(
                    f"Hungarian requires #preds >= #valid GTs, got C={C}, M={M}"
                )

            # Build (C x M) submatrix with BIG where invalid
            sub = cm[:, valid_cols]  # C x M
            sub = torch.where(
                (torch.isfinite(sub) & (sub < inf_thresh)),
                sub,
                torch.as_tensor(BIG, dtype=sub.dtype, device=sub.device),
            )

            # Solve on (M x C): rows = valid GTs, cols = predictions
            sub_np = sub.transpose(0, 1).to(torch.float64).detach().cpu().numpy()  # (M,C)
            row_ind, col_ind = linear_sum_assignment(sub_np, maximize=False)  # len = M

            # Write back: GT_out index -> predicted row index
            # row_ind indexes valid GT rows (0..M-1) -> map back to original GT column ids
            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                gt_col = int(valid_cols[r].item())  # original GT column
                pred_row = int(c)                   # prediction row (0..C-1)
                matches[l, b, gt_col] = pred_row

    # Aggregate losses using assignments
    assigned = matches.ge(0)                          # (L,B,GT)
    matched = int(assigned.sum().item())              # local matched GTs

    layer_mask_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_dice_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)

    if matched > 0:
        idx = assigned.nonzero(as_tuple=False)        # (N,3): [l,b,g]
        l_idx = idx[:, 0]
        b_idx = idx[:, 1]
        g_idx = idx[:, 2]
        p_idx = matches[assigned].to(torch.long)      # (N,)

        sig_vals  = sigmoid_cost[l_idx, b_idx, p_idx, g_idx]
        dice_vals = dice_cost[l_idx, b_idx, p_idx, g_idx]

        layer_mask_sum.index_add_(0, l_idx, sig_vals)
        layer_dice_sum.index_add_(0, l_idx, dice_vals)

    denom = float(num_masks) if (num_masks is not None and num_masks > 0) else float(matched)
    if denom <= 0:
        denom = 1.0

    layer_mask_mean = layer_mask_sum / denom          # (L,)
    layer_dice_mean = layer_dice_sum / denom          # (L,)
    matched_tensor = torch.tensor(matched, device=mask_logits.device, dtype=torch.long)
    return matches, layer_mask_mean, layer_dice_mean, matched_tensor
