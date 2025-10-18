import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from .pw_sigmoid_ce import pairwise_sigmoid_cross_entropy_loss_py
from .pw_dice import pairwise_dice_loss_py

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
        logits, 
        targets,
        smooth,
        sigmoid_scale,
        dice_scale,
        background_index, 
        inf_thresh
    ):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.mask_matching(
            logits, 
            targets,
            smooth if smooth is not None else 1.0,
            sigmoid_scale if sigmoid_scale is not None else 1.0,
            dice_scale if dice_scale is not None else 1.0,
            background_index if background_index is not None else -1,
            inf_thresh if inf_thresh is not None else 1e30
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None

def mask_matching(
    logits,         # (L,B,C,H,W) CUDA
    targets,        # (B,H_t,W_t) CUDA
    smooth,
    sigmoid_scale   = 1.0,
    dice_scale      = 1.0,
    background_index= -1,
    inf_thresh      = 1e30
):
    L, B, C, H, W = logits.shape

    # Pairwise costs: (L,B,C,GT_out)
    sigmoid_cost = pairwise_sigmoid_cross_entropy_loss_py(
        logits, targets, background_index, sigmoid_scale
    )
    dice_cost = pairwise_dice_loss_py(
        logits, targets, smooth, background_index, dice_scale
    )
    costs = sigmoid_cost + dice_cost  # (L,B,C,GT_out)

    # Output matches (L,B,GT_out), default -1 for invalid/ignored GTs
    matches = torch.full(
        (L, B, costs.shape[-1]),
        -1,
        dtype=torch.int64,
        device=logits.device,
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

    return matches  # (L,B,GT_out)
