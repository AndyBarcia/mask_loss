import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.utils.checkpoint import checkpoint

from ..sigmoid.pw_sigmoid_ce import pairwise_sigmoid_cross_entropy_loss_py
from ..dice.pw_dice_loss import pairwise_dice_loss_py
from ..label.pw_label_loss import pairwise_label_loss_py

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseMaskLossFunction(Function):
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
        mask_focal_gamma=None,
        mask_focal_alpha=None,
        cls_focal_gamma=None,
        cls_focal_alpha=None,
    ):
        L, B, C, h, w = mask_logits.shape
        B_t, H_t, W_t = mask_targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"

        mask_logits = mask_logits.contiguous().float()
        mask_targets = mask_targets.contiguous()
        cls_logits = cls_logits.contiguous().float()
        cls_targets = cls_targets.contiguous()
        mask_fg = 0.0 if mask_focal_gamma is None else float(mask_focal_gamma)
        if mask_fg < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if mask_focal_alpha is None:
            mask_fa = -1.0
        else:
            mask_fa = float(mask_focal_alpha)
            if not (0.0 <= mask_fa <= 1.0):
                raise ValueError("focal_alpha must be in [0, 1]")
        if cls_focal_gamma is None:
            cls_fg = mask_fg
        else:
            cls_fg = float(cls_focal_gamma)
            if cls_fg < 0.0:
                raise ValueError("focal_gamma must be non-negative")
        if cls_focal_alpha is None:
            cls_fa = mask_fa
        else:
            cls_fa = float(cls_focal_alpha)
            if not (0.0 <= cls_fa <= 1.0):
                raise ValueError("focal_alpha must be in [0, 1]")
        output = mask_loss.pairwise_mask_loss_forward(
            mask_logits,
            mask_targets,
            cls_logits,
            cls_targets,
            smooth if smooth is not None else 1.0,
            sigmoid_scale if sigmoid_scale is not None else 1.0,
            dice_scale if dice_scale is not None else 1.0,
            cls_scale if cls_scale is not None else 1.0,
            background_index if background_index is not None else -1,
            mask_fg,
            mask_fa,
            cls_fg,
            cls_fa,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None,) * 13


def pairwise_mask_loss_py(
    mask_logits,         # (L,B,Q,H,W)
    mask_targets,        # (B,H_t,W_t)
    cls_logits,          # (L,B,Q,C)
    cls_targets,         # (B,GT)
    smooth,
    sigmoid_scale   = 1.0,
    dice_scale      = 1.0,
    cls_scale       = 1.0,
    background_index= -1,
    uncertainty_gamma: float = 1.0,
    uncertainty_gamma_min: float = 0.05,
    mask_focal_gamma: float = 0.0,
    mask_focal_alpha: Optional[float] = None,
    cls_focal_gamma: Optional[float] = None,
    cls_focal_alpha: Optional[float] = None,
):
    if cls_focal_gamma is None:
        cls_fg = mask_focal_gamma
    else:
        cls_fg = cls_focal_gamma
    if cls_focal_alpha is None:
        cls_fa = mask_focal_alpha
    else:
        cls_fa = cls_focal_alpha
    sigmoid_cost = pairwise_sigmoid_cross_entropy_loss_py(
        mask_logits,
        mask_targets,
        background_index,
        sigmoid_scale,
        mask_focal_gamma,
        mask_focal_alpha,
        uncertainty_gamma,
        uncertainty_gamma_min,
    )  # (L,B,C,GT_out)
    dice_cost = pairwise_dice_loss_py(
        mask_logits,
        mask_targets,
        smooth,
        background_index,
        dice_scale,
        uncertainty_gamma,
        uncertainty_gamma_min,
    )  # (L,B,C,GT_out)
    cls_cost = pairwise_label_loss_py(
        cls_logits,
        cls_targets,
        background_index=background_index,
        scale=cls_scale,
        focal_gamma=cls_fg,
        focal_alpha=cls_fa,
    ) # (L,B,C,GT_out)
    return torch.stack([sigmoid_cost, dice_cost, cls_cost], dim=0)  # (3,L,B,C,GT_out)
