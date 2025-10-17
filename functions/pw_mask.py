import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

from .pw_sigmoid_ce import pairwise_sigmoid_cross_entropy_loss_py
from .pw_dice import pairwise_dice_loss_py

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseMaskLossFunction(Function):
    @staticmethod
    def forward(
        ctx, 
        logits, 
        targets,
        smooth,
        sigmoid_scale,
        dice_scale,
        background_index, 
    ):
        L, B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.pairwise_mask_loss_forward(
            logits, 
            targets,
            smooth if smooth is not None else 1.0,
            sigmoid_scale if sigmoid_scale is not None else 1.0,
            dice_scale if dice_scale is not None else 1.0,
            background_index if background_index is not None else -1,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None


def pairwise_mask_loss_py(
    logits,         # (L,B,C,H,W) CUDA
    targets,        # (B,H_t,W_t) CUDA
    smooth,
    sigmoid_scale   = 1.0,
    dice_scale      = 1.0,
    background_index= -1,
):
    sigmoid_cost = pairwise_sigmoid_cross_entropy_loss_py(
        logits, 
        targets, 
        background_index,
        sigmoid_scale, 
    )  # (L,B,C,GT_out)
    dice_cost = pairwise_dice_loss_py(
        logits, 
        targets, 
        smooth,
        background_index,
        dice_scale
    )  # (L,B,C,GT_out)
    return torch.stack([sigmoid_cost, dice_cost], dim=0)  # (2,L,B,C,GT_out)