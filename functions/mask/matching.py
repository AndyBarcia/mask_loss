import math
from typing import Optional

import numpy as np
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
    """Autograd bridge for the CUDA/C++ hybrid matching kernels.

    The ``forward`` and ``backward`` implementations are thin wrappers over the
    compiled extension.  They perform basic validation / defaulting of the
    Python arguments and expose a clean ``torch.autograd.Function`` interface.

    ``MaskMatchingFunction`` is intentionally minimal so that all behavioural
    documentation (matching rounds, loss aggregation, etc.) can live alongside
    the reference Python implementation below.  Still, we document every input
    to make it easy for readers to map the signature to the kernels.
    """
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
        num_masks,
        force_unmatched_class_to_background,
        force_unmatched_masks_to_empty,
        K,
        assignment_strategy,
        focal_gamma,
        focal_alpha,
        void_class_index=None,
    ):
        """Run the forward pass of the matching op.

        Args:
            ctx: Autograd context used to stash tensors for the backward pass.
            mask_logits (Tensor): Raw mask predictions with shape ``(L, B, Q, H, W)``.
                ``L`` is the number of decoder layers, ``B`` the batch size,
                ``Q`` the number of queries and ``H x W`` the mask resolution.
            mask_targets (Tensor): Integer tensor ``(B, H_t, W_t)`` containing
                per-pixel ground-truth instance identifiers.
            cls_logits (Tensor): Classification logits ``(L, B, Q, C)`` where
                ``C`` is the number of categories (including background).
            cls_targets (Tensor): Ground-truth class labels ``(B, GT)`` where
                ``GT`` is the maximum number of instances per image.
            smooth (float): Additive smoothing constant used by the dice loss.
            sigmoid_scale (float): Scalar multiplier for the sigmoid/BCE cost.
            dice_scale (float): Scalar multiplier applied to the dice cost.
            cls_scale (float): Scalar multiplier applied to the class cost.
            background_index (int): Index in ``cls_targets`` that represents
                the background class. ``-1`` disables background forcing.
            inf_thresh (float): Threshold above which costs are treated as
                ``+inf`` and therefore ignored by the assignment step.
            num_masks (float): Optional normalisation denominator for the mask
                and dice losses. ``-1`` or ``None`` fall back to the number of
                (matched) elements.
            force_unmatched_class_to_background (bool): Whether unmatched
                predictions should contribute a background classification loss.
            force_unmatched_masks_to_empty (bool): Whether unmatched predictions
                should be trained towards an empty mask.
            void_class_index (Optional[int]): Optional index of the "void"
                class in ``cls_logits``. When provided, unmatched predictions
                are supervised to activate this channel regardless of the
                background enforcement flag.
            K (int): Maximum number of detections that can be matched to the
                same ground truth (hybrid matching "top-k" budget).
            assignment_strategy (str): Name of the matching strategy.  One of
                ``{"global", "round", "greedy", "pseudo_greedy"}``.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ``pred_to_gt``
                assignments, ``pred_round`` rank indices, and the per-layer
                mask, dice, and classification loss means.
        """
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
        num_masks_val = float(num_masks if num_masks is not None else -1.0)
        force_unmatched_cls = bool(
            force_unmatched_class_to_background if force_unmatched_class_to_background is not None else False
        )
        force_unmatched_masks = bool(
            force_unmatched_masks_to_empty if force_unmatched_masks_to_empty is not None else False
        )

        assignment_strategy = assignment_strategy if assignment_strategy is not None else "global"

        strategy_map = {
            "global": 0,
            "round": 1,
            "greedy": 2,
            "pseudo_greedy": 3,
        }

        if assignment_strategy not in strategy_map:
            raise ValueError(
                f"Unknown assignment_strategy '{assignment_strategy}'."
                f" Expected one of {sorted(strategy_map.keys())}"
            )

        K_val = int(K if K is not None else 1)
        if K_val < 0:
            raise ValueError("K must be non-negative")

        focal_gamma_val = 0.0 if focal_gamma is None else float(focal_gamma)
        if focal_gamma_val < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if focal_alpha is None:
            focal_alpha_val = -1.0
        else:
            focal_alpha_val = float(focal_alpha)
            if not (0.0 <= focal_alpha_val <= 1.0):
                raise ValueError("focal_alpha must be in [0, 1]")

        num_cls_channels = cls_logits.shape[-1]
        if void_class_index is None:
            void_index_val = -1
        else:
            void_index_val = int(void_class_index)
            if not (0 <= void_index_val < num_cls_channels):
                raise ValueError("incorrect void index")

        if void_index_val != -1 and torch.any(cls_targets == void_index_val):
            raise ValueError("cls_targets must not contain the void class index")

        pred_to_gt, pred_round, layer_mask_mean, layer_dice_mean, layer_cls_mean = mask_loss.mask_matching(
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
            force_unmatched_cls,
            force_unmatched_masks,
            K_val,
            strategy_map[assignment_strategy],
            focal_gamma_val,
            focal_alpha_val,
            void_index_val,
        )

        ctx.save_for_backward(
            mask_logits.detach(),
            mask_targets.detach(),
            cls_logits.detach(),
            cls_targets.detach(),
            pred_to_gt.detach(),
        )
        ctx.logits_dtype = mask_logits.dtype
        ctx.smooth = smooth_val
        ctx.sigmoid_scale = sigmoid_scale_val
        ctx.dice_scale = dice_scale_val
        ctx.cls_scale = cls_scale
        ctx.background_index = background_index_val
        ctx.num_masks = num_masks_val
        ctx.force_unmatched_cls = force_unmatched_cls
        ctx.force_unmatched_masks = force_unmatched_masks
        ctx.focal_gamma = focal_gamma_val
        ctx.focal_alpha = focal_alpha_val
        ctx.void_class_index = void_index_val

        return pred_to_gt, pred_round, layer_mask_mean, layer_dice_mean, layer_cls_mean

    @staticmethod
    def backward(ctx, _, __, grad_layer_mask_mean, grad_layer_dice_mean, grad_layer_cls_mean):
        """Backpropagate the gradients through the CUDA extension."""
        (
            mask_logits,
            mask_targets,
            cls_logits,
            cls_targets,
            pred_to_gt
        ) = ctx.saved_tensors
        smooth = ctx.smooth
        sigmoid_scale = ctx.sigmoid_scale
        dice_scale = ctx.dice_scale
        cls_scale = ctx.cls_scale
        background_index = ctx.background_index
        num_masks = ctx.num_masks
        force_unmatched_cls = ctx.force_unmatched_cls
        force_unmatched_masks = ctx.force_unmatched_masks
        focal_gamma = ctx.focal_gamma
        focal_alpha = ctx.focal_alpha

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
            pred_to_gt,
            smooth,
            sigmoid_scale,
            dice_scale,
            cls_scale,
            background_index,
            num_masks,
            force_unmatched_cls,
            force_unmatched_masks,
            focal_gamma,
            focal_alpha,
            ctx.void_class_index,
        )

        return (
            grad_mask_logits,
            None,
            grad_cls_logits,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

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
    force_unmatched_class_to_background=False,
    force_unmatched_masks_to_empty=False,
    K=1,
    assignment_strategy="global",
    focal_gamma: float = 0.0,
    focal_alpha: Optional[float] = None,
    void_class_index: Optional[int] = None,
):
    """Reference Python implementation of :func:`mask_matching`.

    This function mirrors the CUDA extension in pure Python/Numpy so it can be
    used for debugging or in environments where the extension is not available.

    Args:
        mask_logits (Tensor): Decoder mask logits ``(L, B, Q, H, W)``.
        mask_targets (Tensor): Ground-truth mask indices ``(B, H_t, W_t)``.
        cls_logits (Tensor): Decoder class logits ``(L, B, Q, C)``.
        cls_targets (Tensor): Ground-truth class labels ``(B, GT)``.
        smooth (float): Dice smoothing constant.
        sigmoid_scale (float): Weight applied to the sigmoid/BCE cost.
        dice_scale (float): Weight applied to the dice cost.
        cls_scale (float): Weight applied to the classification cost.
        background_index (int): Background class index or ``-1`` to disable.
        inf_thresh (float): Costs equal/above this value are ignored.
        num_masks (Optional[float]): Optional denominator for loss averaging.
        force_unmatched_class_to_background (bool): If ``True`` enforce
            background classification for unmatched predictions.
        force_unmatched_masks_to_empty (bool): If ``True`` supervise unmatched
            predictions towards empty masks.
        K (int): Maximum number of detections per ground truth.
        assignment_strategy (str): Strategy identifier (see below).
        focal_gamma (float): Focal loss exponent applied to BCE terms.
        focal_alpha (Optional[float]): Positive-class prior in ``[0, 1]``. Use
            ``None`` to disable class re-weighting.
        void_class_index (Optional[int]): Index of the "void" class in
            ``cls_logits``. If set, unmatched detections are trained to predict
            the void class regardless of ``force_unmatched_class_to_background``.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: The same values exposed
        by the CUDA extension: ``pred_to_gt``, ``pred_round``, and the per-layer
        averaged losses.
    """
    L, B, C, H, W = mask_logits.shape
    inf_thresh_val = float(inf_thresh if inf_thresh is not None else 1e30)
    K_val = int(K if K is not None else 1)
    if K_val < 0:
        raise ValueError("K must be non-negative")

    focal_gamma_val = 0.0 if focal_gamma is None else float(focal_gamma)
    if focal_gamma_val < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    if focal_alpha is None:
        focal_alpha_val: Optional[float] = None
        alpha_neg = 1.0
        alpha_pos = 1.0
    else:
        focal_alpha_val = float(focal_alpha)
        if not (0.0 <= focal_alpha_val <= 1.0):
            raise ValueError("focal_alpha must be in [0, 1]")
        alpha_neg = 1.0 - focal_alpha_val
        alpha_pos = focal_alpha_val
    use_gamma = focal_gamma_val != 0.0

    num_cls_channels = cls_logits.shape[-1]
    void_idx: Optional[int]
    if void_class_index is None:
        void_idx = None
    else:
        vi = int(void_class_index)
        if not (0 <= vi < num_cls_channels):
            raise ValueError("incorrect void index")
        void_idx = vi

    if void_idx is not None and torch.any(cls_targets == void_idx):
        raise ValueError("cls_targets must not contain the void class index")
    has_void_class = void_idx is not None

    strategy_map = {"global", "round", "greedy", "pseudo_greedy"}
    if assignment_strategy not in strategy_map:
        raise ValueError(
            f"Unknown assignment_strategy '{assignment_strategy}'."
            f" Expected one of {sorted(strategy_map)}"
        )

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
        focal_gamma=focal_gamma_val,
        focal_alpha=focal_alpha_val,
    )
    sigmoid_cost = costs[0]
    dice_cost = costs[1]
    cls_cost = costs[2]
    costs = costs.sum(dim=0)  # (L,B,Q,GT_out)

    pred_to_gt = torch.full(
        (L, B, costs.shape[-2]),
        -1,
        dtype=torch.int64,
        device=mask_logits.device,
    ) # (L,B,Q)
    pred_round = torch.full_like(pred_to_gt, -1)

    def _big_from(threshold: float) -> float:
        """Return a large finite number used when masking invalid costs."""
        if not math.isfinite(threshold) or threshold <= 0.0:
            return 1e15
        big = threshold * 0.5
        if not math.isfinite(big) or big < 1e6:
            big = 1e15
        if big > 1e290:
            big = 1e290
        return big

    def _assign_predictions_numpy(cost_np: np.ndarray):
        """Solve the matching problem on a ``(Q, GT_out)`` cost matrix.

        Args:
            cost_np (np.ndarray): Dense cost matrix for a single layer/batch
                slice, already transferred to host memory.

        Returns:
            Tuple[np.ndarray, np.ndarray]: ``pred_to_gt`` and ``pred_round``
                arrays encoding the best assignments and round indices.
        """
        Q, GT_out = cost_np.shape
        pred_to_gt_np = np.full(Q, -1, dtype=np.int64)
        pred_round_np = np.full(Q, -1, dtype=np.int64)
        if Q == 0 or GT_out == 0 or K_val == 0:
            return pred_to_gt_np, pred_round_np

        valid_cols = [
            gt for gt in range(GT_out)
            if np.any(np.isfinite(cost_np[:, gt]) & (cost_np[:, gt] < inf_thresh_val))
        ]
        if not valid_cols:
            return pred_to_gt_np, pred_round_np

        BIG = _big_from(inf_thresh_val)

        def hungarian(columns, preds):
            """Execute the Hungarian algorithm on a sub-problem."""
            if not columns or not preds:
                return {}
            sub = np.full((len(columns), len(preds)), BIG, dtype=np.float64)
            for ri, actual in enumerate(columns):
                vals = cost_np[:, actual]
                for ci, q in enumerate(preds):
                    v = vals[q]
                    if np.isfinite(v) and v < inf_thresh_val:
                        sub[ri, ci] = v
            row_ind, col_ind = linear_sum_assignment(sub)
            matches = {}
            for r, c in zip(row_ind, col_ind):
                if sub[r, c] >= BIG:
                    continue
                matches[preds[c]] = columns[r]
            return matches

        if assignment_strategy == "global":
            # Duplicate each valid ground-truth column up to K times and run
            # a *single* Hungarian assignment.  The duplication effectively
            # exposes K independent "slots" per ground truth so that the
            # solver can pick the K lowest-cost detections for every GT in one
            # pass while still respecting the global optimality property.
            columns = []
            for rep in range(K_val):
                if len(columns) >= Q:
                    break
                for actual in valid_cols:
                    columns.append(actual)
                    if len(columns) >= Q:
                        break
            matches = hungarian(columns, list(range(Q)))
            for q, actual in matches.items():
                pred_to_gt_np[q] = actual

        elif assignment_strategy == "round":
            # Run K independent Hungarian rounds.  Each round exposes every
            # ground truth at most once, consumes the detections selected by
            # the solver, and therefore emulates the classic DETR matching
            # schedule where matches from earlier rounds cannot be reused
            # later on.
            capacities = {gt: K_val for gt in valid_cols}
            remaining = list(range(Q))
            for _ in range(K_val):
                active = [gt for gt in valid_cols if capacities[gt] > 0]
                if not active or not remaining:
                    break
                if len(active) > len(remaining):
                    active = active[:len(remaining)]
                matches = hungarian(active, remaining)
                matched = set()
                for q, actual in matches.items():
                    if capacities[actual] <= 0:
                        continue
                    pred_to_gt_np[q] = actual
                    capacities[actual] -= 1
                    matched.add(q)
                remaining = [q for q in remaining if q not in matched]

        elif assignment_strategy == "greedy":
            # Assign each detection to the valid ground truth with the
            # currently-lowest cost.  The running capacities ensure that no GT
            # collects more than K detections.  This is intentionally simple
            # and mirrors the greedy approaches used in several DETR variants.
            capacities = {gt: K_val for gt in valid_cols}
            for q in range(Q):
                best_gt = -1
                best_cost = math.inf
                for actual in valid_cols:
                    if capacities[actual] <= 0:
                        continue
                    v = cost_np[q, actual]
                    if np.isfinite(v) and v < inf_thresh_val and v < best_cost:
                        best_cost = v
                        best_gt = actual
                if best_gt >= 0:
                    pred_to_gt_np[q] = best_gt
                    capacities[best_gt] -= 1

        elif assignment_strategy == "pseudo_greedy":
            # Run one Hungarian assignment to secure the globally best set of
            # matches (subject to the single-slot constraint) and then greedily
            # assign the remaining detections.  This hybrid strategy keeps a
            # strong first round while still allowing cheap assignments for the
            # tail detections.
            capacities = {gt: K_val for gt in valid_cols}
            base_cols = valid_cols[:min(len(valid_cols), Q)]
            matches = hungarian(base_cols, list(range(Q)))
            for q, actual in matches.items():
                if capacities[actual] <= 0:
                    continue
                pred_to_gt_np[q] = actual
                capacities[actual] -= 1

            for q in range(Q):
                if pred_to_gt_np[q] >= 0:
                    continue
                best_gt = -1
                best_cost = math.inf
                for actual in valid_cols:
                    if capacities[actual] <= 0:
                        continue
                    v = cost_np[q, actual]
                    if np.isfinite(v) and v < inf_thresh_val and v < best_cost:
                        best_cost = v
                        best_gt = actual
                if best_gt >= 0:
                    pred_to_gt_np[q] = best_gt
                    capacities[best_gt] -= 1

        for actual in valid_cols:
            # Sort detections assigned to each GT by their final matching cost
            # so we can report the round index (i.e. rank) of every match.
            assigned = [
                q for q in range(Q)
                if pred_to_gt_np[q] == actual
                and np.isfinite(cost_np[q, actual])
                and cost_np[q, actual] < inf_thresh_val
            ]
            assigned.sort(key=lambda q: (cost_np[q, actual], q))
            for round_idx, q in enumerate(assigned):
                if round_idx >= K_val:
                    break
                pred_round_np[q] = round_idx

        return pred_to_gt_np, pred_round_np

    for l in range(L):
        for b in range(B):
            # C x GT
            cm = costs[l, b]  # stays on CUDA
            assign_gt_np, assign_round_np = _assign_predictions_numpy(
                cm.detach().cpu().to(torch.float64).numpy()
            )
            assign_gt = torch.from_numpy(assign_gt_np).to(pred_to_gt.device)
            assign_round = torch.from_numpy(assign_round_np).to(pred_round.device)
            pred_to_gt[l, b] = assign_gt
            pred_round[l, b] = assign_round

    # Aggregate losses using assignments
    assigned = pred_to_gt.ge(0) # (L,B,Q)
    matched_dts = int(assigned.sum().item()) # local matched detections

    layer_mask_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_dice_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_cls_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)

    if matched_dts > 0:
        idx = assigned.nonzero(as_tuple=False) # (N,3): [l,b,q]
        l_idx = idx[:, 0]
        b_idx = idx[:, 1]
        p_idx = idx[:, 2]
        g_idx = pred_to_gt[assigned].to(torch.long) # (N,)

        sig_vals  = sigmoid_cost[l_idx, b_idx, p_idx, g_idx]
        dice_vals = dice_cost[l_idx, b_idx, p_idx, g_idx]
        cls_vals = cls_cost[l_idx, b_idx, p_idx, g_idx]

        layer_mask_sum.index_add_(0, l_idx, sig_vals)
        layer_dice_sum.index_add_(0, l_idx, dice_vals)
        layer_cls_sum.index_add_(0, l_idx, cls_vals)

    total_queries = L * B * C
    unmatched_dts = total_queries - matched_dts

    if (force_unmatched_masks_to_empty or force_unmatched_class_to_background) and total_queries > 0:
        assigned_pred = torch.zeros((L, B, C), device=mask_logits.device, dtype=torch.bool)
        if matched_dts > 0:
            assigned_pred[l_idx, b_idx, p_idx] = True
        unmatched_mask = (~assigned_pred).to(mask_logits.dtype)

        if force_unmatched_masks_to_empty and unmatched_dts > 0:
            logits = mask_logits
            probs = logits.sigmoid()
            ce_neg = F.softplus(logits)
            if use_gamma:
                mod_neg = probs.pow(focal_gamma_val)
            else:
                mod_neg = 1.0
            neg_term = ce_neg * mod_neg
            if focal_alpha_val is not None:
                neg_term = neg_term * alpha_neg
            mask_loss_per = neg_term.mean(dim=(-1, -2)) * sigmoid_scale
            layer_mask_sum += (mask_loss_per * unmatched_mask).sum(dim=(1, 2))

            probs = mask_logits.sigmoid()
            H_t, W_t = mask_targets.shape[1:]
            scale_h = H_t // H
            scale_w = W_t // W
            area_scale = float(scale_h * scale_w)
            p_sum_up = probs.sum(dim=(-1, -2)) * area_scale
            dice_loss = dice_scale * (p_sum_up / (p_sum_up + smooth))
            layer_dice_sum += (dice_loss * unmatched_mask).sum(dim=(1, 2))

        if unmatched_dts > 0:
            logits = cls_logits
            probs = logits.sigmoid()
            ce_neg = F.softplus(logits)
            ce_pos = F.softplus(-logits)
            if use_gamma:
                mod_neg = probs.pow(focal_gamma_val)
                mod_pos = (1.0 - probs).pow(focal_gamma_val)
            else:
                mod_neg = 1.0
                mod_pos = 1.0
            neg_term = ce_neg * mod_neg
            pos_term = ce_pos * mod_pos
            if focal_alpha_val is not None:
                neg_term = neg_term * alpha_neg
                pos_term = pos_term * alpha_pos

            cls_loss = None
            if force_unmatched_class_to_background:
                cls_loss = neg_term.mean(dim=-1)
                if has_void_class and num_cls_channels > 0:
                    void_neg = neg_term[..., void_idx]
                    void_pos = pos_term[..., void_idx]
                    cls_loss = cls_loss + (void_pos - void_neg) / float(num_cls_channels)
            elif has_void_class:
                cls_loss = pos_term[..., void_idx]

            if cls_loss is not None:
                cls_loss = cls_loss * cls_scale
                layer_cls_sum += (cls_loss * unmatched_mask).sum(dim=(1, 2))

    per_layer_matched = assigned.sum(dim=(1, 2)).to(layer_mask_sum.dtype)
    queries_per_layer = layer_mask_sum.new_full((L,), float(B * C))

    if force_unmatched_masks_to_empty:
        mask_denom = queries_per_layer
    elif num_masks is not None and num_masks > 0:
        mask_denom = layer_mask_sum.new_full((L,), float(num_masks))
    else:
        mask_denom = per_layer_matched

    unmatched_per_layer = queries_per_layer - per_layer_matched
    if force_unmatched_class_to_background:
        cls_denom = queries_per_layer
    elif has_void_class and unmatched_dts > 0:
        cls_denom = per_layer_matched + unmatched_per_layer
    elif num_masks is not None and num_masks > 0:
        cls_denom = layer_mask_sum.new_full((L,), float(num_masks))
    else:
        cls_denom = per_layer_matched

    mask_denom = torch.clamp(mask_denom, min=1.0)
    cls_denom = torch.clamp(cls_denom, min=1.0)

    layer_mask_mean = layer_mask_sum / mask_denom  # (L,)
    layer_dice_mean = layer_dice_sum / mask_denom  # (L,)
    layer_cls_mean = layer_cls_sum / cls_denom     # (L,)

    return pred_to_gt, pred_round, layer_mask_mean, layer_dice_mean, layer_cls_mean

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
    dice_cost = costs[1]
    cls_cost = costs[2]
    costs = costs.sum(dim=0)  # (L,B,C,GT_out)

    # Output gt_to_pred (L,B,GT_out), default -1 for invalid/ignored GTs
    gt_to_pred = torch.full(
        (L, B, costs.shape[-1]),
        -1,
        dtype=torch.int64,
        device=mask_logits.device,
    ) # (L,B,GT_out)
    pred_to_gt = torch.full(
        (L, B, costs.shape[-2]),
        -1,
        dtype=torch.int64,
        device=mask_logits.device,
    ) # (L,B,Q)

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
                gt_to_pred[l, b, gt_col] = pred_row
                pred_to_gt[l, b, pred_row] = gt_col

    # Aggregate losses using assignments
    assigned = gt_to_pred.ge(0)                          # (L,B,GT)
    matched = int(assigned.sum().item())              # local matched GTs

    layer_mask_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)
    layer_dice_sum = torch.zeros(L, device=mask_logits.device, dtype=mask_logits.dtype)

    if matched > 0:
        idx = assigned.nonzero(as_tuple=False)        # (N,3): [l,b,g]
        l_idx = idx[:, 0]
        b_idx = idx[:, 1]
        g_idx = idx[:, 2]
        p_idx = gt_to_pred[assigned].to(torch.long)      # (N,)

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
    return gt_to_pred, pred_to_gt, layer_mask_mean, layer_dice_mean, matched_tensor
    focal_gamma = 0.0 if focal_gamma is None else float(focal_gamma)
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")
    if focal_alpha is None or (isinstance(focal_alpha, (int, float)) and float(focal_alpha) < 0.0):
        alpha_neg = 1.0
    else:
        alpha = float(focal_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("focal_alpha must be in [0, 1]")
        alpha_neg = 1.0 - alpha
    use_gamma = focal_gamma != 0.0

