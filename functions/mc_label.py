import torch
import torch.nn.functional as F
from typing import List, Optional, Union

def _as_list_of_1d_tensors(targets):
    """
    Normalize targets to a list of 1D LongTensors.
    Supports:
      - list/tuple of 1D tensors, or
      - torch.nested.nested_tensor([...]) objects.
    """
    if isinstance(targets, (list, tuple)):
        out = list(targets)
    else:
        try:
            # For nested tensors, unbind to get a list of tensors
            out = list(targets.unbind())
        except Exception:
            raise TypeError("targets must be a list/tuple of 1D tensors or a NestedTensor")
    return out


def multiclass_label_loss(
    logits: torch.Tensor,
    targets: Union[List[torch.Tensor], torch.Tensor],
    class_mapping: torch.Tensor,
    bg_class_index: Optional[int] = None,
    loss_type: str = 'cross_entropy',
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: float = 0.25,
    num_detections: Optional[Union[int, float, torch.Tensor]] = None,
):
    """
    Classification loss supporting variable-length targets per batch.

    Args:
        logits: (B, Q, C) classification logits.
        targets: List[Tensor] or NestedTensor of length B, each element is a 1D tensor
                of encoded target values (0-255).
        class_mapping: (B, 256) mapping from encoded values to class indices (-1 = ignore).
        bg_class_index: required for multiclass losses; None for binary losses.
        loss_type: 'cross_entropy', 'focal_loss', 'binary_cross_entropy', or 'binary_focal_loss'.
        focal_loss_gamma, focal_loss_alpha: focal loss parameters.
        num_detections: optional scalar (int/float/torch.Tensor). If provided, final loss is
                (sum of per-element losses) / max(1.0, num_detections). If None,
                defaults to normalization by B * Q.

    Returns:
        Scalar tensor with the mean loss.
    """
    if logits.dim() != 3:
        raise ValueError("logits must have shape (B, Q, C)")
    B, Q, C = logits.shape

    # Normalize targets to list
    target_list = _as_list_of_1d_tensors(targets)
    if len(target_list) != B:
        raise ValueError(f"Number of target tensors ({len(target_list)}) must match batch size ({B})")

    # class_mapping checks
    if class_mapping.dim() != 2 or class_mapping.shape[1] != 256:
        raise ValueError("class_mapping must have shape (B, 256)")
    if class_mapping.shape[0] != B:
        raise ValueError("class_mapping first dimension must match batch size")

    device = logits.device
    class_mapping = class_mapping.long().to(device)

    is_binary_loss = loss_type in ['binary_cross_entropy', 'binary_focal_loss']
    if is_binary_loss and bg_class_index is not None:
        raise ValueError("bg_class_index must be None for binary loss variants.")
    if not is_binary_loss and bg_class_index is None:
        raise ValueError("bg_class_index must be specified for multiclass loss variants.")
    if not is_binary_loss and not (0 <= bg_class_index < C):
        raise ValueError("bg_class_index must be a valid class index")

    # Build targets (same assignment policy: queries 0..N-1 get GTs, upto Q)
    if is_binary_loss:
        # will compute element-wise BCE and sum later
        built_binary_targets = torch.zeros_like(logits, dtype=logits.dtype, device=device)  # (B,Q,C)
    else:
        built_target_classes = torch.full((B, Q), bg_class_index, dtype=torch.long, device=device)  # (B,Q)

    for i in range(B):
        t_i = target_list[i].to(device).long()
        if t_i.numel() == 0:
            continue
        if torch.any(t_i < 0) or torch.any(t_i > 255):
            raise ValueError("targets must be in range [0, 255]")

        mapped = class_mapping[i, t_i]  # (num_gt,)
        valid_mask = (mapped >= 0) & (mapped < C)
        valid = mapped[valid_mask]
        if valid.numel() == 0:
            continue

        num_to_assign = min(valid.numel(), Q)
        assigned = valid[:num_to_assign]

        if is_binary_loss:
            query_idx = torch.arange(num_to_assign, device=device)
            built_binary_targets[i, query_idx, assigned] = 1.0
        else:
            built_target_classes[i, :num_to_assign] = assigned

    # Determine denominator for final normalization
    if num_detections is None:
        denom = torch.tensor(float(B * Q), dtype=logits.dtype, device=device)
    else:
        if isinstance(num_detections, torch.Tensor):
            denom = num_detections.to(device=device, dtype=logits.dtype)
            if denom.numel() != 1:
                raise ValueError("num_detections tensor must be a scalar")
        else:
            denom = torch.tensor(float(num_detections), dtype=logits.dtype, device=device)
    # Also normalize by number of classes.
    denom *= C

    # Compute summed loss (not averaged) then divide by denom
    if is_binary_loss:
        if loss_type == 'binary_cross_entropy':
            # element-wise BCE, sum over all elements
            ce = F.binary_cross_entropy_with_logits(logits, built_binary_targets, reduction='sum')
            total_loss = ce
        else:  # binary_focal_loss
            p = logits.sigmoid()
            ce_elem = F.binary_cross_entropy_with_logits(logits, built_binary_targets, reduction='none')  # (B,Q,C)
            p_t = p * built_binary_targets + (1 - p) * (1 - built_binary_targets)
            alpha_t = focal_loss_alpha * built_binary_targets + (1 - focal_loss_alpha) * (1 - built_binary_targets)
            focal_elem = alpha_t * ((1 - p_t).pow(focal_loss_gamma)) * ce_elem
            total_loss = focal_elem.sum()
    else:
        logits_flat = logits.view(B * Q, C)
        targets_flat = built_target_classes.view(B * Q)
        if loss_type == 'cross_entropy':
            # sum of cross-entropy over all items
            total_loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
        else:  # focal_loss
            ce_elem = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*Q,)
            pt = torch.exp(-ce_elem)
            focal_elem = focal_loss_alpha * (1 - pt).pow(focal_loss_gamma) * ce_elem
            total_loss = focal_elem.sum()

    loss = total_loss / denom
    return loss
