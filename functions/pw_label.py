import torch
import torch.nn.functional as F
from typing import List, Union

from .mc_label import _as_list_of_1d_tensors

def pairwise_label_loss(
    logits: torch.Tensor,
    targets: Union[List[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Computes a pairwise loss tensor for sequence-based predictions.

    This function combines query-based target assignment with a pairwise binary
    cross-entropy calculation. For each of the Q queries, it computes a separate
    loss against every possible ground truth class (from 0 to C-1).

    Args:
        logits (torch.Tensor): A tensor of shape (B, Q, C) representing the
                               predicted logits for each of Q queries. C is the
                               number of classes.
        targets (Union[List[torch.Tensor], torch.Tensor]): A list of 1D tensors,
                               one for each item in the batch. Each tensor
                               contains the ground truth class indices for that item.

    Returns:
        torch.Tensor: A tensor of shape (B, Q, C) containing the pairwise
                      binary cross-entropy loss. For a given batch item `b` and
                      ground truth class `gt`, if `gt` is not present in
                      `targets[b]`, the loss values at `pairwise_loss[b, :, gt]`
                      will be infinity.
    """
    if logits.dim() != 3:
        raise ValueError("Logits must have shape (B, Q, C)")
    B, Q, C = logits.shape
    GT = C  # Assume the number of ground truth classes is the same as logit classes

    target_list = _as_list_of_1d_tensors(targets)
    if len(target_list) != B:
        raise ValueError(f"Number of target tensors ({len(target_list)}) must match batch size ({B})")

    device = logits.device

    # `built_target_classes` stores the GT class index assigned to each query.
    # A value of -1 serves as an ignore index for unassigned queries.
    built_target_classes = torch.full((B, Q), -1, dtype=torch.long, device=device)

    # `has_gt_in_item` tracks which GT classes were present in the original targets
    # for each batch item, before assignment to queries.
    has_gt_in_item = torch.zeros((B, GT), dtype=torch.bool, device=device)

    for i in range(B):
        t_i = target_list[i].to(device).long()
        if t_i.numel() == 0:
            continue
        
        # Validate that targets are valid class indices
        if torch.any(t_i < 0) or torch.any(t_i >= GT):
            raise ValueError(f"Target class indices must be in the range [0, {GT-1}]")

        # Mark which GT classes are present in this batch item
        has_gt_in_item[i, t_i] = True

        # Assign the first N ground truths to the first N queries
        num_to_assign = min(t_i.numel(), Q)
        built_target_classes[i, :num_to_assign] = t_i[:num_to_assign]

    # Initialize the final loss tensor to infinity.
    pairwise_loss = torch.full((B, Q, GT), torch.inf, device=device, dtype=logits.dtype)

    # Iterate over each possible ground truth class to compute the pairwise loss
    for gt_class in range(GT):
        # Create a binary target `y` of shape (B, Q).
        # y[b, q] is 1.0 if query `q` in batch `b` was assigned `gt_class`.
        y = (built_target_classes == gt_class).to(dtype=logits.dtype)

        # Select the logits corresponding to the current ground truth class.
        # Shape: (B, Q)
        logits_for_gt = logits[..., gt_class]

        # Compute the element-wise binary cross-entropy with logits.
        # The result `bce` has shape (B, Q).
        bce = F.binary_cross_entropy_with_logits(logits_for_gt, y, reduction='none')

        # Identify which batch items originally contained the current gt_class.
        # Shape: (B,)
        class_is_present = has_gt_in_item[:, gt_class]

        # Use `where` to fill the loss tensor. If a gt_class was not in the
        # original targets for a batch item, its loss for all queries remains infinity.
        # `unsqueeze` broadcasts the (B,) mask to (B, 1) to align with `bce`'s (B, Q) shape.
        pairwise_loss[..., gt_class] = torch.where(
            class_is_present.unsqueeze(1),
            bce,
            torch.tensor(torch.inf, device=device, dtype=bce.dtype)
        )

    return pairwise_loss