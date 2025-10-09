import torch
from scipy.optimize import linear_sum_assignment
from typing import List

from .pw_sigmoid_ce import PairwiseSigmoidCELossFunction
from .pw_dice import PairwiseDiceLossFunction
from .pw_label import pairwise_label_loss


def hungarian_match_assignment(
    cls_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    mask_targets: torch.Tensor,
    cls_targets: List[torch.Tensor],
    cost_class: float = 1.0,
    cost_mask: float = 1.0,
    cost_dice: float = 1.0,
    use_binary_classification: bool = False,
    use_focal_classification: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """
    Performs Hungarian matching with configurable costs and loss types.

    Args:
        cls_logits: (B, Q, C) classification logits for each query.
        mask_logits: (B, Q, H, W) mask logits for each query.
        mask_targets: (B, H_t, W_t) ground truth masks.
        cls_targets: List of B tensors with class labels for each mask.
        cost_class: Weight for the classification cost.
        cost_mask: Weight for the sigmoid cross-entropy mask cost.
        cost_dice: Weight for the Dice mask cost.
        use_binary_classification: If True, use binary variants of the classification loss.
        use_focal_classification: If True, use focal loss for classification cost.
        focal_alpha: Alpha parameter for focal loss.
        focal_gamma: Gamma parameter for focal loss.

    Returns:
        class_mapping: (B, 256) tensor mapping mask values to assigned query indices.
    """
    B, Q, C = cls_logits.shape
    device = cls_logits.device

    # Mask costs
    mask_ce_cost = PairwiseSigmoidCELossFunction.apply(mask_logits, mask_targets)
    mask_dice_cost = PairwiseDiceLossFunction.apply(mask_logits, mask_targets)

    # Classification cost
    if use_focal_classification:
        cls_loss_type = 'binary_focal_loss' if use_binary_classification else 'focal_loss'
    else:
        cls_loss_type = 'binary_cross_entropy' if use_binary_classification else 'cross_entropy'
    cls_cost = pairwise_label_loss(
        cls_logits,
        cls_targets,
        loss_type = cls_loss_type,
        focal_loss_alpha = focal_alpha if use_focal_classification else None,
        focal_loss_gamma = focal_gamma if use_focal_classification else None
    )

    # Final mapping tensor, initialized to -1 (no assignment)
    class_mapping = torch.full((B, 256), -1, dtype=torch.long, device=device)

    # Iterate per batch item to perform the assignment
    for b in range(B):
        gt_labels = cls_targets[b].to(device)
        num_gt_objects = len(gt_labels)

        if num_gt_objects == 0:
            continue

        # Select costs for non-background masks
        batch_mask_ce_cost = mask_ce_cost[b, :, 1:num_gt_objects + 1]
        batch_mask_dice_cost = mask_dice_cost[b, :, 1:num_gt_objects + 1]

        # Select costs for the corresponding ground truth class labels
        batch_cls_cost = cls_cost[b, :, gt_labels]

        # Combine the costs with their respective weights
        cost_matrix = (
            cost_mask * batch_mask_ce_cost +
            cost_dice * batch_mask_dice_cost +
            cost_class * batch_cls_cost
        )

        # Run the Hungarian Algorithm
        cost_matrix_cpu = cost_matrix.cpu().detach().numpy()
        query_indices, gt_indices = linear_sum_assignment(cost_matrix_cpu)

        # Populate the Class Mapping to later used for loss computation.
        for query_idx, gt_obj_idx in zip(query_indices, gt_indices):
            gt_mask_value = gt_obj_idx + 1
            if gt_mask_value < 256:
                class_mapping[b, gt_mask_value] = query_idx

    return class_mapping