import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List

from .pw_sigmoid_ce import PairwiseSigmoidCELossFunction
from .pw_dice import PairwiseDiceLossFunction
from .pw_label import pairwise_label_loss


def hungarian_match_assignment(
    cls_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    mask_targets: torch.Tensor,
    cls_targets: List[torch.Tensor]
) -> torch.Tensor:
    """
    Performs Hungarian matching between predictions and ground truth targets in an
    optimized, batched manner.

    Args:
        cls_logits: (B, Q, C) classification logits for each query.
        mask_logits: (B, Q, H, W) mask logits for each query.
        mask_targets: (B, H_t, W_t) ground truth masks, where integer values
                      greater than 0 represent different object instances.
        cls_targets: A list of B tensors, where the n-th entry of each tensor
                     is the class label for the mask with value n+1.

    Returns:
        class_mapping: (B, 256) tensor mapping mask target values (0-255) to the
                       index of the query (0 to Q-1) assigned to it. A value of -1
                       indicates no assignment.
    """
    B, Q, C = cls_logits.shape
    device = cls_logits.device

    # Calculate Pairwise Costs for the Entire Batch
    mask_ce_cost = PairwiseSigmoidCELossFunction.apply(mask_logits, mask_targets)
    mask_dice_cost = PairwiseDiceLossFunction.apply(mask_logits, mask_targets)
    cls_cost = pairwise_label_loss(cls_logits, cls_targets)

    # The final mapping from mask value to the assigned query index.
    # Initialize with -1 (no assignment).
    class_mapping = torch.full((B, 256), -1, dtype=torch.long, device=device)

    # The matching process is iterative per batch item because the number
    # of ground truth objects can vary.
    for b in range(B):
        gt_labels = cls_targets[b].to(device)
        num_gt_objects = len(gt_labels)

        if num_gt_objects == 0:
            continue

        # Get costs of non-background massks
        batch_mask_ce_cost = mask_ce_cost[b, :, 1:num_gt_objects + 1]
        batch_mask_dice_cost = mask_dice_cost[b, :, 1:num_gt_objects + 1]

        # Get costs of labels.
        batch_cls_cost = cls_cost[b, :, gt_labels]

        # Combine the costs to get the final cost matrix.
        cost_matrix = (
            batch_mask_ce_cost +
            batch_mask_dice_cost +
            batch_cls_cost
        )

        # Run the Hungarian Algorithm
        cost_matrix_cpu = cost_matrix.cpu().detach().numpy()
        query_indices, gt_indices = linear_sum_assignment(cost_matrix_cpu)

        # Populate the Class Mapping
        for query_idx, gt_obj_idx in zip(query_indices, gt_indices):
            gt_mask_value = gt_obj_idx + 1
            if gt_mask_value < 256:
                class_mapping[b, gt_mask_value] = query_idx

    return class_mapping