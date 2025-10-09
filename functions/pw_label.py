import torch
import torch.nn.functional as F
from typing import List

def pairwise_label_loss(
    logits: torch.Tensor,
    targets: List[torch.Tensor],
):
    """
    Computes a pairwise binary cross-entropy loss and returns a list of tensors.

    For each item in the batch, this function computes the binary cross-entropy
    loss between each of the C predicted classes (aggregated across all Q queries)
    and each of the present ground truth classes.

    Args:
        logits (torch.Tensor): A tensor of shape (B, Q, C) representing the
                             predicted logits. B is the batch size, Q is the
                             number of queries, and C is the number of classes.
        targets (List[torch.Tensor]): A list of B tensors. Each tensor is a 1D
                                    tensor of shape (num_gt,) containing integer
                                    class labels for a batch item.

    Returns:
        List[torch.Tensor]: A list of B tensors, where each tensor has the shape
                            (Q, num_present_gt_classes). Each column in a tensor
                            corresponds to a ground truth class and contains the
                            loss for each of the C predicted classes against that
                            ground truth class.
    """
    # Validate input shapes
    if logits.dim() != 3:
        raise ValueError(f"Logits must have shape (B, Q, C), but got {logits.shape}")
    B, Q, C = logits.shape
    device = logits.device

    if not isinstance(targets, list) or len(targets) != B:
        raise ValueError(f"Targets must be a list of length B, but got {len(targets)}")

    # A list to hold the final loss tensor for each item in the batch.
    batch_losses = []

    # Iterate over each item in the batch
    for i in range(B):
        logits_i = logits[i]  # Shape: (Q, C)
        targets_i = targets[i].long().to(device)
        num_gt = targets_i.numel()

        # If there are no ground truth classes for this item, append an empty tensor
        if num_gt == 0:
            batch_losses.append(torch.empty((C, 0), dtype=logits.dtype, device=device))
            continue

        # Create a one-hot representation of the ground truth labels for this item.
        # Shape: (num_gt, C)
        one_hot_targets = F.one_hot(targets_i, num_classes=C).to(logits.dtype)

        # Prepare tensors for broadcasting to compute pairwise loss efficiently.
        # We want to compare each of the (Q, C) logits with each of the num_gt labels.
        
        # Reshape logits to (1, Q, C) for broadcasting against targets
        l = logits_i.unsqueeze(0)
        
        # Reshape targets to (num_gt, 1, C) for broadcasting against logits
        t = one_hot_targets.unsqueeze(1)

        # l broadcasts to (num_gt, Q, C)
        # t broadcasts to (num_gt, Q, C)
        # The result `bce` contains the element-wise loss for each
        # (ground_truth_class, query, predicted_class) triplet.
        bce = F.binary_cross_entropy_with_logits(l, t, reduction='none')

        # Mean over the class dimension Result shape: (num_gt, Q)
        loss_per_pair = bce.mean(dim=-1)
        
        # Transpose to get the desired shape (Q, num_gt), matching the reference function.
        # Each column now represents a ground truth class.
        final_item_loss = loss_per_pair.T
        
        batch_losses.append(final_item_loss)

    return batch_losses