import torch
import torch.nn.functional as F
from typing import List, Union


def pairwise_label_loss(
    logits: torch.Tensor,
    targets: Union[List[torch.Tensor], torch.Tensor],
    loss_type: str = 'binary_cross_entropy',
    focal_loss_gamma: float = 2.0,
    focal_loss_alpha: float = 0.25,
) -> torch.Tensor:
    """
    Computes a pairwise loss tensor for sequence-based predictions.

    This function combines query-based target assignment with a pairwise binary
    loss calculation. For each of the Q queries, it computes a separate
    loss against every possible ground truth class (from 0 to C-1).

    Args:
        logits (torch.Tensor): A tensor of shape (B, Q, C) representing the
                               predicted logits for each of Q queries. C is the
                               number of classes.
        targets (Union[List[torch.Tensor], torch.Tensor]): A list of 1D tensors,
                               one for each item in the batch. Each tensor
                               contains the ground truth class indices for that item.
        loss_type (str): The type of loss to compute. Must be either 'cross_entropy', 
                               'binary_cross_entropy', 'focal_loss', or 'binary_focal_loss'
        focal_loss_gamma (float): The gamma parameter for focal loss.
        focal_loss_alpha (float): The alpha parameter for focal loss.

    Returns:
        torch.Tensor: A tensor of shape (B, Q, max_GT) containing the pairwise
                      loss between each query and ground truth. As batches contain
                      different number of elements, the output will be masked
                      with infinity values.
    """
    if loss_type not in {
        'cross_entropy', 'binary_cross_entropy', 'focal_loss', 'binary_focal_loss'
    }:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    B, Q, C = logits.shape
    device = logits.device
    dtype = logits.dtype

    # Normalize targets into a list of 1D LongTensors (one per batch element).
    if isinstance(targets, torch.Tensor):
        if targets.dim() == 2:
            # treat negative indices as padding
            targets_list = []
            for b in range(B):
                row = targets[b]
                if row.numel() == 0:
                    targets_list.append(torch.empty(0, dtype=torch.long, device=device))
                else:
                    # allow -1 or any negative to be padding marker
                    valid = row >= 0
                    targets_list.append(row[valid].to(torch.long).to(device))
        elif targets.dim() == 1:
            # only allowed if batch size is 1
            if B != 1:
                raise ValueError("When passing 1D targets tensor, logits batch size must be 1.")
            targets_list = [targets.to(torch.long).to(device)]
        else:
            raise ValueError("Unsupported targets tensor shape.")
    else:
        # assume list-like of tensors
        targets_list = []
        for t in targets:
            t = t.to(device)
            if t.numel() == 0:
                targets_list.append(torch.empty(0, dtype=torch.long, device=device))
            else:
                targets_list.append(t.to(torch.long).to(device))

    # find max number of GTs across batch
    max_G = max((t.numel() for t in targets_list), default=0)

    # output initialized with +inf for masked entries
    out = torch.full((B, Q, max_G), float('inf'), device=device, dtype=dtype)

    # Loop over batch items
    for b in range(B):
        gt = targets_list[b]  # 1D LongTensor of length G_b
        G_b = gt.numel()
        if G_b == 0:
            continue  # leave as +inf
        logits_b = logits[b]  # (Q, C)

        if loss_type == 'cross_entropy':
            # multiclass CE per (Q, G_b): -log_softmax(logits)[:, gt_indices]
            log_probs = F.log_softmax(logits_b, dim=-1)  # (Q, C)
            # indexing columns by gt produces (Q, G_b)
            loss_qg = -log_probs[:, gt]  # (Q, G_b)

        elif loss_type == 'focal_loss':
            # multiclass focal: FL = -alpha * (1 - p_t)^gamma * log(p_t)
            probs = F.softmax(logits_b, dim=-1)  # (Q, C)
            log_probs = torch.log(probs + 1e-12)
            p_t = probs[:, gt]  # (Q, G_b)
            log_p_t = log_probs[:, gt]  # (Q, G_b)
            modulator = (1.0 - p_t).pow(focal_loss_gamma)
            loss_qg = -focal_loss_alpha * modulator * log_p_t  # (Q, G_b)

        elif loss_type == 'binary_cross_entropy':
            # treat each GT as one-hot vector across C, compute sum over classes of
            # per-class binary cross entropy -> same pattern as batch_sigmoid_ce_loss
            # pos = BCE logits vs 1  ; neg = BCE logits vs 0
            pos = F.binary_cross_entropy_with_logits(logits_b, torch.ones_like(logits_b), reduction='none')  # (Q,C)
            neg = F.binary_cross_entropy_with_logits(logits_b, torch.zeros_like(logits_b), reduction='none')  # (Q,C)
            # targets one-hot: use einsum to sum pos over classes where target=1 and neg where target=0
            # gt_one_hot shape (G_b, C)
            gt_one_hot = F.one_hot(gt, num_classes=C).to(dtype=dtype)  # (G_b, C)
            # einsum: "qc,gc->qg"
            loss_qg = torch.einsum("qc,gc->qg", pos, gt_one_hot) + torch.einsum("qc,gc->qg", neg, (1.0 - gt_one_hot))

        elif loss_type == 'binary_focal_loss':
            # per-class sigmoid focal loss with alpha for positives and (1-alpha) for negatives:
            # pos_base = BCE_with_logits(inputs,1) = -log(sigmoid(x)); neg_base = BCE(...,0) = -log(1-sigmoid(x))
            p = torch.sigmoid(logits_b)  # (Q,C)
            pos_base = F.binary_cross_entropy_with_logits(logits_b, torch.ones_like(logits_b), reduction='none')  # (Q,C)
            neg_base = F.binary_cross_entropy_with_logits(logits_b, torch.zeros_like(logits_b), reduction='none')  # (Q,C)
            pos_mod = (1.0 - p).pow(focal_loss_gamma)  # (Q,C)
            neg_mod = (p).pow(focal_loss_gamma)        # (Q,C)
            pos = focal_loss_alpha * pos_base * pos_mod
            neg = (1.0 - focal_loss_alpha) * neg_base * neg_mod

            gt_one_hot = F.one_hot(gt, num_classes=C).to(dtype=dtype)  # (G_b, C)
            loss_qg = torch.einsum("qc,gc->qg", pos, gt_one_hot) + torch.einsum("qc,gc->qg", neg, (1.0 - gt_one_hot))

        else:
            raise RuntimeError("Unhandled loss_type; should not reach here.")

        # place into output; transpose to (Q, G_b) already
        # ensure dtype matches out dtype
        out[b, :, :G_b] = loss_qg.to(dtype=dtype, device=device)

    return out