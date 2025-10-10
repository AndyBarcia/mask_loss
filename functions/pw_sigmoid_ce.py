import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, einsum
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint

try:
    import mask_loss
except ImportError:
    print("CUDA extension pos_mlp_bias not found. Please compile it first.")
    print("Run: pip3 install --no-build-isolation .")

class PairwiseSigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, background_index=None):
        B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        logits = logits.contiguous().float()
        targets = targets.contiguous()
        output = mask_loss.forward_pw_sigmoid_ce_loss(
            logits, 
            targets, 
            background_index if background_index is not None else -1
        )
        ctx.background_index = background_index
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.background_index is not None:
            return None, None, None
        else:
            return None, None

def pairwise_sigmoid_cross_entropy_loss_py(logits, targets, background_index=None):
    """
    Computes pairwise sigmoid cross-entropy loss.

    This function calculates the loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the binary cross-entropy with logits.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, h, w) representing the
                            predicted logits for each class. B is the batch size,
                            C is the number of classes, and h, w are the spatial
                            dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                            labels for the ground truth. H_t and W_t are the
                            spatial dimensions of the targets.
        background_index (Optional[int]): The index that corresponds to the background
                            to be ignored. If not provided, all classses are
                            computed normally. If specified, the output tensor
                            has the column corresponding to the background removed.

    Returns:
        torch.Tensor: A tensor of shape (B, C, max_GT) where max_GT is the maximum value
                      in the targets tensor + 1 if no background index is provided, or
                      th maximum value in the targets tensor otherwise. It contains the 
                      pairwise loss for each class against each possible target. For target 
                      masks with 0 area, a value of infinity is returned.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"

    # Upsample logits to match the spatial dimensions of the targets
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')

    device = logits.device
    targets_long = targets.long().to(device)

    # Determine the full range of GT classes present in the targets
    gt_max = targets_long.max().item()
    GT_all = gt_max + 1

    # Build list of ground-truth classes to evaluate, optionally excluding background
    if background_index is None:
        gt_classes = list(range(GT_all))
    else:
        # If background_index is outside the observed range, it has no effect
        gt_classes = [i for i in range(GT_all) if i != background_index]

    GT = len(gt_classes)

    # Initialized to infinity.
    pairwise_loss = torch.full((B, C, GT), torch.inf, device=device)

    # Iterate over each ground truth class we will evaluate
    for out_idx, gt_class in enumerate(gt_classes):
        # Create a binary target mask for the current ground truth class
        y = (targets_long == gt_class).unsqueeze(1).to(dtype=logits.dtype)  # shape (B,1,H_t,W_t)

        # Check which batch elements contain this GT class
        has_class = y.sum(dim=(2, 3)) > 0  # Shape: (B, 1)

        # Stable BCE-with-logits calculation (broadcasts y over channels)
        maxL = torch.clamp(logits_up, min=0.0)
        logexp = torch.log1p(torch.exp(-torch.abs(logits_up)))
        bce_elem = maxL - logits_up * y + logexp  # shape (B,C,H_t,W_t)

        # Sum the loss for each predicted class and normalize
        loss_per_class = bce_elem.sum(dim=(2, 3)) / (H_t * W_t)  # shape (B,C)

        # Only update loss where the class exists in the batch element, otherwise keep inf
        inf_tensor = torch.tensor(torch.inf, device=device, dtype=loss_per_class.dtype)
        pairwise_loss[:, :, out_idx] = torch.where(
            has_class.expand(-1, C),
            loss_per_class,
            inf_tensor
        )

    return pairwise_loss


def pairwise_sigmoid_cross_entropy_loss_efficient_py(logits, targets, background_index=None):
    """
    Computes pairwise sigmoid cross-entropy loss in an efficient way.

    This function calculates the loss for each predicted class against every
    possible ground truth class. It upsamples the logits to the resolution of the
    targets, creates a one-hot representation for each potential target class,
    and then computes the binary cross-entropy with logits.

    Args:
        logits (torch.Tensor): A tensor of shape (B, C, h, w) representing the
                            predicted logits for each class. B is the batch size,
                            C is the number of classes, and h, w are the spatial
                            dimensions of the logits.
        targets (torch.Tensor): A tensor of shape (B, H_t, W_t) with integer
                            labels for the ground truth. H_t and W_t are the
                            spatial dimensions of the targets.
        background_index (Optional[int]): The index that corresponds to the background
                            to be ignored. If not provided, all classses are
                            computed normally. If specified, the output tensor
                            has the column corresponding to the background removed.

    Returns:
        torch.Tensor: A tensor of shape (B, C, max_GT) where max_GT is the maximum value
                      in the targets tensor + 1 if no background index is provided, or
                      th maximum value in the targets tensor otherwise. It contains the 
                      pairwise loss for each class against each possible target. For target 
                      masks with 0 area, a value of infinity is returned.
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch between logits and targets"
    assert H_t % h == 0 and W_t % w == 0, \
        "Target dimensions must be integer multiples of logit dimensions"
    
    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation requires equal scaling factors for height and width.")
    s = sH
    N2 = s * s

    device = logits.device
    targets_long = targets.long().to(device)

    # Determine the full range of GT classes present in the targets
    gt_max = targets_long.max().item()
    GT_all = gt_max + 1

    # Build list of ground-truth classes to evaluate, optionally excluding background
    if background_index is None:
        gt_classes = list(range(GT_all))
    else:
        # If background_index is outside the observed range, it has no effect
        gt_classes = [i for i in range(GT_all) if i != background_index]

    GT = len(gt_classes)

    # Initialized to infinity.
    pairwise_loss = torch.full((B, C, GT), torch.inf, device=device)
    
    L_reshaped = logits.reshape(B, C, h * w)
    
    # Precompute parts of the stable BCE-with-logits formula
    maxL = torch.clamp(L_reshaped, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L_reshaped)))

    # Iterate over each ground truth class we will evaluate
    for out_idx, gt_class in enumerate(gt_classes):
        # Create a binary mask for the current ground truth class
        onehot_gt = (targets_long == gt_class).unsqueeze(1).to(dtype=logits.dtype)
        
        # Check which batch elements contain this GT class
        has_class = onehot_gt.sum(dim=(2, 3)) > 0  # Shape: (B, 1)
        
        # Reshape for unfolding to count ground truth classes in each block
        Bc = onehot_gt.reshape(B, 1, H_t, W_t)
        unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))
        unf = unf.reshape(B, 1, s * s, h * w)
        n_k = unf.sum(dim=2)  # Shape: (B, 1, h*w)

        # Stable BCE-with-logits summed across the block for each predicted class
        loss_block = N2 * maxL - L_reshaped * n_k + N2 * logexp
        
        # Sum the loss over the spatial dimensions and normalize
        loss_sum = loss_block.sum(dim=2) / (H_t * W_t)
        
        # Only update loss where the class exists in the batch element
        pairwise_loss[:, :, out_idx] = torch.where(
            has_class.expand(-1, C),
            loss_sum,
            torch.tensor(torch.inf, device=device, dtype=loss_sum.dtype)
        )
        
    return pairwise_loss