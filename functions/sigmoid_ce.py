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

class SigmoidCELossFunction(Function):
    @staticmethod
    def forward(ctx, logits, targets, num_masks):
        B, C, h, w = logits.shape
        B_t, H_t, W_t = targets.shape
        assert B == B_t, "Batch size mismatch between logits and targets"
        
        if num_masks is None:
            B, C = logits.shape[:2]
            num_masks = float(B*C)
        ctx.num_masks = num_masks

        logits = logits.contiguous().float()
        targets = targets.contiguous()
        ctx.save_for_backward(logits, targets)
        output = mask_loss.forward_sigmoid_ce_loss(logits, targets, num_masks)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_weights = mask_loss.backward_sigmoid_ce_loss(grad_output, logits, targets, ctx.num_masks)
        return grad_weights, None, None

def sigmoid_cross_entropy_loss_inefficient_py(logits, targets, num_masks=None):
    """
    Naive approach: upsample logits (nearest) to high-res, build per-class one-hot targets,
    and compute BCEWithLogits per pixel then mean.
    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, C-1]
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    logits_up = F.interpolate(logits, size=(H_t, W_t), mode='nearest')
    device = logits.device
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=logits.dtype)

    L = logits_up
    y = onehot
    maxL = torch.clamp(L, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L)))
    bce_elem = maxL - L * y + logexp

    num_masks = B*C if num_masks is None else num_masks
    loss = bce_elem.sum() / (num_masks * H_t * W_t)
    return loss

def sigmoid_cross_entropy_loss_py(logits, targets, num_masks=None):
    """
    Efficient count-based BCE-with-logits for non-mutually-exclusive multi-class case.
    logits: (B, C, h, w)
    targets: (B, H_t, W_t) integer labels in [0, C-1] (interpreted as one-hot per-class)
             H_t and W_t must be integer multiples of h and w respectively.
    Returns: scalar tensor (mean over all elements: B*C*H_t*W_t)
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"
    sH = H_t // h
    sW = W_t // w
    if sH != sW:
        raise ValueError("This implementation assumes equal scale factor for height and width (square blocks)")
    s = sH
    N2 = s * s

    device = logits.device
    targets_long = targets.long().to(device)

    # One-hot encode targets.
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=logits.dtype)  # (B,C,H_t,W_t)

    # Reshape for unfolding into high-res blocks. This will allows us
    # to count the number of ground truth classes in each block.
    Bc = onehot.reshape(B * C, 1, H_t, W_t)
    unf = F.unfold(Bc, kernel_size=(s, s), stride=(s, s))  # (B*C, s*s, h*w)
    unf = unf.reshape(B, C, s * s, h * w)
    n_k = unf.sum(dim=2)  # (B, C, h*w)

    # Stable BCE-with-logits summed across block
    L = logits.reshape(B, C, h * w)
    maxL = torch.clamp(L, min=0.0)
    logexp = torch.log1p(torch.exp(-torch.abs(L)))
    loss_block = N2 * maxL - L * n_k + N2 * logexp  # (B, C, h*w)

    num_masks = B*C if num_masks is None else num_masks
    loss = loss_block.sum() / (num_masks * H_t * W_t)
    return loss


def calculate_uncertainty(logits):
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, 
    uncertainty_func, 
    num_points, 
    oversample_ratio, 
    importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def _sigmoid_bce_from_logits_labels(logits, labels):
    """
    Stable BCE-with-logits elementwise, summed over all elements.
    logits, labels: same shape (...), labels in [0,1].
    """
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')

def sigmoid_cross_entropy_loss_sampling_py(
    logits,        # (B, C, h, w) logits
    targets,          # (B, H_t, W_t) integer labels in [0, C-1]
    num_masks=None,   # normalization (defaults to B*C)
    num_points=5000,
    oversample_ratio=3,
    importance_sample_ratio=0.75,
):
    """
    logits: (B, C, h, w) logits (float)
    targets:   (B, H_t, W_t) integer labels (long) in [0, C-1]
    """
    B, C, h, w = logits.shape
    B_t, H_t, W_t = targets.shape
    assert B == B_t, "Batch size mismatch"
    assert H_t % h == 0 and W_t % w == 0, "High-res dims must be integer multiples of low-res dims"

    device = logits.device
    dtype = logits.dtype

    # Sampled loss but performed per-(image,class) independently.
    # Prepare high-res one-hot per class: (B, C, H_t, W_t)
    targets_long = targets.long().to(device)
    onehot = F.one_hot(targets_long, num_classes=C).permute(0, 3, 1, 2).to(dtype=dtype)

    # Collapse batch & class into "boxes": (B*C, 1, h, w) and (B*C, H_t, W_t)
    per_class_logits = logits.reshape(B * C, 1, h, w)
    per_class_targets = onehot.reshape(B * C, H_t, W_t)  # values in {0,1}

    # Use PointRend sampler which expects inputs shaped (N, 1, Hmask, Wmask)
    point_coords = get_uncertain_point_coords_with_randomness(
        per_class_logits,                      # (N=B*C, 1, h, w)
        lambda logits: calculate_uncertainty(logits),  # calculate_uncertainty expects channel==1
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    )  # -> (B*C, P, 2)

    # Sample logits and labels at those points
    sampled_logits = point_sample(per_class_logits, point_coords, align_corners=False)  # (B*C,1,P)
    if sampled_logits.dim() == 4 and sampled_logits.size(-1) == 1:
        sampled_logits = sampled_logits.squeeze(-1)
    # ensure shape (B*C, 1, P) or (B*C, P) -> make (B*C, P)
    sampled_logits = sampled_logits.squeeze(1)  # (B*C, P)

    per_class_targets = per_class_targets.unsqueeze(1)  # (B*C,1,H_t,W_t) for point_sample
    sampled_labels = point_sample(per_class_targets, point_coords, align_corners=False)  # (B*C,1,P)
    if sampled_labels.dim() == 4 and sampled_labels.size(-1) == 1:
        sampled_labels = sampled_labels.squeeze(-1)
    sampled_labels = sampled_labels.squeeze(1)  # (B*C, P)

    # Compute stable BCE sum across all sampled (class,point) pairs
    sampled_loss_sum = _sigmoid_bce_from_logits_labels(sampled_logits, sampled_labels)  # scalar sum

    # Normalization: sampled_loss_sum / (num_masks * P) to produce comparable mean-per-(mask,pixel)
    num_masks = B * C if num_masks is None else num_masks
    P = sampled_logits.shape[-1]
    loss_mask_sampled = sampled_loss_sum / (float(num_masks) * float(P))

    return loss_mask_sampled