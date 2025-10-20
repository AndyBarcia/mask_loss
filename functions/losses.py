from .sigmoid_ce import SigmoidCELossFunction, sigmoid_cross_entropy_loss_efficient_py
from .mc_sigmoid_ce import MultiClassSigmoidCELossFunction, multiclass_sigmoid_cross_entropy_loss_py

from .dice import DiceLossFunction, dice_loss_efficient_py
from .mc_dice import MultiClassDiceLossFunction, multiclass_dice_loss_efficient_py


def sigmoid_cross_entropy_loss(
    logits, # (B,C,h,w)
    targets, # (B,H,W)
    class_mapping=None, # (256,)
    num_masks=None,
    implementation="cuda",
    scale=1.0,
):
    """
    Efficient count-based BCE-with-logits for non-mutually-exclusive multi-class case.
    logits: (B, C, h, w)
    targets: (B, H, W) integer labels in [0, C-1] or [0, 255] if using class mappings.
    class_mapping: (256,) a 1D tensor that maps the encoded values in
            targets to class indices from logits.
    Returns: scalar tensor (mean over all elements: B*C*H*W)
    """
    if implementation == "python":
        if class_mapping is None:
            return sigmoid_cross_entropy_loss_efficient_py(logits, targets, num_masks, scale)
        else:
            return multiclass_sigmoid_cross_entropy_loss_py(logits, targets, class_mapping, num_masks)
    else:
        if class_mapping is None:
            return SigmoidCELossFunction(logits, targets, num_masks, scale)
        else:
            return MultiClassSigmoidCELossFunction(logits, targets, class_mapping, num_masks)


def dice_loss(
    logits, # (B,C,h,w)
    targets, # (B,H,W)
    smooth=1e-6,
    class_mapping=None, # (256,)
    num_masks=None,
    implementation="cuda",
    scale=1.0,
):
    """
    Efficient count-based Dice loss consistent with nearest upsampling behavior.
    Assumes H_t and W_t are integer multiples of h and w respectively and that
    nearest upsampling repeats the same logit value across each s x s block.

    logits: (B, C, h, w)
    targets: (B, H, W) integer labels in [0, C-1] or [0, 255] if using class mappings.
    smooth: small float to avoid division by zero
    class_mapping: (256,) a 1D tensor that maps the encoded values in
        targets to class indices from logits.

    Returns: scalar tensor (mean Dice loss over B and C)
    """
    if implementation == "python":
        if class_mapping is None:
            return dice_loss_efficient_py(logits, targets, smooth, num_masks, scale)
        else:
            return multiclass_dice_loss_efficient_py(logits, targets, class_mapping, smooth, num_masks)
    else:
        if class_mapping is None:
            return DiceLossFunction(logits, targets, smooth, num_masks, scale)
        else:
            return MultiClassDiceLossFunction(logits, targets, class_mapping, smooth, num_masks)

