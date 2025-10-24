#include <torch/extension.h>

torch::Tensor sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    const float num_masks,
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f
);

torch::Tensor sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out, // (L,), float
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    const float num_masks,
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f
);

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    int64_t background_index = -1,
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f
);

std::vector<torch::Tensor> dice_loss_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    const float smooth,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor dice_loss_backward(
    const torch::Tensor& grad_out,               // (L,), float
    const torch::Tensor& logits,                 // (L,B,C,H,W), float
    const torch::Tensor& targets,                // (B,H_t,W_t), int64
    const torch::Tensor& total_intersection_sum, // (L,B,C), float
    const torch::Tensor& total_p_sum,            // (L,B,C), float
    const torch::Tensor& total_t_sum,            // (L,B,C), float
    const float smooth,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor pairwise_dice_loss_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    const float smooth,
    int64_t background_index = -1,
    const float scale = 1.0f
);

torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT),      int64,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    const float cls_scale = 1.0f,
    int64_t background_index = -1,
    const float mask_gamma = 0.0f,
    const float mask_alpha = -1.0f,
    const float cls_gamma = 0.0f,
    const float cls_alpha = -1.0f
);

// Hybrid mask matcher front-end.  Returns ``{pred_to_gt, pred_round,
// layer_mask_mean, layer_dice_mean, layer_cls_mean}``.
std::vector<torch::Tensor> mask_matching(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT),      int64,
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    float   cls_scale       = 1.0f,
    int64_t background_index= -1,
    double  inf_thresh      = 1e30,
    double  num_masks       = -1.0,
    bool    force_unmatched_class_to_background = false,
    bool    force_unmatched_masks_to_empty      = false,
    int64_t topk_matches    = 1,
    int64_t strategy_id     = 0,
    float   mask_gamma      = 0.0f,
    float   mask_alpha      = -1.0f,
    float   cls_gamma       = 0.0f,
    float   cls_alpha       = -1.0f,
    int64_t void_class_index = -1
);

std::vector<torch::Tensor> mask_matching_backward(
    const torch::Tensor& grad_layer_mask_mean, // (L,), float
    const torch::Tensor& grad_layer_dice_mean, // (L,), float
    const torch::Tensor& grad_layer_cls_mean,  // (L,), float
    const torch::Tensor& mask_logits,          // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,         // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,           // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,          // (B,GT_total), int64
    const torch::Tensor& pred_to_gt,           // (L,B,Q), int64
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    int64_t background_index,
    const double num_masks,
    const bool force_unmatched_class_to_background,
    const bool force_unmatched_masks_to_empty,
    const float mask_gamma,
    const float mask_alpha,
    const float cls_gamma,
    const float cls_alpha,
    int64_t void_class_index
);

torch::Tensor pairwise_label_loss_forward(
    const torch::Tensor& logits,   // (L,B,Q,C), float
    const torch::Tensor& targets,  // (B,GT_total), int64 with -1 padding
    int64_t background_index = -1,
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sigmoid_ce_loss", &sigmoid_cross_entropy_forward, "Sigmoid Cross Entropy forward (CUDA)");
    m.def("backward_sigmoid_ce_loss", &sigmoid_cross_entropy_backward, "Sigmoid Cross Entropy backward (CUDA)");
    m.def("forward_pw_sigmoid_ce_loss", &pairwise_sigmoid_cross_entropy_forward, "Sigmoid Cross Entropy forward (CUDA)");
    m.def("forward_dice_loss", &dice_loss_forward, "Dice loss forward (CUDA)");
    m.def("backward_dice_loss", &dice_loss_backward, "Dice loss backward (CUDA)");
    m.def("forward_pw_dice_loss", &pairwise_dice_loss_forward, "Dice loss forward (CUDA)");
    m.def("pairwise_mask_loss_forward", &pairwise_mask_loss_forward, "Dice+Sigmoid loss forward (CUDA)");
    m.def("mask_matching", &mask_matching, "Mask matching using OR-Tools (CUDA)");
    m.def("mask_matching_backward", &mask_matching_backward, "Mask matching backward (CUDA)");
    m.def("forward_pw_label_loss", &pairwise_label_loss_forward, "Sigmoid Cross Entropy label forward (CUDA)");
}

