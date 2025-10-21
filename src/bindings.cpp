#include <torch/extension.h>

torch::Tensor sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    int64_t background_index = -1,
    const float scale = 1.0f
);

std::vector<torch::Tensor> dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor dice_loss_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& total_intersection_sum,
    const torch::Tensor& total_p_sum,
    const torch::Tensor& total_t_sum,
    const float smooth,
    const float num_masks,
    const float scale = 1.0f
);

torch::Tensor pairwise_dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    int64_t background_index = -1,
    const float scale = 1.0f
);

torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    int64_t background_index = -1
);

std::vector<torch::Tensor> mask_matching(
    const torch::Tensor& logits,         // (L,B,C,H,W) CUDA
    const torch::Tensor& targets,        // (B,H_t,W_t) CUDA
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    int64_t background_index= -1,
    double  inf_thresh      = 1e30,
    int64_t num_masks       = -1
);

torch::Tensor mask_matching_backward(
    const torch::Tensor& grad_layer_mask_mean,
    const torch::Tensor& grad_layer_dice_mean,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& matches,
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    int64_t background_index= -1,
    int64_t num_masks       = -1,
    int64_t matched_count   = 0
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
}

