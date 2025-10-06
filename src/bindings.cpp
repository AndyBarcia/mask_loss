#include <torch/extension.h>

torch::Tensor sigmoid_cross_entropy_forward(
    const torch::Tensor& logits, 
    const torch::Tensor& targets
);

torch::Tensor sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out, 
    const torch::Tensor& logits, 
    const torch::Tensor& targets
);

torch::Tensor mc_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& class_mapping
);

torch::Tensor mc_sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& class_mapping
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sigmoid_ce_loss", &sigmoid_cross_entropy_forward, "Sigmoid Cross Entropy forward (CUDA)");
    m.def("backward_sigmoid_ce_loss", &sigmoid_cross_entropy_backward, "Sigmoid Cross Entropy backward (CUDA)");
    m.def("forward_mc_sigmoid_ce_loss", &mc_sigmoid_cross_entropy_forward, "Sigmoid Cross Entropy forward (CUDA)");
    m.def("backward_mc_sigmoid_ce_loss", &mc_sigmoid_cross_entropy_backward, "Sigmoid Cross Entropy backward (CUDA)");
}