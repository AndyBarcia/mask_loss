#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"
#include "pw_sigmoid_dice.cuh"

torch::Tensor pairwise_sigmoid_dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    int64_t background_index = -1
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int L = logits.size(0);
    const int B = logits.size(1);
    const int C = logits.size(2);
    const int H = logits.size(3);
    const int W = logits.size(4);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    // Automatically compute GT_total from the max value in targets
    torch::Tensor targets_max_tensor = targets.max();
    const int64_t GT_total_64 = targets_max_tensor.item<int64_t>() + 1;
    const int GT_total = static_cast<int>(GT_total_64);

    // Intermediate tensor to store counts of GT_total labels per low-res block
    auto counts = torch::zeros({B, GT_total, H, W}, logits.options().dtype(torch::kUInt8));

    // Launch count kernel (counts for ALL GT labels)
    {
        dim3 block(16,16);
        dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y, B);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value, decltype(W_val)::value,
                decltype(H_t_val)::value, decltype(W_t_val)::value>
                <<<grid, block>>>(
                    targets.data_ptr<int64_t>(),
                    counts.data_ptr<uint8_t>(),
                    B, GT_total
                );
        };
        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 256>{}),  // H
            std::make_tuple(std::integral_constant<int, 256>{}),  // W
            std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
            std::make_tuple(std::integral_constant<int, 1024>{})  // W_t
        );
        const auto runtime_dims = std::make_tuple(H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after count kernel: ", cudaGetErrorString(err));

    // Calculate the total number of pixels for each ground truth label (mask area)
    // This is used to mask 0-area masks with a loss of infinity.
    auto total_counts = counts.sum({2, 3}).to(torch::kInt32).contiguous();

    // Determine whether the output matrix should have one less column (skip background).
    const bool skip_class = (background_index >= 0 && background_index < GT_total);
    if (background_index < 0) {
        // If negative, set to an invalid index to simplify device code
        background_index = GT_total; 
    }
    const int GT_out = GT_total - (skip_class ? 1 : 0);

    // If no classes left to evaluate (edge case), return an empty tensor of shape (B,C,0)
    if (GT_out == 0) {
        return torch::zeros({L, B, C, 0}, logits.options().dtype(torch::kFloat32));
    }

    // Launch reduction kernel only for the compacted GT_out entries
    auto out_accum = torch::zeros({2, L, B, C, GT_out}, logits.options().dtype(torch::kFloat32));
    {
        dim3 grid(L, B, C);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            reduce_pairwise_sigmoid_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value,
                decltype(W_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<uint8_t>(),
                    out_accum.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    static_cast<int32_t>(background_index),
                    GT_total,
                    GT_out,
                    B, L,
                    sigmoid_scale
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}), // C
            std::make_tuple(std::integral_constant<int, 256>{}),  // H
            std::make_tuple(std::integral_constant<int, 256>{}),  // W
            std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
            std::make_tuple(std::integral_constant<int, 1024>{}) // W_t
        );
        // W_t is not needed by reduce_loss_kernel, so we only pass needed dims
        const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    // Launch reduction kernel only for the compacted GT_out entries
    {
        dim3 grid(GT_total, C, L*B);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val) {
            reduce_pairwise_dice_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<uint8_t>(),
                    out_accum.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    GT_total,
                    GT_out,
                    B, L, smooth,
                    background_index,
                    dice_scale
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}), // C
            std::make_tuple(std::integral_constant<int, 256>{}),  // H
            std::make_tuple(std::integral_constant<int, 256>{}),  // W
            std::make_tuple(std::integral_constant<int, 1024>{}) // H_t
        );
        const auto runtime_dims = std::make_tuple(C, H, W, H_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    return out_accum;
}