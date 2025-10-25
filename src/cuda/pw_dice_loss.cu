#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"

template <int C, int H, int W, int H_t>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK) reduce_pairwise_dice_kernel(
    const float* __restrict__ logits,               // shape (L, B, C, H, W)
    const uint8_t* __restrict__ counts,             // shape (B, GT_total, H, W)
    float* __restrict__ out,                        // shapeshape (L, B, C, GT_out)
    const int32_t* __restrict__ total_counts,       // shape (B, GT_total)
    const int GT_total,
    const int GT_out,
    const int B,
    const int L,
    const float smooth,
    const int background_index,
    const float scale,
    const float uncertainty_gamma,
    const float uncertainty_gamma_min
) {
    // Shared memory for performing the reduction of the three Dice components
    __shared__ float s_intersection[REDUCTION_THREADS_PER_BLOCK];
    __shared__ float s_p_sum[REDUCTION_THREADS_PER_BLOCK];
    __shared__ float s_t_sum[REDUCTION_THREADS_PER_BLOCK];

    const int out_gt_idx = blockIdx.x; // Ground Truth class index
    const int ci = blockIdx.y; // Logit class index
    const int b = blockIdx.z % B; // Batch index
    const int l = blockIdx.z / B; // Layer index
    const int tid = threadIdx.x;

    // Map to the actual ground-truth label index
    const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);

    // If the total count for this ground truth label is 0, it's a zero-area mask.
    // Set the loss to infinity and return, matching the Python implementation.
    if (total_counts[b * GT_total + gt_actual] == 0) {
        if (tid == 0) {
            // doScale don't matter; set to infinity
            out[((l * B + b) * C + ci) * GT_out + out_gt_idx] = INFINITY;
        }
        return;
    }

    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);

    const float eps = 1e-12f;
    const float inv_log2 = 1.4426950408889634f;
    const bool use_gamma = uncertainty_gamma != 0.0f;

    float thread_intersection_sum = 0.0f;
    float thread_p_sum = 0.0f;
    float thread_t_sum = 0.0f;

    // Each thread computes a partial sum of the Dice components over the HxW logit plane
    for (int idx = tid; idx < H * W; idx += REDUCTION_THREADS_PER_BLOCK) {
        int i = idx / W;
        int j = idx % W;

        float L = logits[((l * B + b) * C + ci) * H * W + i * W + j];
        float p = 1.0f / (1.0f + expf(-L));
        float p_clamped = fminf(fmaxf(p, eps), 1.0f - eps);
        float entropy = -(p_clamped * logf(p_clamped)
            + (1.0f - p_clamped) * logf(1.0f - p_clamped));
        entropy *= inv_log2;
        float weight = use_gamma ? powf(entropy, uncertainty_gamma) : 1.0f;
        weight = fminf(1.0f, fmaxf(weight, uncertainty_gamma_min));

        float n = static_cast<float>(counts[((b * GT_total + gt_actual) * H + i) * W + j]);

        thread_intersection_sum += weight * p * n;
        thread_p_sum += N2 * weight * p;
        thread_t_sum += weight * n;
    }

    // Store each thread's accumulated sums into shared memory
    s_intersection[tid] = thread_intersection_sum;
    s_p_sum[tid] = thread_p_sum;
    s_t_sum[tid] = thread_t_sum;
    __syncthreads();

    // Perform the block-level reduction for all three sums
    for (int s_reduce = REDUCTION_THREADS_PER_BLOCK / 2; s_reduce > 0; s_reduce >>= 1) {
        if (tid < s_reduce) {
            s_intersection[tid] += s_intersection[tid + s_reduce];
            s_p_sum[tid] += s_p_sum[tid + s_reduce];
            s_t_sum[tid] += s_t_sum[tid + s_reduce];
        }
        __syncthreads();
    }

    // The first thread computes the final Dice loss and writes the result for the block
    if (tid == 0) {
        float total_intersection = s_intersection[0];
        float total_p = s_p_sum[0];
        float total_t = s_t_sum[0];
        float dice = 1.0f - (2.0f * total_intersection + smooth) / (total_p + total_t + smooth);
        out[((l * B + b) * C + ci) * GT_out + out_gt_idx] = dice * scale;
    }
}


torch::Tensor pairwise_dice_loss_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    const float smooth,
    int64_t background_index = -1,
    const float scale = 1.0f,
    const float uncertainty_gamma = 1.0f,
    const float uncertainty_gamma_min = 0.05f
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
    auto out = torch::zeros({L, B, C, GT_out}, logits.options().dtype(torch::kFloat32));
    {
        dim3 grid(GT_out, C, L*B);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val) {
            reduce_pairwise_dice_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<uint8_t>(),
                    out.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    GT_total,
                    GT_out,
                    B, L, smooth,
                    background_index,
                    scale,
                    uncertainty_gamma,
                    uncertainty_gamma_min
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}), // C
            std::make_tuple(std::integral_constant<int, 256>{}), // H
            std::make_tuple(std::integral_constant<int, 256>{}), // W
            std::make_tuple(std::integral_constant<int, 1024>{}) // H_t
        );
        const auto runtime_dims = std::make_tuple(C, H, W, H_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    return out;
}