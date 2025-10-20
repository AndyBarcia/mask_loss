#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK = 64;

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) dice_loss_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    float* __restrict__ total_intersection_sum,
    float* __restrict__ total_p_sum,
    float* __restrict__ total_t_sum,
    const int B,
    const int L
) {
    // Each CUDA block processes one (l, b, i, j) low-res block
    // Grid: dim.x = W (low-res width), dim.y = H (low-res height), dim.z = L * B (level * batch)
    __shared__ int sh_counts[C];

    int j = blockIdx.x; // low-res x (0..W-1)
    int i = blockIdx.y; // low-res y (0..H-1)
    int lb = blockIdx.z;
    int b = lb % B;     // batch index
    int l = lb / B;     // level index

    int tid = threadIdx.x;
    const int s = H_t / H; // Stride
    const int s2 = s * s;

    // Initialize shared counts to zero
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        sh_counts[ci] = 0;
    }
    __syncthreads();

    // Each thread block covers an s x s region of the high-resolution target tensor.
    // Top-left corner of the high-res block:
    int base_y = i * s;
    int base_x = j * s;

    // Each thread loops over several pixels if necessary to compute counts
    for (int idx = tid; idx < s2; idx += THREADS_PER_BLOCK) {
        int dy = idx / s;
        int dx = idx % s;
        int yy = base_y + dy;
        int xx = base_x + dx;
        if (yy < H_t && xx < W_t) {
            // targets layout: (B, H_t, W_t)
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (lab >= 0 && lab < C) {
                // Atomically accumulate counts in shared memory
                atomicAdd(&sh_counts[(int)lab], 1);
            }
        }
    }
    __syncthreads();

    // Each thread computes the intersection and sums for a subset of classes
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        // logits layout: (L, B, C, H, W)
        float logit = logits[((((l * B + b) * C + ci) * H + i) * W + j)];
        float p = 1.0f / (1.0f + expf(-logit));
        float n_k = (float)sh_counts[ci];
        float N2 = (float)s2;

        float intersection = p * n_k;
        float p_sum = N2 * p;
        float t_sum = n_k;

        int base_idx = ((l * B + b) * C) + ci;
        atomicAdd(&total_intersection_sum[base_idx], intersection);
        atomicAdd(&total_p_sum[base_idx], p_sum);
        atomicAdd(&total_t_sum[base_idx], t_sum);
    }
}

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) dice_loss_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ total_intersection_sum,
    const float* __restrict__ total_p_sum,
    const float* __restrict__ total_t_sum,
    const float* __restrict__ grad_out_levels,
    float* __restrict__ grad_logits,
    const int B,
    const int L,
    const float smooth,
    const float scale
) {
    // Grid: dim.x = W (low-res x), dim.y = H (low-res y), dim.z = L * B
    int j = blockIdx.x;
    int i = blockIdx.y;
    int lb = blockIdx.z;
    int b = lb % B;
    int l = lb / B;

    int tid = threadIdx.x;

    const int s = H_t / H;
    const int s2 = s * s;

    __shared__ int sh_counts[C];

    // Initialize shared counts to zero
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        sh_counts[ci] = 0;
    }
    __syncthreads();

    // Re-compute counts for the current block.
    int base_y = i * s;
    int base_x = j * s;
    for (int idx = tid; idx < s2; idx += THREADS_PER_BLOCK) {
        int dy = idx / s;
        int dx = idx % s;
        int yy = base_y + dy;
        int xx = base_x + dx;
        if (yy < H_t && xx < W_t) {
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (lab >= 0 && lab < C) {
                atomicAdd(&sh_counts[(int)lab], 1);
            }
        }
    }
    __syncthreads();

    // Each thread computes the gradient for a subset of classes
    float grad_level = grad_out_levels[l] * scale;
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        float logit = logits[((((l * B + b) * C + ci) * H + i) * W + j)];
        float p = 1.0f / (1.0f + expf(-logit));
        float n_k = (float)sh_counts[ci];
        const float N2 = (float) s2;

        int base_idx = ((l * B + b) * C) + ci;
        float I = total_intersection_sum[base_idx];
        float P = total_p_sum[base_idx];
        float T = total_t_sum[base_idx];

        float denominator = P + T + smooth;
        float term1_numerator = 2.0 * n_k * (P + T + smooth);
        float term2_numerator = N2 * (2.0 * I + smooth);

        float d_dice_dp = (term1_numerator - term2_numerator) / (denominator * denominator);
        float dp_dL = p * (1.0 - p);

        grad_logits[((((l * B + b) * C + ci) * H + i) * W + j)] = grad_level * d_dice_dp * dp_dL;
    }
}

std::vector<torch::Tensor> dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    const float num_masks,
    const float scale
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

    if (logits.numel() == 0) {
        return {
            torch::zeros({L}, logits.options()),
            torch::zeros({L, B, C}, logits.options()),
            torch::zeros({L, B, C}, logits.options()),
            torch::zeros({L, B, C}, logits.options())
        };
    }

    auto total_intersection_sum = torch::zeros({L, B, C}, logits.options());
    auto total_p_sum = torch::zeros({L, B, C}, logits.options());
    auto total_t_sum = torch::zeros({L, B, C}, logits.options());

    dim3 grid(W, H, L * B);

    auto static_launcher = [&](auto... Dims) {
        dice_loss_forward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK>>>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            total_intersection_sum.data_ptr<float>(),
            total_p_sum.data_ptr<float>(),
            total_t_sum.data_ptr<float>(),
            B,
            L
        );
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 133>{}), // C
        std::make_tuple(std::integral_constant<int, 256>{}),  // H
        std::make_tuple(std::integral_constant<int, 256>{}),  // W
        std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
        std::make_tuple(std::integral_constant<int, 1024>{})  // W_t
    );
    const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);

    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after forward kernel: ", cudaGetErrorString(err));

    auto dice = (2.0 * total_intersection_sum + smooth) / (total_p_sum + total_t_sum + smooth);
    float norm = num_masks / static_cast<float>(L);
    auto loss_per_level = (1.0 - dice).reshape({L, -1}).sum(1) / norm;

    loss_per_level.mul_(scale);
    return {loss_per_level.contiguous(), total_intersection_sum, total_p_sum, total_t_sum};
}

torch::Tensor dice_loss_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& total_intersection_sum,
    const torch::Tensor& total_p_sum,
    const torch::Tensor& total_t_sum,
    const float smooth,
    const float num_masks,
    const float scale
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(total_intersection_sum);
    CHECK_INPUT(total_p_sum);
    CHECK_INPUT(total_t_sum);

    const int L = logits.size(0);
    const int B = logits.size(1);
    const int C = logits.size(2);
    const int H = logits.size(3);
    const int W = logits.size(4);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    auto grad_logits = torch::empty_like(logits);
    if (logits.numel() == 0) return grad_logits;

    auto grad_out_contig = grad_out.contiguous();
    float norm = num_masks / static_cast<float>(L);
    auto grad_out_levels = (-grad_out_contig / norm).to(torch::kFloat32);

    dim3 grid(W, H, L * B);

    auto static_launcher = [&](auto... Dims) {
        dice_loss_backward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK>>>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            total_intersection_sum.data_ptr<float>(),
            total_p_sum.data_ptr<float>(),
            total_t_sum.data_ptr<float>(),
            grad_out_levels.data_ptr<float>(),
            grad_logits.data_ptr<float>(),
            B,
            L,
            smooth,
            scale
        );
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 133>{}), // C
        std::make_tuple(std::integral_constant<int, 256>{}),  // H
        std::make_tuple(std::integral_constant<int, 256>{}),  // W
        std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
        std::make_tuple(std::integral_constant<int, 1024>{})  // W_t
    );
    const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);

    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after backward kernel: ", cudaGetErrorString(err));

    return grad_logits;
}