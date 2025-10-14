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
    const int B
) {
    // Each CUDA block processes one (b, i, j) low-res block
    // Grid: dim.x = W (low-res width), dim.y = H (low-res height), dim.z = B (batch)
    extern __shared__ int sh_counts[C];

    int j = blockIdx.x; // low-res x (0..W-1)
    int i = blockIdx.y; // low-res y (0..H-1)
    int b = blockIdx.z; // batch index

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
        // logits layout: (B, C, H, W)
        float L = logits[((b * C + ci) * H + i) * W + j];
        float p = 1.0f / (1.0f + expf(-L));
        float n_k = (float)sh_counts[ci];
        float N2 = (float)s2;

        float intersection = p * n_k;
        float p_sum = N2 * p;
        float t_sum = n_k;

        atomicAdd(&total_intersection_sum[b * C + ci], intersection);
        atomicAdd(&total_p_sum[b * C + ci], p_sum);
        atomicAdd(&total_t_sum[b * C + ci], t_sum);
    }
}

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) dice_loss_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ total_intersection_sum,
    const float* __restrict__ total_p_sum,
    const float* __restrict__ total_t_sum,
    const float grad_out_scalar,
    float* __restrict__ grad_logits,
    const int B,
    const float smooth
) {
    // Grid: dim.x = W (low-res x), dim.y = H (low-res y), dim.z = B
    int j = blockIdx.x;
    int i = blockIdx.y;
    int b = blockIdx.z;

    int tid = threadIdx.x;

    const int s = H_t / H;
    const float N2 = (float)(s * s);

    extern __shared__ int sh_counts[C];

    // Initialize shared counts to zero
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        sh_counts[ci] = 0;
    }
    __syncthreads();

    // Re-compute counts for the current block.
    int base_y = i * s;
    int base_x = j * s;
    for (int idx = tid; idx < N2; idx += THREADS_PER_BLOCK) {
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
    float scale = -grad_out_scalar;
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        float L = logits[((b * C + ci) * H + i) * W + j];
        float p = 1.0f / (1.0f + expf(-L));
        float n_k = (float)sh_counts[ci];

        float I = total_intersection_sum[b * C + ci];
        float P = total_p_sum[b * C + ci];
        float T = total_t_sum[b * C + ci];

        float denominator = P + T + smooth;
        float term1_numerator = 2.0 * n_k * (P + T + smooth);
        float term2_numerator = 2.0 * N2 * (2.0 * I + smooth);

        float d_dice_dp = (term1_numerator - term2_numerator) / (denominator * denominator);
        float dp_dL = p * (1.0 - p);

        grad_logits[((b * C + ci) * H + i) * W + j] = scale * d_dice_dp * dp_dL;
    }
}

std::vector<torch::Tensor> dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth,
    const float num_masks
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    if (logits.numel() == 0) {
        return {
            torch::tensor(0.0, logits.options()),
            torch::zeros({B, C}, logits.options()),
            torch::zeros({B, C}, logits.options()),
            torch::zeros({B, C}, logits.options())
        };
    }

    auto total_intersection_sum = torch::zeros({B, C}, logits.options());
    auto total_p_sum = torch::zeros({B, C}, logits.options());
    auto total_t_sum = torch::zeros({B, C}, logits.options());

    dim3 grid(W, H, B);

    auto static_launcher = [&](auto... Dims) {
        dice_loss_forward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK>>>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            total_intersection_sum.data_ptr<float>(),
            total_p_sum.data_ptr<float>(),
            total_t_sum.data_ptr<float>(),
            B
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
    auto loss = (1.0 - dice).sum() / num_masks;

    return {loss, total_intersection_sum, total_p_sum, total_t_sum};
}

torch::Tensor dice_loss_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& total_intersection_sum,
    const torch::Tensor& total_p_sum,
    const torch::Tensor& total_t_sum,
    const float smooth,
    const float num_masks
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(total_intersection_sum);
    CHECK_INPUT(total_p_sum);
    CHECK_INPUT(total_t_sum);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    auto grad_logits = torch::empty_like(logits);
    if (logits.numel() == 0) return grad_logits;

    const float grad_out_scalar = grad_out.item<float>() / num_masks;

    dim3 grid(W, H, B);

    auto static_launcher = [&](auto... Dims) {
        dice_loss_backward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK>>>(
            logits.data_ptr<float>(),
            targets.data_ptr<int64_t>(),
            total_intersection_sum.data_ptr<float>(),
            total_p_sum.data_ptr<float>(),
            total_t_sum.data_ptr<float>(),
            grad_out_scalar,
            grad_logits.data_ptr<float>(),
            B,
            smooth
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