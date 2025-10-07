#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK = 256;

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) sigmoid_cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    double* __restrict__ total_loss_sum,
    const int B
) {
    // Each CUDA block processes one (b, i, j) low-res block
    // Grid: dim.x = W (low-res width), dim.y = H (low-res height), dim.z = B (batch)
    // threadIdx.x loops over s*s pixels
    extern __shared__ int sh_counts[]; // Shared memory for per-class counts, size C

    int j = blockIdx.x;  // low-res x (0..W-1)
    int i = blockIdx.y;  // low-res y (0..H-1)
    int b = blockIdx.z;  // batch index

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

    // Each thread loops over several pixels if necessary
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

    // Each thread computes the loss for a subset of classes
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        // logits layout: (B, C, H, W)
        float L = logits[((b * C + ci) * H + i) * W + j];
        float n = (float) sh_counts[ci];
        float N2 = (float)s2;

        // Stable BCE-with-logits sum over the block:
        // loss_block = N2*max(L,0) - L*n + N2*log1p(exp(-|L|))
        float maxL = L > 0.0f ? L : 0.0f;
        float absL = fabsf(L);
        float logexp = log1pf(expf(-absL));
        double loss_block = static_cast<double>(N2 * maxL - L * n + N2 * logexp);

        // Atomically add the block's loss to the total sum
        atomicAdd(total_loss_sum, loss_block);
    }
}

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) sigmoid_cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float grad_out_scalar,
    float* __restrict__ grad_logits,
    const int B
) {
    // Grid: dim.x = W (low-res x), dim.y = H (low-res y), dim.z = B
    int j = blockIdx.x;
    int i = blockIdx.y;
    int b = blockIdx.z;

    // Parallelize across classes in threads
    int tid = threadIdx.x;

    const int s = H_t / H; // Stride
    const float N2 = (float)(s * s);

    // To calculate the gradient, we need to re-compute the counts for each block.
    // This is a trade-off to avoid storing the counts tensor from the forward pass.
    // Shared memory is used for efficient recounting.
    extern __shared__ int sh_counts[]; // Shared memory for per-class counts, size C

    // Initialize shared counts to zero
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        sh_counts[ci] = 0;
    }
    __syncthreads();

    // Re-compute counts for the current block
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
    
    // Base index for the current block
    int idx_base = ((b * C) * H + i) * W + j;
    float scale = grad_out_scalar / (B * C * H * W);

    // Each thread computes the gradient for a subset of classes
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        float L = logits[idx_base + ci * H * W];
        int32_t n = sh_counts[ci];

        // sigma = 1 / (1 + exp(-L))
        float sigma = 1.0f / (1.0f + expf(-L));
        // derivative: dLoss/dL = N2 * sigma - n
        float g = N2 * sigma - (float)n;
        
        // Apply scaling
        grad_logits[idx_base + ci * H * W] = g * scale;
    }
}

torch::Tensor sigmoid_cross_entropy_forward(
    const torch::Tensor& logits, 
    const torch::Tensor& targets
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);
    
    const int total_elements = B * C * H_t * W_t;
    if (total_elements == 0) return torch::tensor(0.0, logits.options());

    auto total_loss_sum_tensor = torch::zeros({1}, logits.options().dtype(torch::kFloat64));
    
    // Set grid dimensions based on the low-resolution output
    dim3 grid(W, H, B);
    
    // Shared memory size is C * sizeof(int32) for the counts
    const size_t shared_mem_size = C * sizeof(int32_t);

    auto static_launcher = [&](auto... Dims) {
        sigmoid_cross_entropy_forward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
            logits.data_ptr<float>(), targets.data_ptr<int64_t>(), total_loss_sum_tensor.data_ptr<double>(), B);
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 256>{}), // C
        std::make_tuple(std::integral_constant<int, 64>{}),  // H
        std::make_tuple(std::integral_constant<int, 64>{}),  // W
        std::make_tuple(std::integral_constant<int, 512>{}), // H_t
        std::make_tuple(std::integral_constant<int, 512>{})  // W_t
    );
    const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);

    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after forward kernel: ", cudaGetErrorString(err));
    
    return (total_loss_sum_tensor.to(torch::kFloat32) / total_elements).squeeze();
}

torch::Tensor sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out, 
    const torch::Tensor& logits, 
    const torch::Tensor& targets
) {
    
    CHECK_INPUT(grad_out);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    auto grad_logits = torch::empty_like(logits);
    const int total_elements = B * C * H * W;
    if (total_elements == 0) return grad_logits;

    const float grad_out_scalar = grad_out.item<float>();
    
    // Set grid dimensions based on the low-resolution output
    dim3 grid(W, H, B);
    const size_t shared_mem_size = C * sizeof(int32_t);

    auto static_launcher = [&](auto... Dims) {
        sigmoid_cross_entropy_backward_kernel<decltype(Dims)::value...><<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
            logits.data_ptr<float>(), targets.data_ptr<int64_t>(), grad_out_scalar, grad_logits.data_ptr<float>(), B);
    };
    
    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 256>{}), // C
        std::make_tuple(std::integral_constant<int, 64>{}),  // H
        std::make_tuple(std::integral_constant<int, 64>{}),  // W
        std::make_tuple(std::integral_constant<int, 512>{}), // H_t
        std::make_tuple(std::integral_constant<int, 512>{})  // W_t
    );
    const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);

    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after backward kernel: ", cudaGetErrorString(err));

    return grad_logits;
}