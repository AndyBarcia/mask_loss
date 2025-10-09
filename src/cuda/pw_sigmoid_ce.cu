#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

// Process regions of 16x16, perfect for logits of shape
// 64x64 and ground truth of shape 1024x1024.
const int THREADS_PER_BLOCK = 16*16;

template <int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) count_labels_per_block_kernel(
    const int64_t* __restrict__ targets,
    int32_t* __restrict__ counts, // out: shape (B, GT, H, W)
    const int B,
    const int GT
) {
    extern __shared__ int32_t sh_mem_counts[];
    int* sh_counts = sh_mem_counts;

    int j = blockIdx.x;  // low-res x
    int i = blockIdx.y;  // low-res y
    int b = blockIdx.z;  // batch

    int tid = threadIdx.x;
    const int s = H_t / H;
    const int s2 = s * s;

    // Initialize shared memory counts for this block
    for (int idx = tid; idx < GT; idx += THREADS_PER_BLOCK) {
        sh_counts[idx] = 0;
    }
    __syncthreads();

    // Base corner of the corresponding high-resolution block
    int base_y = i * s;
    int base_x = j * s;

    // Parallel count of pixels per GT label within the block
    for (int idx = tid; idx < s2; idx += THREADS_PER_BLOCK) {
        int dy = idx / s;
        int dx = idx % s;
        int yy = base_y + dy;
        int xx = base_x + dx;
        if (yy < H_t && xx < W_t) {
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (lab >= 0 && lab < GT) {
                atomicAdd(&sh_counts[(int)lab], 1);
            }
        }
    }
    __syncthreads();

    // Write the counts from shared memory to the global counts tensor
    for (int gt = tid; gt < GT; gt += THREADS_PER_BLOCK) {
        counts[((b * GT + gt) * H + i) * W + j] = sh_counts[gt];
    }
}

template <int C, int H, int W, int H_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) reduce_loss_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ counts,
    double* __restrict__ out, // (B, C, GT)
    const int32_t* __restrict__ total_counts,  // (B,GT)
    const int B,
    const int GT
) {
    extern __shared__ double s_block_loss[];

    const int gt = blockIdx.x; // Ground Truth class index
    const int ci = blockIdx.y; // Logit class index
    const int b = blockIdx.z;  // Batch index

    // If the total count for this ground truth label is 0, it's a zero-area mask.
    // Set the loss to infinity and return. Only one thread writes the output
    if (total_counts[b * GT + gt] == 0) {
        if (threadIdx.x == 0) {
            out[(b * C + ci) * GT + gt] = INFINITY;
        }
        return;
    }

    const int tid = threadIdx.x;
    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);

    double thread_loss_sum = 0.0;

    // Each thread computes a partial sum of the loss over the HxW logit plane
    for (int idx = tid; idx < H * W; idx += THREADS_PER_BLOCK) {
        int i = idx / W;
        int j = idx % W;

        float L = logits[((b * C + ci) * H + i) * W + j];
        int32_t n = counts[((b * GT + gt) * H + i) * W + j];

        float maxL = L > 0.0f ? L : 0.0f;
        float absL = fabsf(L);
        float logexp = log1pf(expf(-absL));

        thread_loss_sum += static_cast<double>(N2 * maxL - L * n + N2 * logexp);
    }

    // Store each thread's accumulated loss into shared memory
    s_block_loss[tid] = thread_loss_sum;
    __syncthreads();

    // Perform the block-level reduction
    for (int s_reduce = THREADS_PER_BLOCK / 2; s_reduce > 0; s_reduce >>= 1) {
        if (tid < s_reduce) {
            s_block_loss[tid] += s_block_loss[tid + s_reduce];
        }
        __syncthreads();
    }

    // The first thread writes the final reduced result for the block, no atomic needed
    if (tid == 0) {
        out[(b * C + ci) * GT + gt] = s_block_loss[0];
    }
}

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
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

    // Automatically compute GT from the max value in targets
    torch::Tensor targets_max_tensor = targets.max();
    const int64_t GT = targets_max_tensor.item<int64_t>() + 1;

    // Intermediate tensor to store counts
    auto counts = torch::zeros({B, GT, H, W}, logits.options().dtype(torch::kInt32));

    // Launch count kernel
    {
        dim3 grid(W, H, B);
        const size_t shared_mem_size = GT * sizeof(int32_t);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value, decltype(W_val)::value,
                decltype(H_t_val)::value, decltype(W_t_val)::value>
                <<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
                    targets.data_ptr<int64_t>(),
                    counts.data_ptr<int32_t>(),
                    B, GT
                );
        };
        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 64>{}),  // H
            std::make_tuple(std::integral_constant<int, 64>{}),  // W
            std::make_tuple(std::integral_constant<int, 512>{}), // H_t
            std::make_tuple(std::integral_constant<int, 512>{})  // W_t
        );
        const auto runtime_dims = std::make_tuple(H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after count kernel: ", cudaGetErrorString(err));

    // Calculate the total number of pixels for each ground truth label (mask area)
    // This is used to mask 0-area masks with a loss of infinity.
    auto total_counts = counts.sum({2, 3}).to(torch::kInt32).contiguous();

    // Launch reduction kernel
    auto out_accum = torch::zeros({B, C, GT}, logits.options().dtype(torch::kFloat64));
    {
        dim3 grid(GT, C, B);
        const size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(double);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val) {
            reduce_loss_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value>
                <<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<int32_t>(),
                    out_accum.data_ptr<double>(),
                    total_counts.data_ptr<int32_t>(),
                    B, GT
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 256>{}), // C
            std::make_tuple(std::integral_constant<int, 64>{}),  // H
            std::make_tuple(std::integral_constant<int, 64>{}),  // W
            std::make_tuple(std::integral_constant<int, 512>{}) // H_t
        );
        // W_t is not needed by reduce_loss_kernel, so we only pass needed dims
        const auto runtime_dims = std::make_tuple(C, H, W, H_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    return out_accum.to(logits.options().dtype(torch::kFloat32)) / (H_t * W_t);
}