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
__global__ void __launch_bounds__(THREADS_PER_BLOCK) reduce_pairwise_dice_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ counts,
    float* __restrict__ out, // out: shape (B, C, GT)
    const int32_t* __restrict__ total_counts, // in: shape (B, GT)
    const int B,
    const int GT,
    const float smooth
) {
    // Shared memory for performing the reduction of the three Dice components
    extern __shared__ float sh_mem[];
    float* s_intersection = sh_mem;
    float* s_p_sum = s_intersection + THREADS_PER_BLOCK;
    float* s_t_sum = s_p_sum + THREADS_PER_BLOCK;

    const int gt = blockIdx.x; // Ground Truth class index
    const int ci = blockIdx.y; // Logit class index
    const int b = blockIdx.z;  // Batch index

    // If the total count for this ground truth label is 0, it's a zero-area mask.
    // Set the loss to infinity and return, matching the Python implementation.
    if (total_counts[b * GT + gt] == 0) {
        if (threadIdx.x == 0) {
            out[(b * C + ci) * GT + gt] = INFINITY;
        }
        return;
    }

    const int tid = threadIdx.x;
    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);

    float thread_intersection_sum = 0.0f;
    float thread_p_sum = 0.0f;
    float thread_t_sum = 0.0f;

    // Each thread computes a partial sum of the Dice components over the HxW logit plane
    for (int idx = tid; idx < H * W; idx += THREADS_PER_BLOCK) {
        int i = idx / W;
        int j = idx % W;

        float L = logits[((b * C + ci) * H + i) * W + j];
        float p = 1.0f / (1.0f + expf(-L));
        float n = static_cast<float>(counts[((b * GT + gt) * H + i) * W + j]);

        thread_intersection_sum += p * n;
        thread_p_sum += N2 * p;
        thread_t_sum += n;
    }

    // Store each thread's accumulated sums into shared memory
    s_intersection[tid] = thread_intersection_sum;
    s_p_sum[tid] = thread_p_sum;
    s_t_sum[tid] = thread_t_sum;
    __syncthreads();

    // Perform the block-level reduction for all three sums
    for (int s_reduce = THREADS_PER_BLOCK / 2; s_reduce > 0; s_reduce >>= 1) {
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
        float dice = (2.0f * total_intersection + smooth) / (total_p + total_t + smooth);
        out[(b * C + ci) * GT + gt] = 1.0f - dice;
    }
}


std::vector<torch::Tensor> pairwise_dice_loss_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const float smooth
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

    // Intermediate tensor to store counts of GT labels per low-res block
    auto counts = torch::zeros({B, GT, H, W}, logits.options().dtype(torch::kInt32));

    // Kernel 1: Count Labels
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
    auto out_accum = torch::zeros({B, C, GT}, logits.options());

    // Kernel 2: Reduce and Compute Pairwise Dice Loss
    {
        dim3 grid(GT, C, B);
        // Shared memory for 3 float arrays used in reduction
        const size_t shared_mem_size = 3 * THREADS_PER_BLOCK * sizeof(float);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val) {
            reduce_pairwise_dice_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value>
                <<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<int32_t>(),
                    out_accum.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    B, GT, smooth
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 256>{}), // C
            std::make_tuple(std::integral_constant<int, 64>{}),  // H
            std::make_tuple(std::integral_constant<int, 64>{}),  // W
            std::make_tuple(std::integral_constant<int, 512>{}) // H_t
        );
        const auto runtime_dims = std::make_tuple(C, H, W, H_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    auto out_float = out_accum.to(logits.options().dtype(torch::kFloat32));

    std::vector<torch::Tensor> final_tensors;
    final_tensors.reserve(B);

    for (int b = 0; b < B; ++b) {
        torch::Tensor present_classes = torch::where(total_counts[b] > 0)[0];
        if (present_classes.numel() > 0) {
            final_tensors.push_back(out_float[b].index_select(1, present_classes));
        } else {
            final_tensors.push_back(torch::empty({C, 0}, out_float.options()));
        }
    }

    return final_tensors;
}