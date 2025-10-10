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
const int HIGH_RES_BLOCK = 16;
const int COUNTER_THREADS_PER_BLOCK = HIGH_RES_BLOCK*HIGH_RES_BLOCK;
const int REDUCTION_THREADS_PER_BLOCK = 256;

template <int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(COUNTER_THREADS_PER_BLOCK) count_labels_per_block_kernel(
    const int64_t* __restrict__ targets,
    int32_t* __restrict__ counts, // out: shape (B, GT_total, H, W)
    const int B,
    const int GT_total
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
    for (int idx = tid; idx < GT_total; idx += COUNTER_THREADS_PER_BLOCK) {
        sh_counts[idx] = 0;
    }
    __syncthreads();

    // Base corner of the corresponding high-resolution block
    int base_y = i * s;
    int base_x = j * s;

    // Parallel count of pixels per GT label within the block
    for (int idx = tid; idx < s2; idx += COUNTER_THREADS_PER_BLOCK) {
        int dy = idx / s;
        int dx = idx % s;
        int yy = base_y + dy;
        int xx = base_x + dx;
        if (yy < H_t && xx < W_t) {
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (lab >= 0 && lab < GT_total) {
                atomicAdd(&sh_counts[(int)lab], 1);
            }
        }
    }
    __syncthreads();

    // Write the counts from shared memory to the global counts tensor
    for (int gt = tid; gt < GT_total; gt += COUNTER_THREADS_PER_BLOCK) {
        counts[((b * GT_total + gt) * H + i) * W + j] = sh_counts[gt];
    }
}

template <int C, int H, int W, int H_t>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK) reduce_loss_kernel(
    const float* __restrict__ logits,           // shape (L, B, C, H, W)
    const int32_t* __restrict__ counts,         // shape (B, GT_total, H, W)
    float* __restrict__ out,                   // shape (L, B, C, GT_out)
    const int32_t* __restrict__ total_counts,   // shape (B, GT_total)
    const int32_t* __restrict__ gt_map,         // length GT_out: maps output index -> actual GT label
    const int GT_total,
    const int GT_out,
    const int B,
    const int L
) {
    extern __shared__ float s_block_loss[];

    const int out_gt_idx = blockIdx.x; // compacted output index (0..GT_out-1)
    const int ci = blockIdx.y;         // Logit class index
    const int flat_b_l = blockIdx.z;   // flat index combining layer and batch: 0 .. (B*L - 1)

    // Recover layer and batch indices
    const int b = flat_b_l % B;
    const int l = flat_b_l / B;

    // Map to the actual ground-truth label index
    const int gt_actual = gt_map[out_gt_idx];

    // If the total count for this ground truth label is 0, it's a zero-area mask.
    // Set the loss to infinity and return. Only one thread writes the output
    if (total_counts[b * GT_total + gt_actual] == 0) {
        if (threadIdx.x == 0) {
            out[((l*B + b)*C + ci) * GT_out + out_gt_idx] = INFINITY;
        }
        return;
    }

    const int tid = threadIdx.x;
    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);

    float thread_loss_sum = 0.0;

    // Each thread computes a partial sum of the loss over the HxW logit plane
    for (int idx = tid; idx < H * W; idx += REDUCTION_THREADS_PER_BLOCK) {
        int i = idx / W;
        int j = idx % W;

        float L = logits[(((l * B + b) * C + ci) * H + i) * W + j];
        int32_t n = counts[((b * GT_total + gt_actual) * H + i) * W + j];

        float maxL = L > 0.0f ? L : 0.0f;
        float absL = fabsf(L);
        float logexp = my_log1pf(__expf(-absL));

        // This follows: sum_{pixels in HR patch} [ max(L,0) - L*y + log(1+exp(-|L|)) ]
        // where y is the per-pixel binary GT. Here n is the count of y==1 in the HR patch,
        // and N2 is number of HR pixels in the patch.
        thread_loss_sum += (N2 * (maxL + logexp) - L * n);
    }

    // Warp-level-reduction: Sum thread-level losses within each warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { // 16 is half the warp size
        thread_loss_sum += __shfl_down_sync(0xffffffff, thread_loss_sum, offset);
    }

    // The first thread (lane 0) of each warp writes the warp's sum to shared memory.
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    if (lane_id == 0) {
        s_block_loss[warp_id] = thread_loss_sum;
    }

    // Store each thread's accumulated loss into shared memory
    //s_block_loss[tid] = thread_loss_sum;
    __syncthreads();

    // Perform the block-level reduction
    const int num_warps = REDUCTION_THREADS_PER_BLOCK / 32;
    for (int s_reduce = num_warps / 2; s_reduce > 0; s_reduce >>= 1) {
        if (tid < s_reduce) {
            s_block_loss[tid] += s_block_loss[tid + s_reduce];
        }
        __syncthreads();
    }

    // The first thread writes the final reduced result for the block, no atomic needed
    if (tid == 0) {
        out[((l * B + b) * C + ci) * GT_out + out_gt_idx] = s_block_loss[0];
    }
}

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const int64_t background_index = -1
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

    // Intermediate tensor to store counts for every possible GT label (including background)
    auto counts = torch::zeros({B, GT_total, H, W}, logits.options().dtype(torch::kInt32));

    // Launch count kernel (counts for ALL GT labels)
    {
        dim3 grid(W, H, B);
        const size_t shared_mem_size = GT_total * sizeof(int32_t);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value, decltype(W_val)::value,
                decltype(H_t_val)::value, decltype(W_t_val)::value>
                <<<grid, COUNTER_THREADS_PER_BLOCK, shared_mem_size>>>(
                    targets.data_ptr<int64_t>(),
                    counts.data_ptr<int32_t>(),
                    B, GT_total
                );
        };
        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 64>{}),  // H
            std::make_tuple(std::integral_constant<int, 64>{}),  // W
            std::make_tuple(std::integral_constant<int, 512>{}, std::integral_constant<int, 1024>{}), // H_t
            std::make_tuple(std::integral_constant<int, 512>{}, std::integral_constant<int, 1024>{})  // W_t
        );
        const auto runtime_dims = std::make_tuple(H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after count kernel: ", cudaGetErrorString(err));

    // Calculate the total number of pixels for each ground truth label (mask area)
    // This is used to mask 0-area masks with a loss of infinity.
    auto total_counts = counts.sum({2, 3}).to(torch::kInt32).contiguous(); // shape (B, GT_total)

    // Build gt_map: list of actual GT labels to evaluate (exclude background_index if requested and valid)
    std::vector<int32_t> host_gt_map;
    host_gt_map.reserve(GT_total);
    for (int i = 0; i < GT_total; ++i) {
        if (background_index >= 0 && i == static_cast<int>(background_index)) {
            continue;
        }
        host_gt_map.push_back(i);
    }
    const int GT_out = static_cast<int>(host_gt_map.size());
    // If no classes left to evaluate (edge case), return an empty tensor of shape (B,C,0)
    if (GT_out == 0) {
        return torch::zeros({L, B, C, 0}, logits.options().dtype(torch::kFloat32));
    }

    // Copy gt_map to device
    auto gt_map = torch::from_blob(host_gt_map.data(), {GT_out}, torch::kInt32).clone().to(logits.device());

    // Launch reduction kernel only for the compacted GT_out entries (mapping via gt_map)
    auto out_accum = torch::zeros({L, B, C, GT_out}, logits.options().dtype(torch::kFloat32));
    {
        dim3 grid(GT_out, C, B*L);
        const size_t shared_mem_size = (REDUCTION_THREADS_PER_BLOCK / 32) * sizeof(float);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val) {
            reduce_loss_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK, shared_mem_size>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<int32_t>(),
                    out_accum.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    gt_map.data_ptr<int32_t>(),
                    GT_total,
                    GT_out,
                    B, L
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 256>{}), // C
            std::make_tuple(std::integral_constant<int, 64>{}),  // H
            std::make_tuple(std::integral_constant<int, 64>{}),  // W
            std::make_tuple(std::integral_constant<int, 512>{}, std::integral_constant<int, 1024>{}) // H_t
        );
        // W_t is not needed by reduce_loss_kernel, so we only pass needed dims
        const auto runtime_dims = std::make_tuple(C, H, W, H_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    // Normalize by total high-res pixels per block (H_t * W_t)
    auto out_final = out_accum.to(logits.options().dtype(torch::kFloat32)) / static_cast<float>(H_t * W_t);

    return out_final;
}