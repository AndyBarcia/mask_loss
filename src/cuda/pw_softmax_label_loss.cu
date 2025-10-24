#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"

template <int C>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK)
reduce_pairwise_softmax_label_kernel(
    const float* __restrict__ logits,      // (L, B, Q, C), float
    const int64_t* __restrict__ targets,   // (B, GT_total), int64
    float* __restrict__ out,               // (L, B, Q, GT_out), float
    const int32_t background_index,        // column to drop; GT_total if none
    const int32_t GT_total,                // columns in targets
    const int32_t GT_out,                  // GT_total - (dropped ? 1 : 0)
    const int32_t B,
    const int32_t Q,
    const int32_t L,
    const float scale
) {
    constexpr int NUM_WARPS = REDUCTION_THREADS_PER_BLOCK / 32;
    __shared__ float s_warp[NUM_WARPS];

    const int l   = blockIdx.x;  // layer index
    const int b   = blockIdx.y;  // batch index
    const int qid = blockIdx.z;  // query index
    const int tid = threadIdx.x;

    // Pass 1: compute maximum logit for (l, b, q)
    float thread_max = -INFINITY;
    for (int c = tid; c < C; c += REDUCTION_THREADS_PER_BLOCK) {
        const float z = logits[(((l * B + b) * Q + qid) * C) + c];
        thread_max = fmaxf(thread_max, z);
    }

    float block_max = thread_max;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, block_max, off);
        block_max = fmaxf(block_max, other);
    }
    if ((tid & 31) == 0) {
        s_warp[tid >> 5] = block_max;
    }
    __syncthreads();
    for (int s = NUM_WARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_warp[tid] = fmaxf(s_warp[tid], s_warp[tid + s]);
        }
        __syncthreads();
    }
    const float max_val = s_warp[0];
    __syncthreads();

    // Pass 2: compute log-sum-exp denominator
    float thread_sum = 0.f;
    for (int c = tid; c < C; c += REDUCTION_THREADS_PER_BLOCK) {
        const float z = logits[(((l * B + b) * Q + qid) * C) + c];
        thread_sum += __expf(z - max_val);
    }

    float block_sum = thread_sum;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        block_sum += __shfl_down_sync(0xffffffff, block_sum, off);
    }
    if ((tid & 31) == 0) {
        s_warp[tid >> 5] = block_sum;
    }
    __syncthreads();
    for (int s = NUM_WARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_warp[tid] += s_warp[tid + s];
        }
        __syncthreads();
    }
    const float sum_val = s_warp[0];

    if (tid == 0) {
        const float safe_sum = fmaxf(sum_val, 1e-20f);
        const float log_denom = logf(safe_sum) + max_val;

        for (int out_gt_idx = 0; out_gt_idx < GT_out; ++out_gt_idx) {
            const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);
            const int64_t y64   = targets[b * GT_total + gt_actual];

            if (y64 < 0 || y64 >= static_cast<int64_t>(C)) {
                out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = INFINITY;
                continue;
            }

            const int y = static_cast<int>(y64);
            const float z_val = logits[(((l * B + b) * Q + qid) * C) + y];
            const float loss = (log_denom - z_val) * scale;
            out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = loss;
        }
    }
}

torch::Tensor pairwise_softmax_label_loss_forward(
    const torch::Tensor& logits,   // (L,B,Q,C), float
    const torch::Tensor& targets,  // (B,GT_total), int64 with -1 padding
    int64_t background_index = -1,
    const float scale = 1.0f
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    TORCH_CHECK(logits.dim() == 4, "pairwise_softmax_label_loss_forward: logits must be (L,B,Q,C)");
    TORCH_CHECK(targets.dim() == 2, "pairwise_softmax_label_loss_forward: targets must be (B,GT)");

    const int L  = static_cast<int>(logits.size(0));
    const int B  = static_cast<int>(logits.size(1));
    const int Q  = static_cast<int>(logits.size(2));
    const int C  = static_cast<int>(logits.size(3));
    const int GT_total = static_cast<int>(targets.size(1));

    TORCH_CHECK(B == targets.size(0), "pairwise_softmax_label_loss_forward: batch size mismatch between logits and targets");
    TORCH_CHECK(C > 0, "pairwise_softmax_label_loss_forward: class dimension must be positive");

    bool drop_bg_col = false;
    if (background_index >= 0 && background_index < GT_total) {
        drop_bg_col = true;
    } else {
        background_index = GT_total;
    }
    const int GT_out = GT_total - (drop_bg_col ? 1 : 0);

    if (GT_out == 0) {
        return torch::zeros({L, B, Q, 0}, logits.options().dtype(torch::kFloat32));
    }

    auto logits_f = logits.contiguous().to(torch::kFloat32);
    auto targets_i64 = targets.contiguous();

    auto out = torch::empty({L, B, Q, GT_out}, logits.options().dtype(torch::kFloat32));

    dim3 grid(L, B, Q);

    auto static_launcher = [&](auto C_val) {
        reduce_pairwise_softmax_label_kernel<decltype(C_val)::value>
            <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                logits_f.data_ptr<float>(),
                targets_i64.data_ptr<int64_t>(),
                out.data_ptr<float>(),
                static_cast<int32_t>(background_index),
                static_cast<int32_t>(GT_total),
                static_cast<int32_t>(GT_out),
                static_cast<int32_t>(B),
                static_cast<int32_t>(Q),
                static_cast<int32_t>(L),
                scale
            );
    };

    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 128>{})
    );
    const auto runtime_dims = std::make_tuple(C);
    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error in pairwise_softmax_label_loss_forward kernel: ",
                cudaGetErrorString(err));

    return out;
}

