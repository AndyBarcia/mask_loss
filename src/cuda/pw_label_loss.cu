#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"

// Kernel: for each (l,b,q) reduce across C once to get sum_neg,
// then emit (optionally compacted) GT_out values using the one-hot identity:
// BCE(one-hot(y)) = sum_c neg(z_c) - z_y, where
// neg(z) = max(z,0) + log1p(exp(-|z|))
template <int C>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK)
reduce_pairwise_label_kernel(
    const float* __restrict__ logits,      // (L, B, Q, C)
    const int64_t* __restrict__ targets,   // (B, GT_total)
    float* __restrict__ out,               // (L, B, Q, GT_out)
    const int32_t background_index,        // fixed column to drop; set to GT_total if none
    const int32_t GT_total,                // number of GT slots (columns in targets)
    const int32_t GT_out,                  // GT_total - (background dropped ? 1 : 0)
    const int32_t B,
    const int32_t Q,
    const int32_t L,
    const float scale,
    const float gamma,
    const float alpha
) {
    constexpr int NUM_WARPS = REDUCTION_THREADS_PER_BLOCK / 32;
    __shared__ float s_warp[NUM_WARPS];

    const int l   = blockIdx.x;  // layer
    const int b   = blockIdx.y;  // batch
    const int qid = blockIdx.z;  // query
    const int tid = threadIdx.x;

    // Reduce across C to get sum_neg(l,b,q)
    float thread_sum = 0.f;
    const float alpha_pos = (alpha >= 0.0f) ? alpha : 1.0f;
    const float alpha_neg = (alpha >= 0.0f) ? (1.0f - alpha) : 1.0f;
    const bool use_gamma = (gamma > 0.0f);

    // Stride across C by blockDim.x; template C is compile-time for efficient looping
    for (int c = tid; c < C; c += REDUCTION_THREADS_PER_BLOCK) {
        const float z = logits[(((l * B + b) * Q + qid) * C) + c];
        const float maxL  = z > 0.f ? z : 0.f;
        const float absL  = fabsf(z);
        const float logex = log1pf(__expf(-absL));
        const float ce_neg = logex + maxL;
        const float sigma  = 1.0f / (1.0f + __expf(-z));
        const float mod_neg = use_gamma ? powf(sigma, gamma) : 1.0f;
        thread_sum += alpha_neg * mod_neg * ce_neg;
    }

    // Warp reduce to a single value per block
    float base_sum = thread_sum;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        base_sum += __shfl_down_sync(0xffffffff, base_sum, off);
    }
    if ((tid & 31) == 0) s_warp[tid >> 5] = base_sum;
    __syncthreads();
    for (int s = NUM_WARPS >> 1; s > 0; s >>= 1) {
        if (tid < s) s_warp[tid] += s_warp[tid + s];
        __syncthreads();
    }
    base_sum = (tid == 0) ? s_warp[0] : 0.f;
    // Broadcast to all threads in warp 0 (for symmetry; only tid==0 will write outputs)
    base_sum = __shfl_sync(0xffffffff, base_sum, 0);

    // For each output GT slot, write loss or +inf for padding ---
    // We only need one writer (tid==0) since (l,b,q,*) are independent
    if (tid == 0) {
        const float invC = (C > 0) ? (1.f / static_cast<float>(C)) : 0.f;

        for (int out_gt_idx = 0; out_gt_idx < GT_out; ++out_gt_idx) {
            const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);
            const int64_t y64   = targets[b * GT_total + gt_actual];

            // Padding / invalid label -> +inf (do not apply scale, mirroring other kernel)
            if (y64 < 0 || y64 >= static_cast<int64_t>(C)) {
                out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = INFINITY;
                continue;
            }

            const int y = static_cast<int>(y64);
            const float z_val = logits[(((l * B + b) * Q + qid) * C) + y];
            const float absL  = fabsf(z_val);
            const float maxL  = z_val > 0.f ? z_val : 0.f;
            const float maxNeg = (-z_val) > 0.f ? -z_val : 0.f;
            const float logex = log1pf(__expf(-absL));
            const float ce_neg = logex + maxL;
            const float ce_pos = logex + maxNeg;
            const float sigma  = 1.0f / (1.0f + __expf(-z_val));
            const float one_minus = 1.0f - sigma;
            const float mod_neg = use_gamma ? powf(sigma, gamma) : 1.0f;
            const float mod_pos = use_gamma ? powf(one_minus, gamma) : 1.0f;
            const float delta = alpha_pos * mod_pos * ce_pos - alpha_neg * mod_neg * ce_neg;
            const float v = (base_sum + delta) * invC;
            out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = v * scale;
        }
    }
}

torch::Tensor pairwise_label_loss_forward(
    const torch::Tensor& logits,   // (L,B,Q,C), float
    const torch::Tensor& targets,  // (B,GT), int64 with -1 padding
    int64_t background_index = -1, // drop column targets[:, background_index]
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    TORCH_CHECK(logits.dim() == 4, "pairwise_label_loss_forward: logits must be (L,B,Q,C)");
    TORCH_CHECK(targets.dim() == 2, "pairwise_label_loss_forward: targets must be (B,GT)");
    TORCH_CHECK(gamma >= 0.0f, "pairwise_label_loss_forward: focal_gamma must be non-negative");
    TORCH_CHECK(alpha < 0.0f || (alpha >= 0.0f && alpha <= 1.0f),
                "pairwise_label_loss_forward: focal_alpha must be in [0,1] or negative to disable");

    const int L  = static_cast<int>(logits.size(0));
    const int B  = static_cast<int>(logits.size(1));
    const int Q  = static_cast<int>(logits.size(2));
    const int C  = static_cast<int>(logits.size(3));
    const int GT_total = static_cast<int>(targets.size(1));

    TORCH_CHECK(B == targets.size(0), "pairwise_label_loss_forward: batch size mismatch between logits and targets");

    // Determine whether to drop a fixed GT column across the batch
    const bool drop_bg_col = (background_index >= 0 && background_index < GT_total);
    if (background_index < 0) {
        // Set to an invalid index so device-side MAP_OUT_TO_ACTUAL is a no-op
        background_index = GT_total;
    }
    const int GT_out = GT_total - (drop_bg_col ? 1 : 0);

    // Edge case: nothing to compute
    if (GT_out == 0) {
        return torch::zeros({L, B, Q, 0}, logits.options().dtype(torch::kFloat32));
    }

    // Ensure dtype/layout
    auto logits_f = logits.contiguous().to(torch::kFloat32);
    auto targets_i64 = targets.contiguous(); // keep int64

    // Allocate output
    auto out = torch::empty({L, B, Q, GT_out}, logits.options().dtype(torch::kFloat32));

    // Launch kernel: one block per (l,b,q), reduce across C
    dim3 grid(L, B, Q);

    auto static_launcher = [&](auto C_val) {
        reduce_pairwise_label_kernel<decltype(C_val)::value>
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
                scale,
                gamma,
                alpha
            );
    };

    // Template-dispatch over C for performance (matches your style)
    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 128>{}) // C
    );
    const auto runtime_dims = std::make_tuple(C);
    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA error in pairwise_label_loss_forward kernel: ",
                cudaGetErrorString(err));

    return out;
}
