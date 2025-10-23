#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"

enum PairwiseLabelLossType : int32_t {
    kBinaryCrossEntropy      = 0,
    kBinaryCrossEntropyFocal = 1,
};

// Kernel: for each (l,b,q) reduce across C once to get the necessary statistics for
// the requested loss, then emit (optionally compacted) GT_out values.
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
    const int32_t loss_type,
    const float focal_alpha,
    const float focal_gamma
) {
    constexpr int NUM_WARPS = REDUCTION_THREADS_PER_BLOCK / 32;
    __shared__ float s_warp[NUM_WARPS];

    const int l   = blockIdx.x;  // layer
    const int b   = blockIdx.y;  // batch
    const int qid = blockIdx.z;  // query
    const int tid = threadIdx.x;

    const bool use_bce      = (loss_type == kBinaryCrossEntropy) || (loss_type == kBinaryCrossEntropyFocal);
    const bool use_bce_focal= (loss_type == kBinaryCrossEntropyFocal);

    const bool   has_alpha = focal_alpha >= 0.f;
    const float  alpha_clamped = has_alpha ? fminf(fmaxf(focal_alpha, 0.f), 1.f) : 1.f;
    const float  alpha_pos = use_bce_focal ? alpha_clamped : 1.f;
    const float  alpha_neg = use_bce_focal ? (has_alpha ? (1.f - alpha_clamped) : 1.f) : 1.f;

    float thread_sum_softplus = 0.f;
    float thread_sum_mod_softplus = 0.f;

    // Stride across C by blockDim.x; template C is compile-time for efficient looping
    for (int c = tid; c < C; c += REDUCTION_THREADS_PER_BLOCK) {
        const float z = logits[(((l * B + b) * Q + qid) * C) + c];
        if (use_bce) {
            const float maxL  = z > 0.f ? z : 0.f;
            const float absL  = fabsf(z);
            const float logex = log1pf(__expf(-absL));
            const float softplus_z = maxL + logex; // softplus(z)
            thread_sum_softplus += softplus_z;
            if (use_bce_focal) {
                const float sig = 1.f / (1.f + __expf(-z));
                thread_sum_mod_softplus += powf(sig, focal_gamma) * softplus_z;
            }
        }
    }

    float block_sum_softplus = 0.f;
    if (use_bce) {
        float sum_val = thread_sum_softplus;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, off);
        }
        if ((tid & 31) == 0) {
            s_warp[tid >> 5] = sum_val;
        }
        __syncthreads();
        for (int s = NUM_WARPS >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                s_warp[tid] += s_warp[tid + s];
            }
            __syncthreads();
        }
        block_sum_softplus = (tid == 0) ? s_warp[0] : 0.f;
        block_sum_softplus = __shfl_sync(0xffffffff, block_sum_softplus, 0);
    }

    float block_sum_mod_softplus = 0.f;
    if (use_bce_focal) {
        float sum_val = thread_sum_mod_softplus;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, off);
        }
        if ((tid & 31) == 0) {
            s_warp[tid >> 5] = sum_val;
        }
        __syncthreads();
        for (int s = NUM_WARPS >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                s_warp[tid] += s_warp[tid + s];
            }
            __syncthreads();
        }
        block_sum_mod_softplus = (tid == 0) ? s_warp[0] : 0.f;
        block_sum_mod_softplus = __shfl_sync(0xffffffff, block_sum_mod_softplus, 0);
    }

    const float invC = (C > 0) ? (1.f / static_cast<float>(C)) : 0.f;

    // For each output GT slot, write loss or +inf for padding ---
    // We only need one writer (tid==0) since (l,b,q,*) are independent
    if (tid == 0) {
        for (int out_gt_idx = 0; out_gt_idx < GT_out; ++out_gt_idx) {
            const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);
            const int64_t y64   = targets[b * GT_total + gt_actual];

            // Padding / invalid label -> +inf (do not apply scale, mirroring other kernel)
            if (y64 < 0 || y64 >= static_cast<int64_t>(C)) {
                out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = INFINITY;
                continue;
            }

            const int y = static_cast<int>(y64);
            const float z_pos = logits[(((l * B + b) * Q + qid) * C) + y];
            float v = 0.f;

            if (loss_type == kBinaryCrossEntropy) {
                v = (block_sum_softplus - z_pos) * invC;
            } else if (loss_type == kBinaryCrossEntropyFocal) {
                const float sig_pos = 1.f / (1.f + __expf(-z_pos));
                const float softplus_pos = log1pf(__expf(z_pos));
                const float softplus_neg = log1pf(__expf(-z_pos));
                const float mod_pos = powf(sig_pos, focal_gamma);
                const float mod_neg = powf(fmaxf(1.f - sig_pos, 0.f), focal_gamma);
                const float neg_except = block_sum_mod_softplus - mod_pos * softplus_pos;
                const float pos_term = softplus_neg * mod_neg;
                v = (alpha_pos * pos_term + alpha_neg * neg_except) * invC;
            } else {
                v = INFINITY;
            }

            out[(((l * B + b) * Q + qid) * GT_out) + out_gt_idx] = v * scale;
        }
    }
}

torch::Tensor pairwise_label_loss_forward(
    const torch::Tensor& logits,   // (L,B,Q,C), float
    const torch::Tensor& targets,  // (B,GT), int64 with -1 padding
    int64_t background_index = -1, // drop column targets[:, background_index]
    const float scale = 1.0f,
    int64_t loss_type = 0,
    const float focal_alpha = -1.0f,
    const float focal_gamma = 2.0f
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    TORCH_CHECK(logits.dim() == 4, "pairwise_label_loss_forward: logits must be (L,B,Q,C)");
    TORCH_CHECK(targets.dim() == 2, "pairwise_label_loss_forward: targets must be (B,GT)");

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
    TORCH_CHECK(loss_type >= 0 && loss_type <= 1, "pairwise_label_loss_forward: invalid loss type");

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
                static_cast<int32_t>(loss_type),
                focal_alpha,
                focal_gamma
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
