#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"
#include "pw_sigmoid_dice.cuh"

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
    const float scale
);

// This kernel fuses the pairwise sigmoid cross-entropy (BCE) and Dice reductions.
// Computing both losses in a single pass avoids re-loading logits from global
// memory and lets us share expensive intermediate values such as the sigmoid
// probabilities.  The kernel first accumulates the per-logit BCE "base" term
// that is independent of the ground-truth mask and the total probability mass.
// It then iterates over each ground-truth mask to accumulate the remaining
// class-specific terms and writes both losses to the output tensor.
template <int Q, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK)
reduce_pairwise_sigmoid_dice_kernel(
    const float* __restrict__ logits,           // shape (L, B, Q, H, W)
    const uint8_t* __restrict__ counts,         // shape (B, GT_total, H, W)
    float* __restrict__ out,                    // shape (2, L, B, Q, GT_out)
    const int32_t* __restrict__ total_counts,   // shape (B, GT_total)
    const int background_index,
    const int GT_total,
    const int GT_out,
    const int B,
    const int L,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float gamma,
    const float alpha
) {
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 32;
    constexpr int NUM_WARPS = REDUCTION_THREADS_PER_BLOCK / 32;

    __shared__ float s_logits[TILE_H * TILE_W];
    __shared__ float s_warp_a[NUM_WARPS];
    __shared__ float s_warp_b[NUM_WARPS];

    const int l  = blockIdx.x;
    const int b  = blockIdx.y;
    const int ci = blockIdx.z;
    const int tid = threadIdx.x;

    const int stride_slice = L * B * Q * GT_out;
    const int base_offset = ((l * B + b) * Q + ci) * GT_out;
    float* out_bce  = out + base_offset;
    float* out_dice = out + stride_slice + base_offset;

    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);
    const float denom = static_cast<float>(H_t * W_t);
    const float alpha_pos = (alpha >= 0.0f) ? alpha : 1.0f;
    const float alpha_neg = (alpha >= 0.0f) ? (1.0f - alpha) : 1.0f;
    const bool use_gamma = (gamma > 0.0f);

    float thread_base = 0.f;
    float thread_p_total = 0.f;

    // Stage 1: iterate over the low-resolution logits to compute BCE terms that
    // do not depend on the ground-truth masks.  We also accumulate the total
    // sigmoid mass (p_total) that is later reused by every mask.
    for (int ti = 0; ti < H; ti += TILE_H) {
        for (int tj = 0; tj < W; tj += TILE_W) {
            for (int idx = tid; idx < TILE_H * TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                const int di = idx / TILE_W;
                const int dj = idx % TILE_W;
                const int i = ti + di;
                const int j = tj + dj;
                float Lij = 0.f;
                if (i < H && j < W) {
                    Lij = logits[(((l * B + b) * Q + ci) * H + i) * W + j];
                }
                s_logits[idx] = Lij;
            }
            __syncthreads();

            for (int idx = tid; idx < TILE_H * TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                const float Lij = s_logits[idx];
                const float maxL = Lij > 0.f ? Lij : 0.f;
                const float absL = fabsf(Lij);
                const float logexp = log1pf(__expf(-absL));
                const float ce_neg = logexp + maxL;
                const float p = 1.f / (1.f + __expf(-Lij));
                const float mod_neg = use_gamma ? powf(p, gamma) : 1.0f;
                const float coeff_neg = alpha_neg * mod_neg * ce_neg;
                thread_base += N2 * coeff_neg;
                thread_p_total += N2 * p;
            }
            __syncthreads();
        }
    }

    float base_sum = thread_base;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        base_sum += __shfl_down_sync(0xffffffff, base_sum, off);
    }
    if ((tid & 31) == 0) {
        s_warp_a[tid >> 5] = base_sum;
    }
    __syncthreads();
    for (int sred = NUM_WARPS >> 1; sred > 0; sred >>= 1) {
        if (tid < sred) {
            s_warp_a[tid] += s_warp_a[tid + sred];
        }
        __syncthreads();
    }
    base_sum = (tid == 0) ? s_warp_a[0] : 0.f;
    base_sum = __shfl_sync(0xffffffff, base_sum, 0);

    float p_total = thread_p_total;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        p_total += __shfl_down_sync(0xffffffff, p_total, off);
    }
    if ((tid & 31) == 0) {
        s_warp_a[tid >> 5] = p_total;
    }
    __syncthreads();
    for (int sred = NUM_WARPS >> 1; sred > 0; sred >>= 1) {
        if (tid < sred) {
            s_warp_a[tid] += s_warp_a[tid + sred];
        }
        __syncthreads();
    }
    p_total = (tid == 0) ? s_warp_a[0] : 0.f;
    p_total = __shfl_sync(0xffffffff, p_total, 0);

    // Stage 2: iterate over each ground-truth mask, reusing the cached logits
    // and shared sigmoid mass to finish the BCE and Dice computations.
    for (int out_gt_idx = 0; out_gt_idx < GT_out; ++out_gt_idx) {
        const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);
        const int32_t total_count = total_counts[b * GT_total + gt_actual];

        if (total_count == 0) {
            if (tid == 0) {
                out_bce[out_gt_idx] = INFINITY;
                out_dice[out_gt_idx] = INFINITY;
            }
            continue;
        }

        float thread_ce = 0.f;
        float thread_intersection = 0.f;

        // Pointer to the spatial counts for the current ground-truth mask at
        // the down-sampled (H, W) resolution.
        const uint8_t* gt_counts_base = counts + ((b * GT_total + gt_actual) * H) * W;

        for (int ti = 0; ti < H; ti += TILE_H) {
            for (int tj = 0; tj < W; tj += TILE_W) {
                for (int idx = tid; idx < TILE_H * TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                    const int di = idx / TILE_W;
                    const int dj = idx % TILE_W;
                    const int i = ti + di;
                    const int j = tj + dj;
                    float Lij = 0.f;
                    if (i < H && j < W) {
                        Lij = logits[(((l * B + b) * Q + ci) * H + i) * W + j];
                    }
                    s_logits[idx] = Lij;
                }
                __syncthreads();

                for (int idx = tid; idx < TILE_H * TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                    const int di = idx / TILE_W;
                    const int dj = idx % TILE_W;
                    const int i = ti + di;
                    const int j = tj + dj;
                    if (i < H && j < W) {
                        const float Lij = s_logits[idx];
                        const float fn = static_cast<float>(gt_counts_base[i * W + j]);
                        const float absL = fabsf(Lij);
                        const float maxL = Lij > 0.f ? Lij : 0.f;
                        const float maxNegL = (-Lij) > 0.f ? -Lij : 0.f;
                        const float logexp = log1pf(__expf(-absL));
                        const float ce_neg = logexp + maxL;
                        const float ce_pos = logexp + maxNegL;
                        const float p = 1.f / (1.f + __expf(-Lij));
                        const float one_minus = 1.f - p;
                        const float mod_neg = use_gamma ? powf(p, gamma) : 1.0f;
                        const float mod_pos = use_gamma ? powf(one_minus, gamma) : 1.0f;
                        const float coeff_neg = alpha_neg * mod_neg * ce_neg;
                        const float coeff_pos = alpha_pos * mod_pos * ce_pos;
                        const float coeff_delta = coeff_pos - coeff_neg;
                        thread_ce += coeff_delta * fn;
                        thread_intersection += p * fn;
                    }
                }
                __syncthreads();
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            thread_ce += __shfl_down_sync(0xffffffff, thread_ce, off);
            thread_intersection += __shfl_down_sync(0xffffffff, thread_intersection, off);
        }
        if ((tid & 31) == 0) {
            s_warp_a[tid >> 5] = thread_ce;
            s_warp_b[tid >> 5] = thread_intersection;
        }
        __syncthreads();
        for (int sred = NUM_WARPS >> 1; sred > 0; sred >>= 1) {
            if (tid < sred) {
                s_warp_a[tid] += s_warp_a[tid + sred];
                s_warp_b[tid] += s_warp_b[tid + sred];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float ce_val = (base_sum + s_warp_a[0]) / denom * sigmoid_scale;
            const float total_t = static_cast<float>(total_count);
            const float dice = 1.f - (2.f * s_warp_b[0] + smooth) / (p_total + total_t + smooth);
            out_bce[out_gt_idx] = ce_val;
            out_dice[out_gt_idx] = dice * dice_scale;
        }
        __syncthreads();
    }
}

// Host entry point that prepares intermediate buffers and launches the fused
// reduction kernel.  The returned tensor is shaped as (2, L, B, Q, GT_out)
// where the first slice holds the BCE loss and the second slice holds the
// Dice loss for each (level, batch, class, ground-truth) tuple.
torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT_total),  int64,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    const float cls_scale = 1.0f,
    int64_t background_index = -1,
    const float gamma = 0.0f,
    const float alpha = -1.0f
) {
    CHECK_INPUT(mask_logits);
    CHECK_INPUT(mask_targets);
    CHECK_INPUT(cls_logits);
    CHECK_INPUT(cls_targets);

    TORCH_CHECK(mask_logits.dim() == 5, "mask_logits must be (L,B,Q,H,W)");
    TORCH_CHECK(mask_targets.dim() == 3, "mask_targets must be (B,H_t,W_t)");
    TORCH_CHECK(cls_logits.dim()  == 4, "cls_logits must be (L,B,Q,C)");
    TORCH_CHECK(cls_targets.dim() == 2, "cls_targets must be (B,GT)");
    TORCH_CHECK(gamma >= 0.0f, "focal_gamma must be non-negative");
    TORCH_CHECK(alpha < 0.0f || (alpha >= 0.0f && alpha <= 1.0f),
        "focal_alpha must be in [0, 1] or negative to disable");

    const int L = mask_logits.size(0);
    const int B = mask_logits.size(1);
    const int Q = mask_logits.size(2);
    const int H = mask_logits.size(3);
    const int W = mask_logits.size(4);
    const int H_t = mask_targets.size(1);
    const int W_t = mask_targets.size(2);
    const int C = cls_logits.size(3);
    const int GT_total = cls_targets.size(1);

    // Intermediate tensor to store counts of GT_total labels per low-res block
    auto counts = torch::zeros({B, GT_total, H, W}, mask_logits.options().dtype(torch::kUInt8));

    // Launch count kernel (counts for ALL GT labels)
    {
        dim3 block(16,16);
        dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y, B);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value, decltype(W_val)::value,
                decltype(H_t_val)::value, decltype(W_t_val)::value>
                <<<grid, block>>>(
                    mask_targets.data_ptr<int64_t>(),
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

    // If no classes left to evaluate (edge case), return an empty tensor of shape (B,Q,0)
    if (GT_out == 0) {
        return torch::zeros({L, B, Q, 0}, mask_logits.options().dtype(torch::kFloat32));
    }

    auto out_accum = torch::zeros({3, L, B, Q, GT_out}, mask_logits.options().dtype(torch::kFloat32));
    {
        dim3 grid(L, B, Q);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            reduce_pairwise_sigmoid_dice_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value,
                decltype(W_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    mask_logits.data_ptr<float>(),
                    counts.data_ptr<uint8_t>(),
                    out_accum.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    static_cast<int32_t>(background_index),
                    GT_total,
                    GT_out,
                    B,
                    L,
                    smooth,
                    sigmoid_scale,
                    dice_scale,
                    gamma,
                    alpha
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}),  // Q
            std::make_tuple(std::integral_constant<int, 256>{}),  // H
            std::make_tuple(std::integral_constant<int, 256>{}),  // W
            std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
            std::make_tuple(std::integral_constant<int, 1024>{})  // W_t
        );
        const auto runtime_dims = std::make_tuple(Q, H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    {
        auto cls_out = out_accum.data_ptr<float>() + (2 * L * B * Q * GT_out);
        dim3 grid(L, B, Q);

        auto static_launcher = [&](auto C_val) {
            reduce_pairwise_label_kernel<decltype(C_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    cls_logits.data_ptr<float>(),
                    cls_targets.data_ptr<int64_t>(),
                    cls_out,
                    static_cast<int32_t>(background_index),
                    GT_total,
                    GT_out,
                    B,
                    Q,
                    L,
                    cls_scale
                );
        };
        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}) // C
        );
        const auto runtime_dims = std::make_tuple(C);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after class reduce kernel: ", cudaGetErrorString(err));

    return out_accum;
}
