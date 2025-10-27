#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"
#include "utils.cuh"

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK) reduce_pairwise_sigmoid_kernel(
    const float* __restrict__ logits,           // shape (L, B, C, H, W)
    const uint8_t* __restrict__ counts,         // shape (B, GT_total, H, W)
    float* __restrict__ out,                    // shape (L, B, C, GT_out)
    const int32_t* __restrict__ total_counts,   // shape (B, GT_total)
    const int background_index,
    const int GT_total,
    const int GT_out,
    const int B,
    const int L,
    const float scale,
    const float gamma,
    const float alpha,
    const float uncertainty_gamma,
    const float uncertainty_gamma_min,
    const bool normalize_uncertainty
) {
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 32;
    constexpr int NUM_WARPS = REDUCTION_THREADS_PER_BLOCK / 32;
    __shared__ float s_logits[TILE_H * TILE_W];
    __shared__ float s_warp[NUM_WARPS];

    const int l  = blockIdx.x;
    const int b  = blockIdx.y;
    const int ci = blockIdx.z;
    const int tid = threadIdx.x;

    const int s = H_t / H;
    const float N2 = static_cast<float>(s * s);
    const float alpha_pos = (alpha >= 0.0f) ? alpha : 1.0f;
    const float alpha_neg = (alpha >= 0.0f) ? (1.0f - alpha) : 1.0f;
    const bool use_uncertainty = uncertainty_gamma != 0.0f;
    const float inv_log2 = 1.4426950408889634f;
    const float eps = 1e-12f;
    const float num_pixels = static_cast<float>(H_t * W_t);

    // Compute base loss, independent of GT label
    float thread_base = 0.f;
    float thread_weight_sum = 0.f;

    for (int ti = 0; ti < H; ti += TILE_H) {
        for (int tj = 0; tj < W; tj += TILE_W) {
            // load tile logits to SMEM
            for (int idx = tid; idx < TILE_H*TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                int di = idx / TILE_W, dj = idx % TILE_W;
                int i = ti + di, j = tj + dj;
                float Lij = 0.f;
                if (i < H && j < W) {
                    Lij = logits[(((l*B + b)*C + ci)*H + i)*W + j];
                }
                s_logits[idx] = Lij;
            }
            __syncthreads();

            // Base contribution only
            for (int idx = tid; idx < TILE_H*TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                float Lij = s_logits[idx];
                float absL = fabsf(Lij);
                float maxL = Lij > 0.f ? Lij : 0.f;
                float logexp = log1pf(__expf(-absL));
                float ce_neg = logexp + maxL;
                float sigma = 1.0f / (1.0f + __expf(-Lij));
                float p_clamped = fminf(fmaxf(sigma, eps), 1.0f - eps);
                float entropy = -(p_clamped * logf(p_clamped)
                    + (1.0f - p_clamped) * logf(1.0f - p_clamped));
                entropy *= inv_log2;
                float weight = use_uncertainty ? powf(entropy, uncertainty_gamma) : 1.0f;
                weight = fminf(1.0f, fmaxf(weight, uncertainty_gamma_min));
                float mod_neg = (gamma > 0.0f) ? powf(sigma, gamma) : 1.0f;
                float coeff_neg = alpha_neg * mod_neg * ce_neg;
                thread_base += N2 * weight * coeff_neg;
                thread_weight_sum += N2 * weight;
            }
            __syncthreads();
        }
    }

    // Reduce to base_sum
    float base_sum = thread_base;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        base_sum += __shfl_down_sync(0xffffffff, base_sum, off);
    if ((tid & 31) == 0) s_warp[tid >> 5] = base_sum;
    __syncthreads();
    for (int sred = NUM_WARPS>>1; sred > 0; sred >>= 1) {
        if (tid < sred) s_warp[tid] += s_warp[tid + sred];
        __syncthreads();
    }
    base_sum = (tid == 0) ? s_warp[0] : 0.f;
    // Broadcast within warp 0
    base_sum = __shfl_sync(0xffffffff, base_sum, 0);

    float weight_total = thread_weight_sum;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        weight_total += __shfl_down_sync(0xffffffff, weight_total, off);
    if ((tid & 31) == 0) s_warp[tid >> 5] = weight_total;
    __syncthreads();
    for (int sred = NUM_WARPS>>1; sred > 0; sred >>= 1) {
        if (tid < sred) s_warp[tid] += s_warp[tid + sred];
        __syncthreads();
    }
    weight_total = (tid == 0) ? s_warp[0] : 0.f;
    weight_total = __shfl_sync(0xffffffff, weight_total, 0);
    if (!normalize_uncertainty) {
        weight_total = num_pixels;
    }
    weight_total = fmaxf(weight_total, eps);

    // Compute GT-dependent terms
    for (int out_gt_idx = 0; out_gt_idx < GT_out; ++out_gt_idx) {
        const int gt_actual = MAP_OUT_TO_ACTUAL(out_gt_idx, background_index);
        if (total_counts[b*GT_total + gt_actual] == 0) {
            // doScale don't matter; set to infinity
            if (tid == 0)
                out[((l*B + b)*C + ci)*GT_out + out_gt_idx] = INFINITY;
            continue;
        }

        float thread_gt = 0.f;

        for (int ti = 0; ti < H; ti += TILE_H) {
            for (int tj = 0; tj < W; tj += TILE_W) {
                // load logits tile again (cheap; no log1pf)
                for (int idx = tid; idx < TILE_H*TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                    int di = idx / TILE_W, dj = idx % TILE_W;
                    int i = ti + di, j = tj + dj;
                    float Lij = 0.f;
                    if (i < H && j < W) {
                        Lij = logits[(((l*B + b)*C + ci)*H + i)*W + j];
                    }
                    s_logits[idx] = Lij; // reuse SMEM to coalesce loads
                }
                __syncthreads();

                const uint8_t* gt_counts_base = counts + ( (b*GT_total + gt_actual)*H )*W;
                for (int idx = tid; idx < TILE_H*TILE_W; idx += REDUCTION_THREADS_PER_BLOCK) {
                    int di = idx / TILE_W, dj = idx % TILE_W;
                    int i = ti + di, j = tj + dj;
                    if (i < H && j < W) {
                        uint8_t n = gt_counts_base[i*W + j];
                        float Lij = s_logits[idx];
                        float absL = fabsf(Lij);
                        float maxL = Lij > 0.f ? Lij : 0.f;
                        float maxNegL = (-Lij) > 0.f ? -Lij : 0.f;
                        float logexp = log1pf(__expf(-absL));
                        float ce_neg = logexp + maxL;
                        float ce_pos = logexp + maxNegL;
                        float sigma = 1.0f / (1.0f + __expf(-Lij));
                        float p_clamped = fminf(fmaxf(sigma, eps), 1.0f - eps);
                        float entropy = -(p_clamped * logf(p_clamped)
                            + (1.0f - p_clamped) * logf(1.0f - p_clamped));
                        entropy *= inv_log2;
                        float weight = use_uncertainty ? powf(entropy, uncertainty_gamma) : 1.0f;
                        weight = fminf(1.0f, fmaxf(weight, uncertainty_gamma_min));
                        float one_minus = 1.0f - sigma;
                        float mod_neg = (gamma > 0.0f) ? powf(sigma, gamma) : 1.0f;
                        float mod_pos = (gamma > 0.0f) ? powf(one_minus, gamma) : 1.0f;
                        float coeff_neg = alpha_neg * mod_neg * ce_neg;
                        float coeff_pos = alpha_pos * mod_pos * ce_pos;
                        float coeff_delta = weight * (coeff_pos - coeff_neg);
                        thread_gt += coeff_delta * float(n);
                    }
                }
                __syncthreads();
            }
        }

        // reduce and add base_sum
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            thread_gt += __shfl_down_sync(0xffffffff, thread_gt, off);
        if ((tid & 31) == 0) s_warp[tid >> 5] = thread_gt;
        __syncthreads();
        for (int s = NUM_WARPS>>1; s > 0; s >>= 1) {
            if (tid < s) s_warp[tid] += s_warp[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            float v = (base_sum + s_warp[0]) / weight_total;
            out[((l*B + b)*C + ci)*GT_out + out_gt_idx] = v * scale;
        }
    }
}

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,   // (L,B,C,H,W), float
    const torch::Tensor& targets,  // (B,H_t,W_t), int64
    int64_t background_index = -1,
    const float scale = 1.0f,
    const float gamma = 0.0f,
    const float alpha = -1.0f,
    const float uncertainty_gamma = 1.0f,
    const float uncertainty_gamma_min = 0.05f,
    const bool normalize_uncertainty = true
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
    auto counts = torch::zeros({B, GT_total, H, W}, logits.options().dtype(torch::kUInt8));

    // Launch count kernel (counts for ALL GT labels)
    {
        dim3 block(16,16);
        dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y, B);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value, decltype(W_val)::value,
                decltype(H_t_val)::value, decltype(W_t_val)::value>
                <<<grid, block>>>(
                    targets.data_ptr<int64_t>(),
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
    auto total_counts = counts.sum({2, 3}).to(torch::kInt32).contiguous(); // shape (B, GT_total)
    
    // Determine whether the output matrix should have one less column (skip background).
    const bool skip_class = (background_index >= 0 && background_index < GT_total);
    if (background_index < 0) {
        // If negative, set to an invalid index to simplify device code
        background_index = GT_total; 
    }
    const int GT_out = GT_total - (skip_class ? 1 : 0);

    // If no classes left to evaluate (edge case), return an empty tensor of shape (B,C,0)
    if (GT_out == 0) {
        return torch::zeros({L, B, C, 0}, logits.options().dtype(torch::kFloat32));
    }

    // Launch reduction kernel only for the compacted GT_out entries
    auto out = torch::zeros({L, B, C, GT_out}, logits.options().dtype(torch::kFloat32));
    {
        dim3 grid(L, B, C);

        auto static_launcher = [&](auto C_val, auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            reduce_pairwise_sigmoid_kernel<
                decltype(C_val)::value, decltype(H_val)::value,
                decltype(W_val)::value, decltype(H_t_val)::value,
                decltype(W_t_val)::value>
                <<<grid, REDUCTION_THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(),
                    counts.data_ptr<uint8_t>(),
                    out.data_ptr<float>(),
                    total_counts.data_ptr<int32_t>(),
                    static_cast<int32_t>(background_index),
                    GT_total,
                    GT_out,
                    B, L, scale,
                    gamma,
                    alpha,
                    uncertainty_gamma,
                    uncertainty_gamma_min,
                    normalize_uncertainty
                );
        };

        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 128>{}),  // C
            std::make_tuple(std::integral_constant<int, 256>{}),  // H
            std::make_tuple(std::integral_constant<int, 256>{}),  // W
            std::make_tuple(std::integral_constant<int, 1024>{}), // H_t
            std::make_tuple(std::integral_constant<int, 1024>{})  // W_t
        );
        // W_t is not needed by reduce_loss_kernel, so we only pass needed dims
        const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
    }

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after reduce kernel: ", cudaGetErrorString(err));

    return out;
}