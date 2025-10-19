#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>

#include "utils.h"
#include "utils.cuh"

namespace {

// CUDA kernels implementing the backward pass for the mask-matching loss.
// The kernels reuse the forward Hungarian assignments and accumulate
// sigmoid + dice gradients directly on device buffers.

// Threads used for computing gradients.
constexpr int GRAD_THREADS = 256;

__global__ void mask_matching_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ matches,
    const uint8_t* __restrict__ counts,
    const float* __restrict__ grad_mask_mean,
    const float* __restrict__ grad_dice_mean,
    float* __restrict__ grad_logits,
    const int64_t L,
    const int64_t B,
    const int64_t C,
    const int64_t H,
    const int64_t W,
    const int64_t GT_out,
    const int64_t GT_total,
    const int64_t background_index,
    const float smooth,
    const float sigmoid_factor,
    const float dice_scale,
    const float area_scale,
    const float inv_denom
) {
    const int64_t g = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t l = blockIdx.z;

    if (l >= L || b >= B || g >= GT_out) {
        return;
    }

    const int64_t match_index = ((l * B) + b) * GT_out + g;
    const int64_t pred = matches[match_index];
    if (pred < 0 || pred >= C) {
        return;
    }

    int64_t actual_gt = g;
    if (background_index >= 0 && background_index < GT_total && g >= background_index) {
        actual_gt += 1;
    }
    if (actual_gt < 0 || actual_gt >= GT_total) {
        return;
    }

    // Each layer shares the same coefficients across ground-truth indices.
    const float mask_coeff = grad_mask_mean[l] * inv_denom;
    const float dice_coeff = grad_dice_mean[l] * inv_denom;
    if (mask_coeff == 0.0f && dice_coeff == 0.0f) {
        return;
    }

    const int64_t HW = H * W;
    const int64_t logits_base = (((l * B) + b) * C + pred) * H * W;
    const int64_t counts_base = (((b * GT_total) + actual_gt) * H) * W;

    __shared__ float sh_mask_sum;
    __shared__ float sh_target_sum;
    __shared__ float sh_inter_sum;
    if (threadIdx.x == 0) {
        sh_mask_sum = 0.0f;
        sh_target_sum = 0.0f;
        sh_inter_sum = 0.0f;
    }
    __syncthreads();

    float local_mask_sum = 0.0f;
    float local_target_sum = 0.0f;
    float local_inter_sum = 0.0f;

    for (int64_t idx = threadIdx.x; idx < HW; idx += blockDim.x) {
        const int64_t h = idx / W;
        const int64_t w = idx % W;

        const int64_t logits_offset = logits_base + h * W + w;
        const float logit = logits[logits_offset];
        const float prob = 1.0f / (1.0f + expf(-logit));
        const float prob_scaled = prob * area_scale;

        const int64_t counts_offset = counts_base + h * W + w;
        const float target = static_cast<float>(counts[counts_offset]);

        local_mask_sum += prob_scaled;
        local_target_sum += target;
        local_inter_sum += prob * target;
    }

    // Aggregate the per-thread partial sums for the dice statistics.
    atomicAdd(&sh_mask_sum, local_mask_sum);
    atomicAdd(&sh_target_sum, local_target_sum);
    atomicAdd(&sh_inter_sum, local_inter_sum);
    __syncthreads();

    const float mask_sum = sh_mask_sum;
    const float target_sum = sh_target_sum;
    const float inter_sum = sh_inter_sum;

    const float denom = mask_sum + target_sum + smooth;
    const float numerator = inter_sum * 2.0f + smooth;
    float denom_sq = denom * denom;
    if (denom_sq < 1e-20f) {
        denom_sq = 1e-20f;
    }
    const float inv_denom_sq = 1.0f / denom_sq;
    const float two_denom = denom * 2.0f;

    for (int64_t idx = threadIdx.x; idx < HW; idx += blockDim.x) {
        const int64_t h = idx / W;
        const int64_t w = idx % W;

        const int64_t logits_offset = logits_base + h * W + w;
        const float logit = logits[logits_offset];
        const float prob = 1.0f / (1.0f + expf(-logit));
        const float prob_scaled = prob * area_scale;
        const float prob_prime = prob * (1.0f - prob);

        const int64_t counts_offset = counts_base + h * W + w;
        const float target = static_cast<float>(counts[counts_offset]);

        // Sigmoid CE gradient and dice gradient for a single pixel.
        const float grad_sigmoid = (prob_scaled - target) * sigmoid_factor;
        const float d_inter = prob_prime * target;
        const float d_denom = prob_prime * area_scale;
        const float grad_dice = (numerator * d_denom - two_denom * d_inter) * inv_denom_sq * dice_scale;

        const float grad = mask_coeff * grad_sigmoid + dice_coeff * grad_dice;
        grad_logits[logits_offset] = grad;
    }
}

} // namespace

torch::Tensor mask_matching_backward(
    const torch::Tensor& grad_layer_mask_mean,
    const torch::Tensor& grad_layer_dice_mean,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& matches,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const int64_t background_index,
    const int64_t num_masks,
    const int64_t matched_count
) {
    // Backward pipeline:
    //   1. Materialize downsampled ground-truth masks once for reuse.
    //   2. Launch one CUDA block per (layer, batch, gt) assignment to compute
    //      sigmoid and dice gradients directly into the logits tensor.
    CHECK_INPUT(grad_layer_mask_mean);
    CHECK_INPUT(grad_layer_dice_mean);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(matches);

    const auto device = logits.device();
    TORCH_CHECK(targets.device() == device, "targets must be on the same device as logits");
    TORCH_CHECK(matches.device() == device, "matches must be on the same device as logits");

    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(grad_layer_mask_mean.scalar_type() == torch::kFloat32, "grad_layer_mask_mean must be float32");
    TORCH_CHECK(grad_layer_dice_mean.scalar_type() == torch::kFloat32, "grad_layer_dice_mean must be float32");

    TORCH_CHECK(targets.scalar_type() == torch::kLong, "logits must be long");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    TORCH_CHECK(matches.is_contiguous(), "matches must be contiguous");
    TORCH_CHECK(grad_layer_mask_mean.is_contiguous(), "grad_layer_mask_mean must be contiguous");
    TORCH_CHECK(grad_layer_dice_mean.is_contiguous(), "grad_layer_dice_mean must be contiguous");

    const int64_t L = logits.size(0);
    const int64_t B = logits.size(1);
    const int64_t C = logits.size(2);
    const int64_t H = logits.size(3);
    const int64_t W = logits.size(4);

    const int64_t H_t = targets.size(1);
    const int64_t W_t = targets.size(2);

    TORCH_CHECK(H_t % H == 0 && W_t % W == 0, "Target resolution must be an integer multiple of logits resolution.");
    const int64_t scale = H_t / H;
    TORCH_CHECK(scale > 0, "Invalid spatial scale");
    const float area_scale = static_cast<float>(scale * scale);

    const int64_t GT_out = matches.size(2);
    TORCH_CHECK(GT_out > 0 || targets.numel() == 0, "matches must have a non-zero last dimension");

    auto grad_logits = torch::zeros_like(logits);

    // The forward pass counts valid assignments while aggregating losses, so we
    // can reuse the same value here without scanning the matches tensor again.
    if (matched_count <= 0) {
        return grad_logits;
    }

    // Derive the normalization factor used for both dice and sigmoid terms.
    int64_t denom_masks = num_masks > 0 ? num_masks : matched_count;
    if (denom_masks <= 0) {
        denom_masks = 1;
    }
    const float inv_denom = 1.0f / static_cast<float>(denom_masks);

    int64_t GT_total = 0;
    if (targets.numel() > 0) {
        GT_total = targets.max().item<int64_t>() + 1;
    }
    if (GT_total == 0) {
        return grad_logits;
    }

    // Pre-compute the per-(batch,gt) downsampled label counts so each CUDA block
    // can reuse them without touching the targets tensor again.
    auto counts = torch::zeros({B, GT_total, H, W}, logits.options().dtype(torch::kUInt8));

    {
        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, B);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value,
                decltype(W_val)::value,
                decltype(H_t_val)::value,
                decltype(W_t_val)::value><<<grid, block>>>(
                    targets.data_ptr<int64_t>(),
                    counts.data_ptr<uint8_t>(),
                    static_cast<int>(B),
                    static_cast<int>(GT_total)
                );
        };

        const auto runtime_dims = std::make_tuple(static_cast<int>(H), static_cast<int>(W), static_cast<int>(H_t), static_cast<int>(W_t));
        const auto supported_dims = std::make_tuple(
            std::make_tuple(std::integral_constant<int, 256>{}),
            std::make_tuple(std::integral_constant<int, 256>{}),
            std::make_tuple(std::integral_constant<int, 1024>{}),
            std::make_tuple(std::integral_constant<int, 1024>{})
        );
        dispatch_kernel(static_launcher, runtime_dims, supported_dims);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    const float norm = 1.0f / static_cast<float>(H_t * W_t);
    const float sigmoid_factor = sigmoid_scale * norm;

    if (L == 0 || B == 0 || GT_out == 0) {
        return grad_logits;
    }

    dim3 grad_grid(static_cast<unsigned int>(GT_out), static_cast<unsigned int>(B), static_cast<unsigned int>(L));
    mask_matching_backward_kernel<<<grad_grid, GRAD_THREADS>>>(
        logits.data_ptr<float>(),
        matches.data_ptr<int64_t>(),
        counts.data_ptr<uint8_t>(),
        grad_layer_mask_mean.data_ptr<float>(),
        grad_layer_dice_mean.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        L,
        B,
        C,
        H,
        W,
        GT_out,
        GT_total,
        background_index,
        smooth,
        sigmoid_factor,
        dice_scale,
        area_scale,
        inv_denom
    );
    CHECK_CUDA_ERROR(cudaGetLastError());

    return grad_logits;
}

