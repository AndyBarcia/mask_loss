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

// CUDA kernels implementing the backward pass for the mask-matching loss.
// The kernels reuse the forward Hungarian assignments and accumulate
// sigmoid + dice gradients directly on device buffers.

// Threads used for computing gradients.
constexpr int GRAD_THREADS = 256;

template <bool ForceUnmatchedMasks>
__global__ void mask_matching_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ pred_to_gt,
    const uint8_t* __restrict__ counts,
    const float* __restrict__ grad_mask_mean,
    const float* __restrict__ grad_dice_mean,
    float* __restrict__ grad_logits,
    const int64_t L,
    const int64_t B,
    const int64_t Q,
    const int64_t H,
    const int64_t W,
    const int64_t GT_total,
    const int64_t GT_out,
    const float smooth,
    const float sigmoid_factor,
    const float dice_scale,
    const float area_scale,
    const float inv_denom,
    const int64_t background_index
) {
    const int64_t q = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t l = blockIdx.z;

    if (l >= L || b >= B || q >= Q) {
        return;
    }

    const int64_t pred_index = ((l * B) + b) * Q + q;
    const int64_t out_gt = pred_to_gt[pred_index];
    const bool is_matched = out_gt >= 0 && out_gt < GT_out;

    if (!is_matched && !ForceUnmatchedMasks) {
        return;
    }

    if (is_matched && counts == nullptr) {
        return;
    }

    const int64_t actual_gt = MAP_OUT_TO_ACTUAL(out_gt, background_index);

    // Each layer shares the same coefficients across ground-truth indices.
    const float mask_coeff = grad_mask_mean[l] * inv_denom;
    const float dice_coeff = grad_dice_mean[l] * inv_denom;
    if (mask_coeff == 0.0f && dice_coeff == 0.0f) {
        return;
    }

    const int64_t HW = H * W;
    const int64_t logits_base = pred_index * H * W;
    const uint8_t* counts_ptr = nullptr;
    if (is_matched) {
        const int64_t counts_base = (((b * GT_total) + actual_gt) * H) * W;
        counts_ptr = counts + counts_base;
    }

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

        float target = 0.0f;
        if (is_matched) {
            target = static_cast<float>(counts_ptr[h * W + w]);
        }

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

        float target = 0.0f;
        if (is_matched) {
            target = static_cast<float>(counts_ptr[h * W + w]);
        }

        // Sigmoid CE gradient and dice gradient for a single pixel.
        const float grad_sigmoid = (prob_scaled - target) * sigmoid_factor;
        const float d_inter = prob_prime * target;
        const float d_denom = prob_prime * area_scale;
        const float grad_dice = (numerator * d_denom - two_denom * d_inter) * inv_denom_sq * dice_scale;

        const float grad = mask_coeff * grad_sigmoid + dice_coeff * grad_dice;
        grad_logits[logits_offset] = grad;
    }
}

template <bool ForceUnmatchedClass>
__global__ void cls_matching_backward_kernel(
    const float* __restrict__ cls_logits,       // (L,B,Q,C)
    const int64_t* __restrict__ cls_targets,    // (B,GT_total), -1 padded
    const int64_t* __restrict__ pred_to_gt,     // (L,B,Q)
    const float* __restrict__ grad_layer_cls,   // (L,), upstream grad of layer-wise mean
    float* __restrict__ grad_cls_logits,        // (L,B,Q,C) (output)
    const int64_t L,
    const int64_t B,
    const int64_t Q,
    const int64_t C,
    const int64_t GT_total,
    const int64_t GT_out,
    const float coeff_base,
    const int64_t background_index
) {
    const int64_t q = blockIdx.x;  // prediction index
    const int64_t b = blockIdx.y;
    const int64_t l = blockIdx.z;

    if (l >= L || b >= B || q >= Q) return;

    const int64_t pred_index = ((l * B) + b) * Q + q;
    const int64_t out_gt = pred_to_gt[pred_index];
    const bool is_matched = out_gt >= 0 && out_gt < GT_out;
    if (!is_matched && !ForceUnmatchedClass) return;

    const int64_t actual_gt = MAP_OUT_TO_ACTUAL(out_gt, background_index);

    // Per-layer coefficient: upstream grad * normalization * cls_scale * (1/C)
    const float coeff = grad_layer_cls[l] * coeff_base / static_cast<float>(C);
    if (coeff == 0.0f) return;

    // Write grads for the entire class vector of the matched/unmatched query
    const int64_t base = pred_index * C;

    int y = -1;
    if (is_matched) {
        const int64_t y64 = cls_targets[b * GT_total + actual_gt];
        if (y64 < 0 || y64 >= C) return; // invalid label (padding or OOR)
        y = static_cast<int>(y64);
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float z = cls_logits[base + c];
        const float p = 1.0f / (1.0f + __expf(-z));   // sigmoid
        const float t = (is_matched && c == y) ? 1.0f : 0.0f;
        // d/dz BCE(one-hot) averaged over C: (p - t) / C
        grad_cls_logits[base + c] = coeff * (p - t);
    }
}

std::vector<torch::Tensor> mask_matching_backward(
    const torch::Tensor& grad_layer_mask_mean, // (L,)
    const torch::Tensor& grad_layer_dice_mean, // (L,)
    const torch::Tensor& grad_layer_cls_mean,  // (L,)
    const torch::Tensor& mask_logits,          // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,         // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,           // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,          // (B,GT_total),int64,
    const torch::Tensor& gt_to_pred,           // (L,B,GT_out),int64
    const torch::Tensor& pred_to_gt,           // (L,B,Q),int64
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    int64_t background_index,
    const int64_t num_masks,
    const int64_t matched_count,
    const bool force_unmatched_class_to_background,
    const bool force_unmatched_masks_to_empty
) {
    // Backward pipeline:
    //   1. Materialize downsampled ground-truth masks once for reuse.
    //   2. Launch one CUDA block per (layer, batch, gt) assignment to compute
    //      sigmoid and dice gradients directly into the mask_logits tensor.
    CHECK_INPUT(grad_layer_mask_mean);
    CHECK_INPUT(grad_layer_dice_mean);
    CHECK_INPUT(grad_layer_cls_mean);
    CHECK_INPUT(mask_logits);
    CHECK_INPUT(mask_targets);
    CHECK_INPUT(cls_logits);
    CHECK_INPUT(cls_targets);
    CHECK_INPUT(gt_to_pred);
    CHECK_INPUT(pred_to_gt);

    const auto device = mask_logits.device();
    TORCH_CHECK(mask_targets.device() == device, "mask_targets must be on the same device as mask_logits");
    TORCH_CHECK(gt_to_pred.device() == device, "gt_to_pred must be on the same device as mask_logits");
    TORCH_CHECK(pred_to_gt.device() == device, "pred_to_gt must be on the same device as mask_logits");

    TORCH_CHECK(mask_logits.scalar_type() == torch::kFloat32, "mask_logits must be float32");
    TORCH_CHECK(grad_layer_mask_mean.scalar_type() == torch::kFloat32, "grad_layer_mask_mean must be float32");
    TORCH_CHECK(grad_layer_dice_mean.scalar_type() == torch::kFloat32, "grad_layer_dice_mean must be float32");

    TORCH_CHECK(mask_targets.scalar_type() == torch::kLong, "mask_logits must be long");
    TORCH_CHECK(mask_logits.is_contiguous(), "mask_logits must be contiguous");
    TORCH_CHECK(gt_to_pred.is_contiguous(), "gt_to_pred must be contiguous");
    TORCH_CHECK(pred_to_gt.is_contiguous(), "pred_to_gt must be contiguous");
    TORCH_CHECK(grad_layer_mask_mean.is_contiguous(), "grad_layer_mask_mean must be contiguous");
    TORCH_CHECK(grad_layer_dice_mean.is_contiguous(), "grad_layer_dice_mean must be contiguous");

    const int64_t L = mask_logits.size(0);
    const int64_t B = mask_logits.size(1);
    const int64_t Q = mask_logits.size(2);
    const int64_t H = mask_logits.size(3);
    const int64_t W = mask_logits.size(4);
    const int64_t C = cls_logits.size(3);
    const int64_t GT_total = cls_targets.size(1);
    const int64_t GT_out = gt_to_pred.size(2);

    TORCH_CHECK(H > 0 && W > 0, "mask_logits must have positive spatial size");
    TORCH_CHECK(GT_out > 0 || mask_targets.numel() == 0, "gt_to_pred must have a non-zero last dimension");

    const int64_t H_t = mask_targets.size(1);
    const int64_t W_t = mask_targets.size(2);

    TORCH_CHECK(H_t % H == 0 && W_t % W == 0, "Target resolution must be an integer multiple of mask_logits resolution.");
    const int64_t scale = H_t / H;
    TORCH_CHECK(scale > 0, "Invalid spatial scale");
    const float area_scale = static_cast<float>(scale * scale);

    if (background_index < 0) {
        // If negative, set to an invalid index to simplify device code
        background_index = GT_total; 
    }

    auto grad_mask_logits = torch::zeros_like(mask_logits);
    auto grad_cls_logits  = torch::zeros_like(cls_logits);

    const int64_t total_queries = L * B * Q;
    const int64_t unmatched_queries = std::max<int64_t>(int64_t{0}, total_queries - matched_count);

    int64_t mask_norm = (num_masks > 0)
        ? num_masks
        : matched_count + (force_unmatched_masks_to_empty ? unmatched_queries : 0);
    if (mask_norm <= 0) mask_norm = 1;
    const float inv_mask_denom = 1.0f / static_cast<float>(mask_norm);

    int64_t cls_norm = (num_masks > 0)
        ? num_masks
        : matched_count + (force_unmatched_class_to_background ? unmatched_queries : 0);
    if (cls_norm <= 0) cls_norm = 1;
    const float inv_cls_denom = 1.0f / static_cast<float>(cls_norm);

    torch::Tensor counts;
    if (matched_count > 0 && GT_total > 0 && mask_targets.numel() > 0 && L > 0 && B > 0 && H > 0 && W > 0) {
        counts = torch::zeros({B, GT_total, H, W}, mask_logits.options().dtype(torch::kUInt8));

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, B);

        auto static_launcher = [&](auto H_val, auto W_val, auto H_t_val, auto W_t_val) {
            count_labels_per_block_kernel<
                decltype(H_val)::value,
                decltype(W_val)::value,
                decltype(H_t_val)::value,
                decltype(W_t_val)::value><<<grid, block>>>(
                    mask_targets.data_ptr<int64_t>(),
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

    const float norm = (H_t > 0 && W_t > 0)
        ? 1.0f / static_cast<float>(H_t * W_t)
        : 0.0f;
    const float sigmoid_factor = sigmoid_scale * norm;
    const float coeff_base = cls_scale * inv_cls_denom;

    const uint8_t* counts_ptr = counts.defined() ? counts.data_ptr<uint8_t>() : nullptr;

    const auto runtime_flags = std::make_tuple(
        static_cast<int>(force_unmatched_masks_to_empty),
        static_cast<int>(force_unmatched_class_to_background)
    );
    const auto supported_flags = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{}),
        std::make_tuple(std::integral_constant<int, 0>{}, std::integral_constant<int, 1>{})
    );

    auto launch_kernels = [&](auto ForceMaskFlag, auto ForceClsFlag) {
        constexpr bool ForceMask = (decltype(ForceMaskFlag)::value != 0);
        constexpr bool ForceCls = (decltype(ForceClsFlag)::value != 0);

        if (L > 0 && B > 0 && Q > 0 && H > 0 && W > 0 && (matched_count > 0 || ForceMask)) {
            dim3 grad_grid(static_cast<unsigned int>(Q), static_cast<unsigned int>(B), static_cast<unsigned int>(L));
            mask_matching_backward_kernel<ForceMask><<<grad_grid, GRAD_THREADS>>>(
                mask_logits.data_ptr<float>(),
                pred_to_gt.data_ptr<int64_t>(),
                counts_ptr,
                grad_layer_mask_mean.data_ptr<float>(),
                grad_layer_dice_mean.data_ptr<float>(),
                grad_mask_logits.data_ptr<float>(),
                L,
                B,
                Q,
                H,
                W,
                GT_total,
                GT_out,
                smooth,
                sigmoid_factor,
                dice_scale,
                area_scale,
                inv_mask_denom,
                background_index
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        if (L > 0 && B > 0 && Q > 0 && C > 0 && (matched_count > 0 || ForceCls)) {
            dim3 cls_grid(static_cast<unsigned int>(Q), static_cast<unsigned int>(B), static_cast<unsigned int>(L));
            cls_matching_backward_kernel<ForceCls><<<cls_grid, GRAD_THREADS>>>(
                cls_logits.data_ptr<float>(),
                cls_targets.data_ptr<int64_t>(),
                pred_to_gt.data_ptr<int64_t>(),
                grad_layer_cls_mean.data_ptr<float>(),
                grad_cls_logits.data_ptr<float>(),
                L,
                B,
                Q,
                C,
                GT_total,
                GT_out,
                coeff_base,
                background_index
            );
            CHECK_CUDA_ERROR(cudaGetLastError());
        }
    };

    dispatch_kernel(launch_kernels, runtime_flags, supported_flags);

    return {grad_mask_logits, grad_cls_logits};
}

