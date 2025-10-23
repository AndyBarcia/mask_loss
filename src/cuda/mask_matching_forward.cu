#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <tuple>
#include <cstdint>

inline __device__ float softplusf(float x) {
    const float abs_x = fabsf(x);
    return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

// Each block processes a single (layer, batch, query) tuple. Matched queries
// directly accumulate their pre-computed losses into the per-layer sums while
// unmatched queries, when requested, evaluate their penalties on the fly. By
// templating on the two enforcement flags we allow the compiler to drop any
// unused code paths and shared-memory allocations for unmatched processing.
template <bool ForceMasks, bool ForceClass>
__global__ void mask_matching_forward_kernel(
    const float* __restrict__ mask_logits,
    const float* __restrict__ cls_logits,
    const float* __restrict__ separate_costs,
    const int64_t* __restrict__ pred_to_gt,
    float* __restrict__ layer_mask_sum,
    float* __restrict__ layer_dice_sum,
    float* __restrict__ layer_cls_sum,
    const int64_t L,
    const int64_t B,
    const int64_t Q,
    const int64_t H,
    const int64_t W,
    const int64_t C,
    const int64_t GT_out,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const float area_scale,
    const int64_t term_stride,
    const int64_t background_index,
    const int32_t label_loss_type,
    const float label_focal_alpha,
    const float label_focal_gamma
) {
    const int64_t q = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t l = blockIdx.z;

    if (l >= L || b >= B || q >= Q) {
        return;
    }

    const int64_t pred_index = ((l * B) + b) * Q + q;
    const int64_t gt_index = pred_to_gt[pred_index];

    if (gt_index >= 0) {
        if (threadIdx.x == 0) {
            const int64_t base = pred_index * GT_out + gt_index;
            atomicAdd(layer_mask_sum + l, separate_costs[base]);
            atomicAdd(layer_dice_sum + l, separate_costs[base + term_stride]);
            atomicAdd(layer_cls_sum + l, separate_costs[base + (term_stride << 1)]);
        }
        return;
    }

    if (!(ForceMasks || ForceClass)) {
        return;
    }

    extern __shared__ float shared[];
    float* cursor = shared;
    float* sh_softplus_mask = nullptr;
    float* sh_sigmoid = nullptr;
    float* sh_softplus_cls = nullptr;

    if (ForceMasks) {
        sh_softplus_mask = cursor;
        cursor += blockDim.x;
        sh_sigmoid = cursor;
        cursor += blockDim.x;
    }
    if (ForceClass) {
        sh_softplus_cls = cursor;
    }

    if (ForceMasks) {
        const int64_t HW = H * W;
        const int64_t base = pred_index * HW;
        float sum_softplus = 0.0f;
        float sum_sigmoid = 0.0f;
        for (int64_t idx = threadIdx.x; idx < HW; idx += blockDim.x) {
            const float logit = mask_logits[base + idx];
            sum_softplus += softplusf(logit);
            sum_sigmoid += 1.0f / (1.0f + expf(-logit));
        }
        sh_softplus_mask[threadIdx.x] = sum_softplus;
        sh_sigmoid[threadIdx.x] = sum_sigmoid;
    }

    if (ForceMasks) {
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh_softplus_mask[threadIdx.x] += sh_softplus_mask[threadIdx.x + stride];
                sh_sigmoid[threadIdx.x] += sh_sigmoid[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float HW = static_cast<float>(H * W);
            const float bce_mean = sh_softplus_mask[0] / HW;
            const float prob_sum = sh_sigmoid[0] * area_scale;
            const float dice_loss = dice_scale * (prob_sum / (prob_sum + smooth));
            atomicAdd(layer_mask_sum + l, sigmoid_scale * bce_mean);
            atomicAdd(layer_dice_sum + l, dice_loss);
        }
    }

    if (ForceClass) {
        __syncthreads();

        const bool use_bce = (label_loss_type == 0) || (label_loss_type == 1);
        const bool use_bce_focal = (label_loss_type == 1);
        const bool use_ce = (label_loss_type == 2) || (label_loss_type == 3);
        const bool has_alpha = label_focal_alpha >= 0.f;
        const float alpha_clamped = has_alpha ? fminf(fmaxf(label_focal_alpha, 0.f), 1.f) : 1.f;
        const float alpha_neg = has_alpha ? (1.f - alpha_clamped) : 1.f;

        const int64_t base_cls = pred_index * C;
        float sum_softplus = 0.0f;
        float sum_mod_softplus = 0.0f;
        float thread_max = -INFINITY;
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            const float logit = cls_logits[base_cls + c];
            if (use_bce) {
                sum_softplus += softplusf(logit);
            }
            if (use_bce_focal) {
                const float sig = 1.0f / (1.0f + expf(-logit));
                sum_mod_softplus += powf(sig, label_focal_gamma) * softplusf(logit);
            }
            if (use_ce) {
                thread_max = fmaxf(thread_max, logit);
            }
        }

        float block_sum_softplus = 0.0f;
        if (use_bce) {
            sh_softplus_cls[threadIdx.x] = sum_softplus;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sh_softplus_cls[threadIdx.x] += sh_softplus_cls[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                block_sum_softplus = sh_softplus_cls[0];
            }
            __syncthreads();
        }

        float block_sum_mod_softplus = 0.0f;
        if (use_bce_focal) {
            sh_softplus_cls[threadIdx.x] = sum_mod_softplus;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sh_softplus_cls[threadIdx.x] += sh_softplus_cls[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                block_sum_mod_softplus = sh_softplus_cls[0];
            }
            __syncthreads();
        }

        float block_logsumexp = 0.0f;
        float z_bg = 0.0f;
        if (use_ce) {
            sh_softplus_cls[threadIdx.x] = thread_max;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sh_softplus_cls[threadIdx.x] = fmaxf(
                        sh_softplus_cls[threadIdx.x],
                        sh_softplus_cls[threadIdx.x + stride]
                    );
                }
                __syncthreads();
            }
            const float block_max = sh_softplus_cls[0];
            __syncthreads();

            float thread_sum = 0.0f;
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                const float logit = cls_logits[base_cls + c];
                thread_sum += expf(logit - block_max);
            }
            sh_softplus_cls[threadIdx.x] = thread_sum;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    sh_softplus_cls[threadIdx.x] += sh_softplus_cls[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                const float sum_exp = sh_softplus_cls[0];
                block_logsumexp = logf(fmaxf(sum_exp, 1e-20f)) + block_max;
                z_bg = cls_logits[base_cls + background_index];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float cls_loss = 0.0f;
            if (label_loss_type == 0) {
                const float inv_C = (C > 0) ? (1.0f / static_cast<float>(C)) : 0.0f;
                cls_loss = cls_scale * (block_sum_softplus * inv_C);
            } else if (label_loss_type == 1) {
                const float inv_C = (C > 0) ? (1.0f / static_cast<float>(C)) : 0.0f;
                cls_loss = cls_scale * (alpha_neg * block_sum_mod_softplus * inv_C);
            } else if (label_loss_type == 2) {
                cls_loss = cls_scale * (block_logsumexp - z_bg);
            } else if (label_loss_type == 3) {
                const float alpha = (label_focal_alpha >= 0.f) ? label_focal_alpha : 1.0f;
                const float log_p = z_bg - block_logsumexp;
                const float p = expf(log_p);
                const float mod = powf(fmaxf(1.0f - p, 0.0f), label_focal_gamma);
                cls_loss = cls_scale * (-alpha * mod * log_p);
            }
            atomicAdd(layer_cls_sum + l, cls_loss);
        }
    }
}

// Launches the CUDA kernel that integrates both matched and unmatched query
// losses into the per-layer accumulators and applies normalization.
std::vector<torch::Tensor> mask_matching_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& separate_costs, // (3,L,B,C,GT_out)
    const torch::Tensor& pred_to_gt,     // (L,B,Q)
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const int64_t target_H,
    const int64_t target_W,
    const double num_masks,
    const bool force_unmatched_masks,
    const bool force_unmatched_class,
    const int64_t background_index,
    const int64_t label_loss_type,
    const float label_focal_alpha,
    const float label_focal_gamma
) {
    TORCH_CHECK(mask_logits.is_cuda(), "mask_logits must be CUDA");
    TORCH_CHECK(separate_costs.is_cuda(), "separate_costs must be CUDA");
    TORCH_CHECK(pred_to_gt.is_cuda(), "pred_to_gt must be CUDA");
    TORCH_CHECK(mask_logits.device() == separate_costs.device(), "mask_logits and separate_costs must be on the same device");
    TORCH_CHECK(mask_logits.device() == pred_to_gt.device(), "mask_logits and pred_to_gt must be on the same device");
    if (force_unmatched_class) {
        TORCH_CHECK(cls_logits.is_cuda(), "cls_logits must be CUDA when forcing unmatched class loss");
        TORCH_CHECK(cls_logits.device() == mask_logits.device(), "cls_logits and mask_logits must be on the same device");
    }

    TORCH_CHECK(mask_logits.dim() == 5, "mask_logits must have shape (L,B,Q,H,W)");
    TORCH_CHECK(separate_costs.dim() == 5 && separate_costs.size(0) == 3,
        "separate_costs must have shape (3,L,B,Q,GT_out)");
    TORCH_CHECK(pred_to_gt.sizes().equals({mask_logits.size(0), mask_logits.size(1), mask_logits.size(2)}),
        "pred_to_gt must have shape (L,B,Q)");

    TORCH_CHECK(separate_costs.dtype() == torch::kFloat32,
        "separate_costs must be float32");

    auto mask_logits_contig = mask_logits.contiguous();
    auto separate_costs_contig = separate_costs.contiguous();
    auto pred_to_gt_contig = pred_to_gt.to(torch::kLong).contiguous();
    torch::Tensor cls_contig;

    const int64_t L = mask_logits.size(0);
    const int64_t B = mask_logits.size(1);
    const int64_t Q = mask_logits.size(2);
    const int64_t H = mask_logits.size(3);
    const int64_t W = mask_logits.size(4);
    const int64_t GT_out = separate_costs_contig.size(4);
    const int threads = 256;
    int shared_elems = 0;

    const float smooth_val = smooth;
    const float sigmoid_scale_val = sigmoid_scale;
    const float dice_scale_val = dice_scale;
    const float cls_scale_val = cls_scale;
    float area_scale_val = 1.0f;

    const int device_index = mask_logits.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    const float* mask_ptr = mask_logits_contig.data_ptr<float>();
    const float* separate_ptr = separate_costs_contig.data_ptr<float>();
    const int64_t* pred_to_gt_ptr = pred_to_gt_contig.data_ptr<int64_t>();

    auto sums_opts = separate_costs_contig.options();
    torch::Tensor layer_mask_sum = torch::zeros({L}, sums_opts);
    torch::Tensor layer_dice_sum = torch::zeros({L}, sums_opts);
    torch::Tensor layer_cls_sum = torch::zeros({L}, sums_opts);

    float* mask_sum_ptr = layer_mask_sum.data_ptr<float>();
    float* dice_sum_ptr = layer_dice_sum.data_ptr<float>();
    float* cls_sum_ptr = layer_cls_sum.data_ptr<float>();

    dim3 grid(Q, B, L);

    const float* cls_ptr = nullptr;
    int64_t C = 0;

    if (force_unmatched_masks) {
        TORCH_CHECK(target_H > 0 && target_W > 0, "mask targets must have positive spatial size");
        TORCH_CHECK(H > 0 && W > 0, "mask logits must have positive spatial size");
        TORCH_CHECK(target_H % H == 0 && target_W % W == 0,
                    "Target resolution must be integer multiples of mask resolution");
        const int64_t scale_h = target_H / H;
        const int64_t scale_w = target_W / W;
        TORCH_CHECK(scale_h == scale_w,
                    "Target downsample factors must be equal in both dimensions");
        area_scale_val = static_cast<float>(scale_h * scale_w);
    }

    if (force_unmatched_class) {
        TORCH_CHECK(cls_logits.dim() == 4, "cls_logits must have shape (L,B,Q,C)");
        TORCH_CHECK(cls_logits.size(0) == L && cls_logits.size(1) == B && cls_logits.size(2) == Q,
            "cls_logits must match mask logits dimensions");
        cls_contig = cls_logits.contiguous();
        C = cls_contig.size(3);
        cls_ptr = cls_contig.data_ptr<float>();
    }

    const int64_t term_stride = L * B * Q * GT_out;

    if (force_unmatched_masks && force_unmatched_class) {
        shared_elems = 3;
        mask_matching_forward_kernel<true, true><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            mask_ptr,
            cls_ptr,
            separate_ptr,
            pred_to_gt_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            GT_out,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val,
            term_stride,
            background_index,
            static_cast<int32_t>(label_loss_type),
            label_focal_alpha,
            label_focal_gamma
        );
    } else if (force_unmatched_masks) {
        shared_elems = 2;
        mask_matching_forward_kernel<true, false><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            mask_ptr,
            cls_ptr,
            separate_ptr,
            pred_to_gt_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            GT_out,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val,
            term_stride,
            background_index,
            static_cast<int32_t>(label_loss_type),
            label_focal_alpha,
            label_focal_gamma
        );
    } else if (force_unmatched_class) {
        shared_elems = 1;
        mask_matching_forward_kernel<false, true><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            mask_ptr,
            cls_ptr,
            separate_ptr,
            pred_to_gt_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            GT_out,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val,
            term_stride,
            background_index,
            static_cast<int32_t>(label_loss_type),
            label_focal_alpha,
            label_focal_gamma
        );
    } else {
        shared_elems = 0;
        mask_matching_forward_kernel<false, false><<<grid, threads, 0, stream.stream()>>>(
            mask_ptr,
            cls_ptr,
            separate_ptr,
            pred_to_gt_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            GT_out,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val,
            term_stride,
            background_index,
            static_cast<int32_t>(label_loss_type),
            label_focal_alpha,
            label_focal_gamma
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto matches = pred_to_gt_contig.ge(0);
    auto matches_f = matches.to(layer_mask_sum.dtype());
    torch::Tensor per_layer_matched = matches_f.sum({1, 2});
    torch::Tensor per_layer_total = torch::full({L}, static_cast<float>(B * Q), layer_mask_sum.options());

    const bool has_num_masks = num_masks > 0.0;

    torch::Tensor mask_denom;
    if (force_unmatched_masks) {
        mask_denom = per_layer_total.clone();
    } else if (has_num_masks) {
        mask_denom = torch::full({L}, static_cast<float>(num_masks), layer_mask_sum.options());
    } else {
        mask_denom = per_layer_matched.clone();
    }

    torch::Tensor cls_denom;
    if (force_unmatched_class) {
        cls_denom = per_layer_total.clone();
    } else if (has_num_masks) {
        cls_denom = torch::full({L}, static_cast<float>(num_masks), layer_mask_sum.options());
    } else {
        cls_denom = per_layer_matched.clone();
    }

    mask_denom = mask_denom.clamp_min(1.0f);
    cls_denom = cls_denom.clamp_min(1.0f);

    torch::Tensor layer_mask_mean = layer_mask_sum / mask_denom;
    torch::Tensor layer_dice_mean = layer_dice_sum / mask_denom;
    torch::Tensor layer_cls_mean = layer_cls_sum / cls_denom;

    auto out_dtype = mask_logits.dtype();
    layer_mask_mean = layer_mask_mean.to(out_dtype);
    layer_dice_mean = layer_dice_mean.to(out_dtype);
    layer_cls_mean = layer_cls_mean.to(out_dtype);

    return { layer_mask_mean, layer_dice_mean, layer_cls_mean };
}

