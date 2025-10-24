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
    const float gamma,
    const float alpha,
    const bool force_unmatched_class_to_background,
    const bool has_void_class,
    const int64_t void_class_index
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

    const bool process_masks = ForceMasks;
    const bool process_class = ForceClass || has_void_class;

    if (!(process_masks || process_class)) {
        return;
    }

    extern __shared__ float shared[];
    float* cursor = shared;
    float* sh_bce_mask = nullptr;
    float* sh_sigmoid = nullptr;
    float* sh_bce_cls = nullptr;
    float* sh_void_pos = nullptr;
    float* sh_void_neg = nullptr;

    if (process_masks) {
        sh_bce_mask = cursor;
        cursor += blockDim.x;
        sh_sigmoid = cursor;
        cursor += blockDim.x;
    }
    if (process_class) {
        sh_bce_cls = cursor;
        cursor += blockDim.x;
        if (has_void_class) {
            sh_void_pos = cursor;
            cursor += blockDim.x;
            sh_void_neg = cursor;
            cursor += blockDim.x;
        }
    }

    const bool use_gamma = gamma > 0.0f;
    const float alpha_pos = (alpha >= 0.0f) ? alpha : 1.0f;
    const float alpha_neg = (alpha >= 0.0f) ? (1.0f - alpha) : 1.0f;

    if (process_masks) {
        const int64_t HW = H * W;
        const int64_t base = pred_index * HW;
        float sum_bce = 0.0f;
        float sum_sigmoid = 0.0f;
        for (int64_t idx = threadIdx.x; idx < HW; idx += blockDim.x) {
            const float logit = mask_logits[base + idx];
            const float prob = 1.0f / (1.0f + expf(-logit));
            const float abs_logit = fabsf(logit);
            const float max_logit = fmaxf(logit, 0.0f);
            const float logexp = log1pf(expf(-abs_logit));
            const float ce_neg = logexp + max_logit;
            const float mod_neg = use_gamma ? powf(prob, gamma) : 1.0f;
            sum_bce += alpha_neg * mod_neg * ce_neg;
            sum_sigmoid += prob;
        }
        sh_bce_mask[threadIdx.x] = sum_bce;
        sh_sigmoid[threadIdx.x] = sum_sigmoid;
    }

    if (process_class) {
        const int64_t base_cls = pred_index * C;
        float sum_neg = 0.0f;
        float void_pos = 0.0f;
        float void_neg = 0.0f;
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            const float logit = cls_logits[base_cls + c];
            const float prob = 1.0f / (1.0f + expf(-logit));
            const float abs_logit = fabsf(logit);
            const float max_logit = fmaxf(logit, 0.0f);
            const float logexp = log1pf(expf(-abs_logit));
            const float ce_neg = logexp + max_logit;
            const float mod_neg = use_gamma ? powf(prob, gamma) : 1.0f;
            const float neg_term = alpha_neg * mod_neg * ce_neg;
            const bool is_void = has_void_class && (c == void_class_index);
            if (force_unmatched_class_to_background) {
                sum_neg += neg_term;
                if (is_void) {
                    const float one_minus = 1.0f - prob;
                    const float ce_pos = logexp + fmaxf(-logit, 0.0f);
                    const float mod_pos = use_gamma ? powf(one_minus, gamma) : 1.0f;
                    void_pos = alpha_pos * mod_pos * ce_pos;
                    void_neg = neg_term;
                }
            } else if (is_void) {
                const float one_minus = 1.0f - prob;
                const float ce_pos = logexp + fmaxf(-logit, 0.0f);
                const float mod_pos = use_gamma ? powf(one_minus, gamma) : 1.0f;
                void_pos = alpha_pos * mod_pos * ce_pos;
            }
        }
        if (sh_bce_cls) {
            sh_bce_cls[threadIdx.x] = sum_neg;
        }
        if (has_void_class) {
            sh_void_pos[threadIdx.x] = void_pos;
            sh_void_neg[threadIdx.x] = void_neg;
        }
    }

    __syncthreads();

    if (process_masks) {
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh_bce_mask[threadIdx.x] += sh_bce_mask[threadIdx.x + stride];
                sh_sigmoid[threadIdx.x] += sh_sigmoid[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float HW = static_cast<float>(H * W);
            const float bce_mean = sh_bce_mask[0] / HW;
            const float prob_sum = sh_sigmoid[0] * area_scale;
            const float dice_loss = dice_scale * (prob_sum / (prob_sum + smooth));
            atomicAdd(layer_mask_sum + l, sigmoid_scale * bce_mean);
            atomicAdd(layer_dice_sum + l, dice_loss);
        }
    }

    if (process_class) {
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                if (sh_bce_cls) {
                    sh_bce_cls[threadIdx.x] += sh_bce_cls[threadIdx.x + stride];
                }
                if (has_void_class) {
                    sh_void_pos[threadIdx.x] += sh_void_pos[threadIdx.x + stride];
                    sh_void_neg[threadIdx.x] += sh_void_neg[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float cls_loss = 0.0f;
            if (force_unmatched_class_to_background) {
                const float inv_C = (C > 0) ? (1.0f / static_cast<float>(C)) : 0.0f;
                cls_loss = sh_bce_cls ? (sh_bce_cls[0] * inv_C) : 0.0f;
                if (has_void_class && C > 0) {
                    cls_loss += (sh_void_pos[0] - sh_void_neg[0]) * inv_C;
                }
            } else if (has_void_class) {
                cls_loss = sh_void_pos[0];
            }
            cls_loss = cls_loss * cls_scale;
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
    const float gamma,
    const float alpha,
    const int64_t target_H,
    const int64_t target_W,
    const double num_masks,
    const bool force_unmatched_masks,
    const bool force_unmatched_class,
    const int64_t void_class_index
) {
    TORCH_CHECK(mask_logits.is_cuda(), "mask_logits must be CUDA");
    TORCH_CHECK(separate_costs.is_cuda(), "separate_costs must be CUDA");
    TORCH_CHECK(pred_to_gt.is_cuda(), "pred_to_gt must be CUDA");
    TORCH_CHECK(mask_logits.device() == separate_costs.device(), "mask_logits and separate_costs must be on the same device");
    TORCH_CHECK(mask_logits.device() == pred_to_gt.device(), "mask_logits and pred_to_gt must be on the same device");
    TORCH_CHECK(mask_logits.dim() == 5, "mask_logits must have shape (L,B,Q,H,W)");
    TORCH_CHECK(separate_costs.dim() == 5 && separate_costs.size(0) == 3,
        "separate_costs must have shape (3,L,B,Q,GT_out)");

    const int64_t L = mask_logits.size(0);
    const int64_t B = mask_logits.size(1);
    const int64_t Q = mask_logits.size(2);
    const int64_t H = mask_logits.size(3);
    const int64_t W = mask_logits.size(4);

    TORCH_CHECK(pred_to_gt.sizes().equals({L, B, Q}),
        "pred_to_gt must have shape (L,B,Q)");

    TORCH_CHECK(separate_costs.dtype() == torch::kFloat32,
        "separate_costs must be float32");

    TORCH_CHECK(cls_logits.dim() == 4, "cls_logits must have shape (L,B,Q,C)");
    TORCH_CHECK(cls_logits.size(0) == L && cls_logits.size(1) == B && cls_logits.size(2) == Q,
        "cls_logits must match mask logits dimensions");

    auto mask_logits_contig = mask_logits.contiguous();
    auto separate_costs_contig = separate_costs.contiguous();
    auto pred_to_gt_contig = pred_to_gt.to(torch::kLong).contiguous();
    torch::Tensor cls_contig;
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
    int64_t C = cls_logits.size(3);
    int64_t void_index = -1;
    if (void_class_index >= 0 && void_class_index < C) {
        void_index = void_class_index;
    }
    bool has_void_class = (void_index >= 0);
    bool need_unmatched_class = force_unmatched_class || has_void_class;

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

    if (need_unmatched_class) {
        TORCH_CHECK(cls_logits.is_cuda(), "cls_logits must be CUDA when supervising unmatched class logits");
        TORCH_CHECK(cls_logits.device() == mask_logits.device(), "cls_logits and mask_logits must be on the same device");
        cls_contig = cls_logits.contiguous();
        C = cls_contig.size(3);
        cls_ptr = cls_contig.data_ptr<float>();
        if (void_index >= C) {
            void_index = -1;
            has_void_class = false;
            need_unmatched_class = force_unmatched_class;
        }
    }

    const int mask_shared = force_unmatched_masks ? 2 : 0;
    const int class_shared = need_unmatched_class ? (1 + (has_void_class ? 2 : 0)) : 0;
    shared_elems = mask_shared + class_shared;

    const int64_t term_stride = L * B * Q * GT_out;

    TORCH_CHECK(gamma >= 0.0f, "focal_gamma must be non-negative");
    TORCH_CHECK(alpha < 0.0f || (alpha >= 0.0f && alpha <= 1.0f),
        "focal_alpha must be in [0, 1] or negative to disable");

    const size_t mask_shared_bytes = static_cast<size_t>(mask_shared) * threads * sizeof(float);
    const size_t class_shared_bytes = static_cast<size_t>(class_shared) * threads * sizeof(float);
    const size_t combined_shared_bytes = static_cast<size_t>(shared_elems) * threads * sizeof(float);

    if (force_unmatched_masks && need_unmatched_class) {
        mask_matching_forward_kernel<true, true><<<grid, threads, combined_shared_bytes, stream.stream()>>>(
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
            gamma,
            alpha,
            force_unmatched_class,
            has_void_class,
            void_index
        );
    } else if (force_unmatched_masks) {
        mask_matching_forward_kernel<true, false><<<grid, threads, mask_shared_bytes, stream.stream()>>>(
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
            gamma,
            alpha,
            false,
            false,
            -1
        );
    } else if (need_unmatched_class) {
        mask_matching_forward_kernel<false, true><<<grid, threads, class_shared_bytes, stream.stream()>>>(
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
            gamma,
            alpha,
            force_unmatched_class,
            has_void_class,
            void_index
        );
    } else {
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
            gamma,
            alpha,
            false,
            false,
            -1
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
    if (force_unmatched_class || has_void_class) {
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

