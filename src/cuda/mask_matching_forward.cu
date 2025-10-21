#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <tuple>
#include <cstdint>

namespace {

inline __device__ float softplusf(float x) {
    const float abs_x = fabsf(x);
    return log1pf(expf(-abs_x)) + fmaxf(x, 0.0f);
}

template <bool ForceMasks, bool ForceClass>
__global__ void unmatched_forward_kernel(
    const float* __restrict__ mask_logits,
    const float* __restrict__ cls_logits,
    const uint8_t* __restrict__ unmatched_flags,
    float* __restrict__ layer_mask_sum,
    float* __restrict__ layer_dice_sum,
    float* __restrict__ layer_cls_sum,
    const int64_t L,
    const int64_t B,
    const int64_t Q,
    const int64_t H,
    const int64_t W,
    const int64_t C,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const float area_scale
) {
    const int64_t q = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t l = blockIdx.z;

    if (l >= L || b >= B || q >= Q) {
        return;
    }

    const int64_t pred_index = ((l * B) + b) * Q + q;
    if (!unmatched_flags[pred_index]) {
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

    if (ForceClass) {
        const int64_t base_cls = pred_index * C;
        float sum_softplus = 0.0f;
        for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
            const float logit = cls_logits[base_cls + c];
            sum_softplus += softplusf(logit);
        }
        sh_softplus_cls[threadIdx.x] = sum_softplus;
    }

    __syncthreads();

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
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh_softplus_cls[threadIdx.x] += sh_softplus_cls[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float inv_C = 1.0f / static_cast<float>(C);
            const float cls_loss = cls_scale * (sh_softplus_cls[0] * inv_C);
            atomicAdd(layer_cls_sum + l, cls_loss);
        }
    }
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mask_matching_unmatched_forward(
    const torch::Tensor& mask_logits,
    const torch::Tensor& cls_logits,
    const torch::Tensor& unmatched_mask,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const float area_scale,
    const bool force_unmatched_masks,
    const bool force_unmatched_class
) {
    TORCH_CHECK(mask_logits.is_cuda(), "mask_logits must be CUDA");
    TORCH_CHECK(unmatched_mask.is_cuda(), "unmatched_mask must be CUDA");
    TORCH_CHECK(mask_logits.device() == unmatched_mask.device(), "mask_logits and unmatched_mask must be on the same device");
    if (force_unmatched_class) {
        TORCH_CHECK(cls_logits.is_cuda(), "cls_logits must be CUDA when forcing unmatched class loss");
        TORCH_CHECK(cls_logits.device() == mask_logits.device(), "cls_logits and mask_logits must be on the same device");
    }

    const int64_t L = mask_logits.size(0);
    const auto options = mask_logits.options().dtype(torch::kFloat);
    torch::Tensor layer_mask_sum = torch::zeros({L}, options);
    torch::Tensor layer_dice_sum = torch::zeros({L}, options);
    torch::Tensor layer_cls_sum = torch::zeros({L}, options);

    if (!force_unmatched_masks && !force_unmatched_class) {
        return {layer_mask_sum, layer_dice_sum, layer_cls_sum};
    }

    TORCH_CHECK(mask_logits.dim() == 5, "mask_logits must have shape (L,B,Q,H,W)");
    TORCH_CHECK(unmatched_mask.sizes().equals({mask_logits.size(0), mask_logits.size(1), mask_logits.size(2)}),
        "unmatched_mask must have shape (L,B,Q)");

    auto mask_logits_contig = mask_logits.contiguous();
    auto unmatched_contig = unmatched_mask.to(torch::kUInt8).contiguous();
    torch::Tensor cls_contig;

    const int64_t B = mask_logits.size(1);
    const int64_t Q = mask_logits.size(2);
    const int64_t H = mask_logits.size(3);
    const int64_t W = mask_logits.size(4);
    const int threads = 256;
    int shared_elems = 0;

    const float smooth_val = smooth;
    const float sigmoid_scale_val = sigmoid_scale;
    const float dice_scale_val = dice_scale;
    const float cls_scale_val = cls_scale;
    const float area_scale_val = area_scale;

    const int device_index = mask_logits.device().index();
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    const float* mask_ptr = mask_logits_contig.data_ptr<float>();
    float* mask_sum_ptr = layer_mask_sum.data_ptr<float>();
    float* dice_sum_ptr = layer_dice_sum.data_ptr<float>();

    const uint8_t* unmatched_ptr = unmatched_contig.data_ptr<uint8_t>();

    dim3 grid(Q, B, L);

    const float* cls_ptr = nullptr;
    float* cls_sum_ptr = layer_cls_sum.data_ptr<float>();
    int64_t C = 0;

    if (force_unmatched_class) {
        TORCH_CHECK(cls_logits.dim() == 4, "cls_logits must have shape (L,B,Q,C)");
        TORCH_CHECK(cls_logits.size(0) == L && cls_logits.size(1) == B && cls_logits.size(2) == Q,
            "cls_logits must match mask logits dimensions");
        cls_contig = cls_logits.contiguous();
        C = cls_contig.size(3);
        cls_ptr = cls_contig.data_ptr<float>();
    }

    if (force_unmatched_masks && force_unmatched_class) {
        shared_elems = 3;
        unmatched_forward_kernel<true, true><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            mask_ptr,
            cls_ptr,
            unmatched_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val
        );
    } else if (force_unmatched_masks) {
        shared_elems = 2;
        unmatched_forward_kernel<true, false><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            mask_ptr,
            nullptr,
            unmatched_ptr,
            mask_sum_ptr,
            dice_sum_ptr,
            nullptr,
            L,
            B,
            Q,
            H,
            W,
            0,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val
        );
    } else if (force_unmatched_class) {
        shared_elems = 1;
        unmatched_forward_kernel<false, true><<<grid, threads, shared_elems * threads * sizeof(float), stream.stream()>>>(
            nullptr,
            cls_ptr,
            unmatched_ptr,
            nullptr,
            nullptr,
            cls_sum_ptr,
            L,
            B,
            Q,
            H,
            W,
            C,
            smooth_val,
            sigmoid_scale_val,
            dice_scale_val,
            cls_scale_val,
            area_scale_val
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {layer_mask_sum, layer_dice_sum, layer_cls_sum};
}

