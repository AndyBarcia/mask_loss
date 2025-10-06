#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK = 256;

template <typename T, int N_POSITIVES, int C, int H, int W>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) multiclass_sigmoid_cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const T* __restrict__ targets,
    const int64_t* __restrict__ class_mapping,
    double* __restrict__ total_loss_sum,
    const int B
) {
    extern __shared__ double s_block_loss[];
    const int tid = threadIdx.x;
    s_block_loss[tid] = 0.0;

    const int total_elements = B * C * H * W;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / (W * H)) % C;
        const int b = idx / (C * W * H);

        const int target_idx = b * H * W + h * W + w;
        const T packed_targets = targets[target_idx];

        float target_multi_hot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_POSITIVES; ++i) {
            const int shift = i * 8;
            const T mask = 0xFF << shift;
            const int64_t extracted_idx = (packed_targets & mask) >> shift;
            const int64_t mapped_class = class_mapping[extracted_idx];
            if (mapped_class == c) {
                target_multi_hot = 1.0f;
                break;
            }
        }

        const float logit = logits[idx];
        s_block_loss[tid] += fmaxf(logit, 0.0f) - logit * target_multi_hot + log1pf(expf(-fabsf(logit)));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_block_loss[tid] += s_block_loss[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total_loss_sum, s_block_loss[0]);
}

template <typename T, int N_POSITIVES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) multiclass_sigmoid_cross_entropy_forward_dynamic_kernel(
    const float* __restrict__ logits,
    const T* __restrict__ targets,
    const int64_t* __restrict__ class_mapping,
    double* __restrict__ total_loss_sum,
    const int B, const int C, const int H, const int W
) {
    extern __shared__ double s_block_loss[];
    const int tid = threadIdx.x;
    s_block_loss[tid] = 0.0;

    const int total_elements = B * C * H * W;

    for (int idx = blockIdx.x * blockDim.x + tid; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / (W * H)) % C;
        const int b = idx / (C * W * H);

        const int target_idx = b * H * W + h * W + w;
        const T packed_targets = targets[target_idx];

        float target_multi_hot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_POSITIVES; ++i) {
            const int shift = i * 8;
            const T mask = 0xFF << shift;
            const int64_t extracted_idx = (packed_targets & mask) >> shift;
            const int64_t mapped_class = class_mapping[extracted_idx];
            if (mapped_class == c) {
                target_multi_hot = 1.0f;
                break;
            }
        }

        const float logit = logits[idx];
        s_block_loss[tid] += fmaxf(logit, 0.0f) - logit * target_multi_hot + log1pf(expf(-fabsf(logit)));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_block_loss[tid] += s_block_loss[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total_loss_sum, s_block_loss[0]);
}

template <typename T, int N_POSITIVES, int C, int H, int W>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) multiclass_sigmoid_cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const T* __restrict__ targets,
    const int64_t* __restrict__ class_mapping,
    const float grad_out_scalar,
    float* __restrict__ grad_logits,
    const int B
) {
    const int total_elements = B * C * H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / (W * H)) % C;
        const int b = idx / (C * W * H);

        const int target_idx = b * H * W + h * W + w;
        const T packed_targets = targets[target_idx];

        float target_multi_hot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_POSITIVES; ++i) {
            const int shift = i * 8;
            const T mask = 0xFF << shift;
            const int64_t extracted_idx = (packed_targets & mask) >> shift;
            const int64_t mapped_class = class_mapping[extracted_idx];
            if (mapped_class == c) {
                target_multi_hot = 1.0f;
                break;
            }
        }

        const float logit = logits[idx];
        const float sigmoid_logit = 1.0f / (1.0f + expf(-logit));
        grad_logits[idx] = (sigmoid_logit - target_multi_hot) * grad_out_scalar / (float)total_elements;
    }
}

template <typename T, int N_POSITIVES>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) multiclass_sigmoid_cross_entropy_backward_dynamic_kernel(
    const float* __restrict__ logits,
    const T* __restrict__ targets,
    const int64_t* __restrict__ class_mapping,
    const float grad_out_scalar,
    float* __restrict__ grad_logits,
    const int B, const int C, const int H, const int W
) {
    const int total_elements = B * C * H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / (W * H)) % C;
        const int b = idx / (C * W * H);

        const int target_idx = b * H * W + h * W + w;
        const T packed_targets = targets[target_idx];

        float target_multi_hot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_POSITIVES; ++i) {
            const int shift = i * 8;
            const T mask = 0xFF << shift;
            const int64_t extracted_idx = (packed_targets & mask) >> shift;
            const int64_t mapped_class = class_mapping[extracted_idx];
            if (mapped_class == c) {
                target_multi_hot = 1.0f;
                break;
            }
        }

        const float logit = logits[idx];
        const float sigmoid_logit = 1.0f / (1.0f + expf(-logit));
        grad_logits[idx] = (sigmoid_logit - target_multi_hot) * grad_out_scalar / (float)total_elements;
    }
}


torch::Tensor mc_sigmoid_cross_entropy_forward(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& class_mapping
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(class_mapping);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);

    const int total_elements = B * C * H * W;
    if (total_elements == 0) return torch::tensor(0.0, logits.options());

    auto total_loss_sum_tensor = torch::zeros({1}, logits.options().dtype(torch::kFloat64));

    const int blocks = std::min((total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 4096);
    const size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(double);

    AT_DISPATCH_INTEGRAL_TYPES(targets.scalar_type(), "mc_sigmoid_ce_forward", [&] {        
        auto launcher = [&](auto n_positives_const) {
            constexpr int N_POSITIVES = decltype(n_positives_const)::value;

            auto static_launcher = [&](auto... Dims) {
                multiclass_sigmoid_cross_entropy_forward_kernel<scalar_t, N_POSITIVES, decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    logits.data_ptr<float>(), targets.data_ptr<scalar_t>(), class_mapping.data_ptr<int64_t>(),
                    total_loss_sum_tensor.data_ptr<double>(), B);
            };

            auto dynamic_launcher = [&]() {
                multiclass_sigmoid_cross_entropy_forward_dynamic_kernel<scalar_t, N_POSITIVES><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                    logits.data_ptr<float>(), targets.data_ptr<scalar_t>(), class_mapping.data_ptr<int64_t>(),
                    total_loss_sum_tensor.data_ptr<double>(), B, C, H, W);
            };
            
            const auto supported_dims = std::make_tuple(
                std::make_tuple(std::integral_constant<int, 80>{}, std::integral_constant<int, 256>{}),
                std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{}),
                std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{})
            );
            const auto runtime_dims = std::make_tuple(C, H, W);

            dispatch_kernel_with_fallback(static_launcher, dynamic_launcher, runtime_dims, supported_dims);
        };

        if (targets.scalar_type() == torch::kByte) launcher(std::integral_constant<int, 1>{});
        else if (targets.scalar_type() == torch::kShort) launcher(std::integral_constant<int, 2>{});
        else if (targets.scalar_type() == torch::kInt) launcher(std::integral_constant<int, 4>{});
        else if (targets.scalar_type() == torch::kLong) launcher(std::integral_constant<int, 8>{});
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after forward kernel: ", cudaGetErrorString(err));

    return (total_loss_sum_tensor.to(torch::kFloat32) / total_elements).squeeze();
}

torch::Tensor mc_sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& class_mapping
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(class_mapping);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);

    auto grad_logits = torch::empty_like(logits);
    const int total_elements = B * C * H * W;
    if (total_elements == 0) return grad_logits;

    const float grad_out_scalar = grad_out.item<float>();
    const int blocks = std::min((total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 4096);

    AT_DISPATCH_INTEGRAL_TYPES(targets.scalar_type(), "mc_sigmoid_ce_backward", [&] {
        auto launcher = [&](auto n_positives_const) {
            constexpr int N_POSITIVES = decltype(n_positives_const)::value;

            auto static_launcher = [&](auto... Dims) {
                multiclass_sigmoid_cross_entropy_backward_kernel<scalar_t, N_POSITIVES, decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(), targets.data_ptr<scalar_t>(), class_mapping.data_ptr<int64_t>(),
                    grad_out_scalar, grad_logits.data_ptr<float>(), B);
            };
            
            auto dynamic_launcher = [&]() {
                multiclass_sigmoid_cross_entropy_backward_dynamic_kernel<scalar_t, N_POSITIVES><<<blocks, THREADS_PER_BLOCK>>>(
                    logits.data_ptr<float>(), targets.data_ptr<scalar_t>(), class_mapping.data_ptr<int64_t>(),
                    grad_out_scalar, grad_logits.data_ptr<float>(), B, C, H, W);
            };

            const auto supported_dims = std::make_tuple(
                std::make_tuple(std::integral_constant<int, 80>{}, std::integral_constant<int, 256>{}),
                std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{}),
                std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{})
            );
            const auto runtime_dims = std::make_tuple(C, H, W);

            dispatch_kernel_with_fallback(static_launcher, dynamic_launcher, runtime_dims, supported_dims);
        };

        if (targets.scalar_type() == torch::kByte) launcher(std::integral_constant<int, 1>{});
        else if (targets.scalar_type() == torch::kShort) launcher(std::integral_constant<int, 2>{});
        else if (targets.scalar_type() == torch::kInt) launcher(std::integral_constant<int, 4>{});
        else if (targets.scalar_type() == torch::kLong) launcher(std::integral_constant<int, 8>{});
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after backward kernel: ", cudaGetErrorString(err));

    return grad_logits;
}