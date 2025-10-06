#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

const int THREADS_PER_BLOCK = 256;

template <int C, int H, int W>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) sigmoid_cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    double* __restrict__ total_loss_sum,
    const int B, 
    const int H_t, 
    const int W_t
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

        const float h_t_float = ((float)h + 0.5f) * (float)H_t / (float)H - 0.5f;
        const float w_t_float = ((float)w + 0.5f) * (float)W_t / (float)W - 0.5f;

        const int h_t_low = floorf(h_t_float);
        const int w_t_low = floorf(w_t_float);

        const float h_weight = h_t_float - (float)h_t_low;
        const float w_weight = w_t_float - (float)w_t_low;

        float interpolated_target = 0.0f;
        for (int i = 0; i <= 1; ++i) {
            for (int j = 0; j <= 1; ++j) {
                const int current_h = fminf(fmaxf(h_t_low + i, 0), H_t - 1);
                const int current_w = fminf(fmaxf(w_t_low + j, 0), W_t - 1);
                const int64_t target_val = targets[b * H_t * W_t + current_h * W_t + current_w];
                const float one_hot = (target_val == c) ? 1.0f : 0.0f;
                const float weight_h = (i == 0) ? 1.0f - h_weight : h_weight;
                const float weight_w = (j == 0) ? 1.0f - w_weight : w_weight;
                interpolated_target += one_hot * weight_h * weight_w;
            }
        }
        
        const float logit = logits[idx];
        s_block_loss[tid] += fmaxf(logit, 0.0f) - logit * interpolated_target + log1pf(expf(-fabsf(logit)));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_block_loss[tid] += s_block_loss[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total_loss_sum, s_block_loss[0]);
}


__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) sigmoid_cross_entropy_forward_dynamic_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    double* __restrict__ total_loss_sum,
    const int B, 
    const int C, 
    const int H, 
    const int W,
    const int H_t, const int W_t
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

        const float h_t_float = ((float)h + 0.5f) * (float)H_t / (float)H - 0.5f;
        const float w_t_float = ((float)w + 0.5f) * (float)W_t / (float)W - 0.5f;

        const int h_t_low = floorf(h_t_float);
        const int w_t_low = floorf(w_t_float);

        const float h_weight = h_t_float - (float)h_t_low;
        const float w_weight = w_t_float - (float)w_t_low;

        float interpolated_target = 0.0f;
        for (int i = 0; i <= 1; ++i) {
            for (int j = 0; j <= 1; ++j) {
                const int current_h = fminf(fmaxf(h_t_low + i, 0), H_t - 1);
                const int current_w = fminf(fmaxf(w_t_low + j, 0), W_t - 1);
                const int64_t target_val = targets[b * H_t * W_t + current_h * W_t + current_w];
                const float one_hot = (target_val == c) ? 1.0f : 0.0f;
                const float weight_h = (i == 0) ? 1.0f - h_weight : h_weight;
                const float weight_w = (j == 0) ? 1.0f - w_weight : w_weight;
                interpolated_target += one_hot * weight_h * weight_w;
            }
        }
        
        const float logit = logits[idx];
        s_block_loss[tid] += fmaxf(logit, 0.0f) - logit * interpolated_target + log1pf(expf(-fabsf(logit)));
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_block_loss[tid] += s_block_loss[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) atomicAdd(total_loss_sum, s_block_loss[0]);
}


template <int C, int H, int W>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) sigmoid_cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float grad_out_scalar,
    float* __restrict__ grad_logits,
    const int B, 
    const int H_t, 
    const int W_t
) {

    const int total_elements = B * C * H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int w = idx % W;
        const int h = (idx / W) % H;
        const int c = (idx / (W * H)) % C;
        const int b = idx / (C * W * H);

        const float h_t_float = ((float)h + 0.5f) * (float)H_t / (float)H - 0.5f;
        const float w_t_float = ((float)w + 0.5f) * (float)W_t / (float)W - 0.5f;

        const int h_t_low = floorf(h_t_float);
        const int w_t_low = floorf(w_t_float);

        const float h_weight = h_t_float - (float)h_t_low;
        const float w_weight = w_t_float - (float)w_t_low;

        float interpolated_target = 0.0f;
        for (int i = 0; i <= 1; ++i) {
            for (int j = 0; j <= 1; ++j) {
                const int current_h = fminf(fmaxf(h_t_low + i, 0), H_t - 1);
                const int current_w = fminf(fmaxf(w_t_low + j, 0), W_t - 1);
                const int64_t target_val = targets[b * H_t * W_t + current_h * W_t + current_w];
                const float one_hot = (target_val == c) ? 1.0f : 0.0f;
                const float weight_h = (i == 0) ? 1.0f - h_weight : h_weight;
                const float weight_w = (j == 0) ? 1.0f - w_weight : w_weight;
                interpolated_target += one_hot * weight_h * weight_w;
            }
        }
        
        const float logit = logits[idx];
        const float sigmoid_logit = 1.0f / (1.0f + expf(-logit));
        grad_logits[idx] = (sigmoid_logit - interpolated_target) * grad_out_scalar / (float)total_elements;
    }
}

torch::Tensor sigmoid_cross_entropy_forward(
    const torch::Tensor& logits, 
    const torch::Tensor& targets
) {
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);
    
    const int total_elements = B * C * H * W;
    if (total_elements == 0) return torch::tensor(0.0, logits.options());

    auto total_loss_sum_tensor = torch::zeros({1}, logits.options().dtype(torch::kFloat64));
    
    const int blocks = std::min((total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 4096);
    const size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(double);

    auto static_launcher = [&](auto... Dims) {
        sigmoid_cross_entropy_forward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
            logits.data_ptr<float>(), targets.data_ptr<int64_t>(), total_loss_sum_tensor.data_ptr<double>(),
            B, H_t, W_t);
    };

    auto dynamic_launcher = [&]() {
        sigmoid_cross_entropy_forward_dynamic_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
            logits.data_ptr<float>(), targets.data_ptr<int64_t>(), total_loss_sum_tensor.data_ptr<double>(),
            B, C, H, W, H_t, W_t);
    };

    const auto supported_dims = std::make_tuple(
        // Supported C
        std::make_tuple(std::integral_constant<int, 80>{}, std::integral_constant<int, 256>{}),
        // Supported H
        std::make_tuple(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{}),
        // Supported W
        std::make_tuple(std::integral_constant<int, 32>{}, std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{})
    );
    const auto runtime_dims = std::make_tuple(C, H, W);

    dispatch_kernel_with_fallback(static_launcher, dynamic_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after forward kernel: ", cudaGetErrorString(err));
    
    return (total_loss_sum_tensor.to(torch::kFloat32) / total_elements).squeeze();
}

torch::Tensor sigmoid_cross_entropy_backward(
    const torch::Tensor& grad_out, 
    const torch::Tensor& logits, 
    const torch::Tensor& targets
) {
    
    CHECK_INPUT(grad_out);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);

    const int B = logits.size(0);
    const int C = logits.size(1);
    const int H = logits.size(2);
    const int W = logits.size(3);
    const int H_t = targets.size(1);
    const int W_t = targets.size(2);

    auto grad_logits = torch::empty_like(logits);
    const int total_elements = B * C * H * W;
    if (total_elements == 0) return grad_logits;

    const float grad_out_scalar = grad_out.item<float>();
    const int blocks = std::min((total_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 4096);

    auto static_launcher = [&](auto... Dims) {
        sigmoid_cross_entropy_backward_kernel<decltype(Dims)::value...><<<blocks, THREADS_PER_BLOCK>>>(
            logits.data_ptr<float>(), targets.data_ptr<int64_t>(), grad_out_scalar, grad_logits.data_ptr<float>(),
            B, H_t, W_t);
    };
    
    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 80>{}, std::integral_constant<int, 256>{}),
        std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{}),
        std::make_tuple(std::integral_constant<int, 64>{}, std::integral_constant<int, 128>{},std::integral_constant<int, 512>{})
    );
    const auto runtime_dims = std::make_tuple(C, H, W);

    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after backward kernel: ", cudaGetErrorString(err));

    return grad_logits;
}