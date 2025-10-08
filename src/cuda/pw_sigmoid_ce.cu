#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.h"

// Process regions of 16x16, perfect for logits of shape
// 64x64 and ground truth of shape 1024x1024.
const int THREADS_PER_BLOCK = 16*16;

// TODO compute number of GTs per block. This way each block avoids doing unnecesary work,
// and we can also initialize unused GTs entries to infinity.

template <int C, int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) pairwise_sigmoid_cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int64_t* __restrict__ targets,
    double* __restrict__ out, // shape B * C * GT
    const int B,
    const int GT
) {
    extern __shared__ int32_t sh_mem_counts[];
    int* sh_counts = sh_mem_counts;

    int j = blockIdx.x;  // low-res x
    int i = blockIdx.y;  // low-res y
    int b = blockIdx.z;  // batch

    int tid = threadIdx.x;
    const int s = H_t / H;
    const int s2 = s * s;
    const float N2 = static_cast<float>(s2);

    // Initialize shared memory counts
    for (int idx = tid; idx < GT; idx += THREADS_PER_BLOCK) {
        sh_counts[idx] = 0;
    }
    __syncthreads();

    // Base corner of high-res block
    int base_y = i * s;
    int base_x = j * s;

    // Count pixels per GT in block
    for (int idx = tid; idx < s2; idx += THREADS_PER_BLOCK) {
        int dy = idx / s;
        int dx = idx % s;
        int yy = base_y + dy;
        int xx = base_x + dx;
        if (yy < H_t && xx < W_t) {
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (lab >= 0 && lab < GT) {
                atomicAdd(&sh_counts[(int)lab], 1);
            }
        }
    }
    __syncthreads();

    // Compute loss per class per GT
    for (int ci = tid; ci < C; ci += THREADS_PER_BLOCK) {
        float L = logits[((b * C + ci) * H + i) * W + j];
        float maxL = L > 0.0f ? L : 0.0f;
        float absL = fabsf(L);
        float logexp = log1pf(expf(-absL));

        for (int gt = 0; gt < GT; ++gt) {
            float n = static_cast<float>(sh_counts[gt]);
            double loss_block = static_cast<double>(N2 * maxL - L * n + N2 * logexp);
            atomicAdd(&out[(b * C + ci) * GT + gt], loss_block);
        }
    }
}

torch::Tensor pairwise_sigmoid_cross_entropy_forward(
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

    // Automatically compute GT from the max value in targets
    torch::Tensor targets_max_tensor = targets.max();
    int64_t GT = targets_max_tensor.item<int64_t>() + 1;

    // Output initialized to infinity
    auto out_accum = torch::zeros({B, C, GT}, logits.options().dtype(torch::kFloat64));

    dim3 grid(W, H, B);
    const size_t shared_mem_size = GT * sizeof(int32_t);

    // Lambda to launch templated kernel
    auto static_launcher = [&](auto... Dims) {
        pairwise_sigmoid_cross_entropy_forward_kernel<
            decltype(Dims)::value...>
            <<<grid, THREADS_PER_BLOCK, shared_mem_size>>>(
                logits.data_ptr<float>(),
                targets.data_ptr<int64_t>(),
                out_accum.data_ptr<double>(),
                B, GT
            );
    };

    // Supported dimension tuples for dispatch
    const auto supported_dims = std::make_tuple(
        std::make_tuple(std::integral_constant<int, 256>{}), // C
        std::make_tuple(std::integral_constant<int, 64>{}),  // H
        std::make_tuple(std::integral_constant<int, 64>{}),  // W
        std::make_tuple(std::integral_constant<int, 512>{}), // H_t
        std::make_tuple(std::integral_constant<int, 512>{})  // W_t
    );
    const auto runtime_dims = std::make_tuple(C, H, W, H_t, W_t);
    dispatch_kernel(static_launcher, runtime_dims, supported_dims);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after pairwise forward kernel: ", cudaGetErrorString(err));

    return out_accum.to(logits.options().dtype(torch::kFloat32)) / (H_t * W_t);
}