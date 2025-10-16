#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#include "utils.cuh"

template <int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(COUNTER_THREADS_PER_BLOCK) count_labels_per_block_kernel(
    const int64_t* __restrict__ targets,
    uint8_t* __restrict__ counts, // out: shape (B, GT_total, H, W)
    const int B,
    const int GT_total
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x; // low-res x
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // low-res y
    const int b = blockIdx.z;
    if (i >= H || j >= W || b >= B) return;

    const int s = H_t / H; // assume divisible
    const int base_y = i * s;
    const int base_x = j * s;

    // Loop 4×4 high-resolution patch; single-thread increments 
    // its (i,j) bin -> no atomics needed
    for (int dy = 0; dy < s; ++dy) {
        const int yy = base_y + dy;
        for (int dx = 0; dx < s; ++dx) {
            const int xx = base_x + dx;
            int64_t lab = targets[(b * H_t + yy) * W_t + xx];
            if (unsigned(lab) < unsigned(GT_total)) {
                uint8_t* cell_counts = counts + ((b * GT_total + (int)lab) * H + i) * W + j;
                // safe: single writer per (b,i,j); value never exceeds 16
                *cell_counts = (uint8_t)(*cell_counts + 1);
            }
        }
    }
}