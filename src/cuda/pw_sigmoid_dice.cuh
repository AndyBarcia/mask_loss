#pragma once

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
    const float scale
);

template <int C, int H, int W, int H_t>
__global__ void __launch_bounds__(REDUCTION_THREADS_PER_BLOCK) reduce_pairwise_dice_kernel(
    const float* __restrict__ logits,               // shape (L, B, C, H, W)
    const uint8_t* __restrict__ counts,             // shape (B, GT_total, H, W)
    float* __restrict__ out,                        // shapeshape (L, B, C, GT_total)
    const int32_t* __restrict__ total_counts,       // shape (B, GT_total)
    const int GT_total,
    const int GT_out,
    const int B,
    const int L,
    const float smooth,
    const int background_index,
    const float scale
);