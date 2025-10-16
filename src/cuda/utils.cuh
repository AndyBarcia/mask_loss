#define COUNTER_THREADS_PER_BLOCK 256
#define REDUCTION_THREADS_PER_BLOCK 256

template <int H, int W, int H_t, int W_t>
__global__ void __launch_bounds__(COUNTER_THREADS_PER_BLOCK) count_labels_per_block_kernel(
    const int64_t* __restrict__ targets, // in: shape (B, H_t, W_t)
    uint8_t* __restrict__ counts, // out: shape (B, GT_total, H, W)
    const int B,
    const int GT_total
);

// Map compacted output index -> actual GT index, skipping exactly one label.
__device__ __forceinline__ int map_out_to_actual(int out_gt_idx, int background_index) {
    // If skipping and we've passed the background slot, bump by 1.
    const int bump = (out_gt_idx >= background_index) ? 1 : 0;
    return out_gt_idx + bump;
}