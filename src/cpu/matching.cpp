#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <limits>
#include <cmath>

torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& logits, // (L,B,C,H,W) float CUDA
    const torch::Tensor& targets, // (B,H_t,W_t) int64 (CPU or CUDA)
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    int64_t background_index = -1
);

static inline void hungarian_assignment(
    const double* cost,   // row-major, shape (C, GT)
    int64_t C,
    int64_t GT,
    double inf_thresh,
    long* out_ptr         // length GT, prefilled with -1
) {
    // Keep only GT columns that have at least one finite (< inf_thresh) cost
    std::vector<int64_t> valid_cols;
    valid_cols.reserve(GT);
    for (int64_t j = 0; j < GT; ++j) {
        bool ok = false;
        for (int64_t i = 0; i < C; ++i) {
            const double v = cost[i * GT + j];
            if (std::isfinite(v) && v < inf_thresh) { ok = true; break; }
        }
        if (ok) valid_cols.push_back(j);
    }
    const int64_t M = static_cast<int64_t>(valid_cols.size());
    // Nothing to assign; return immediately
    if (M == 0) return;

    TORCH_CHECK(M <= C, "Hungarian requires #valid GTs (", M, ") <= #preds (", C, ").");

    // Large finite fallback cost for masked/invalid pairs
    auto big_from = [&](double x)->double {
        if (!std::isfinite(x) || x <= 0) return 1e15;
        double b = x * 0.5;
        if (!std::isfinite(b) || b < 1e6) b = 1e15;
        if (b > 1e290) b = 1e290;
        return b;
    };
    const double BIG = big_from(inf_thresh);

    // Hungarian on a rectangular M (rows = valid GTs) × C (cols = preds) matrix
    std::vector<double> u(M + 1, 0.0), v(C + 1, 0.0);
    std::vector<int64_t> p(C + 1, 0), way(C + 1, 0);

    for (int64_t i = 1; i <= M; ++i) {
        p[0] = i;
        int64_t j0 = 0;

        std::vector<double> minv(C + 1, std::numeric_limits<double>::infinity());
        std::vector<char>   used(C + 1, 0);

        do {
            used[j0] = 1;
            const int64_t i0    = p[j0];               // which GT row (1..M)
            const int64_t col_i = valid_cols[i0 - 1];  // original GT column index

            double  delta = std::numeric_limits<double>::infinity();
            int64_t j1    = 0;

            for (int64_t j = 1; j <= C; ++j) if (!used[j]) {
                // NOTE the transpose here: row = prediction (j-1), col = GT (col_i)
                const double raw = cost[(j - 1) * GT + col_i];
                const double cij = (std::isfinite(raw) && raw < inf_thresh) ? raw : BIG;

                const double cur = cij - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }

            TORCH_CHECK(std::isfinite(delta), "Hungarian: no augmenting path found; check costs/inf_thresh.");

            for (int64_t j = 0; j <= C; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // Augment along the alternating path
        do {
            const int64_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // Convert prediction->GT (p) into GT->prediction for only valid GT columns
    std::vector<long> row_to_pred(M, -1); // index by GT-row (0..M-1)
    for (int64_t j = 1; j <= C; ++j) {
        const int64_t i = p[j];           // 0..M
        if (i > 0) row_to_pred[i - 1] = static_cast<long>(j - 1);
    }
    for (int64_t i = 0; i < M; ++i) {
        const int64_t col = valid_cols[i];
        out_ptr[col] = row_to_pred[i];    // invalid GT columns stay -1
    }
}

std::vector<torch::Tensor> mask_matching(
    const torch::Tensor& logits,         // (L,B,C,H,W) CUDA
    const torch::Tensor& targets,        // (B,H_t,W_t) CUDA
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    int64_t background_index= -1,
    double  inf_thresh      = 1e30,
    int64_t num_masks       = -1
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    const auto device = logits.device();
    const auto dtype  = logits.dtype();

    // Get the mask and dice pairwise costs, shape {2, L, B, C, GT_out}
    torch::Tensor costs2 = pairwise_mask_loss_forward(
        logits, 
        targets, 
        smooth, 
        sigmoid_scale, 
        dice_scale, 
        background_index
    );
    TORCH_CHECK(
        costs2.dim() == 5 && costs2.size(0) == 2,
        "pairwise_sigmoid_dice_loss_forward must return {2,L,B,C,GT_out}"
    );

    // Get total pairwise cost -> (L,B,C,GT_out)
    torch::Tensor costs = costs2.sum(0);

    const int64_t L  = costs.size(0);
    const int64_t B  = costs.size(1);
    const int64_t C  = costs.size(2);
    const int64_t GT = costs.size(3);

    // Allocate dense output (L,B,GT) on CPU (filled with -1), then copy to CUDA
    auto cpu_i64 = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor gt_to_pred_cpu = torch::full({L, B, GT}, -1, cpu_i64);
    auto* out_ptr = gt_to_pred_cpu.data_ptr<long>();

    // Parallel over all (L*B) slices. Each thread uses its own CUDA stream
    auto pinned_opts = torch::TensorOptions()
        .dtype(torch::kDouble)
        .device(torch::kCPU)
        .pinned_memory(true);
    const int dev_index = device.index();
    const int64_t N = L * B;

    at::parallel_for(0, N, /*grain_size=*/1, [&](int64_t begin, int64_t end){
        for (int64_t t = begin; t < end; ++t) {
            const int64_t l = t / B;
            const int64_t b = t % B;

            at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(/*high_priority=*/true, dev_index);
            at::cuda::CUDAStreamGuard guard(stream);

            // Take (C,GT) slice, ensure contiguous, copy to pinned host as double
            torch::Tensor slice = costs.index({l, b}).contiguous(); // (C,GT) on CUDA

            torch::Tensor host = torch::empty({C, GT}, pinned_opts);
            host.copy_(slice.to(torch::kDouble), /*non_blocking=*/true);
            stream.synchronize();

            auto* out_ptr_lb = out_ptr + ((l * B) + b) * GT;
            hungarian_assignment(host.data_ptr<double>(), C, GT, inf_thresh, out_ptr_lb);
        }
    });

    // Convert matches to CUDA
    torch::Tensor matches = gt_to_pred_cpu.to(device, /*non_blocking=*/false);

    // Aggregate per-layer sigmoid/dice using assignments
    torch::Tensor assigned = matches.ge(0);                       // (L,B,GT)
    const int64_t matched = assigned.sum().item<int64_t>();       // local matched GTs

    torch::Tensor layer_mask_sum = torch::zeros({L}, device).to(costs2.dtype());
    torch::Tensor layer_dice_sum = torch::zeros({L}, device).to(costs2.dtype());

    if (matched > 0) {
        // idx: (N,3) with [l,b,g]
        torch::Tensor idx = assigned.nonzero();                   // CUDA int64
        auto l_idx = idx.select(1, 0);
        auto b_idx = idx.select(1, 1);
        auto g_idx = idx.select(1, 2);

        // p: (N,) prediction rows
        torch::Tensor p_idx = matches.masked_select(assigned).to(torch::kLong);

        // Gather values
        torch::Tensor sig_vals  = costs2.index({0, l_idx, b_idx, p_idx, g_idx});
        torch::Tensor dice_vals = costs2.index({1, l_idx, b_idx, p_idx, g_idx});

        // Scatter-add into per-layer sums
        layer_mask_sum.index_add_(0, l_idx, sig_vals);
        layer_dice_sum.index_add_(0, l_idx, dice_vals);
    }

    // Normalization
    const double denom_val = (num_masks > 0) ? static_cast<double>(num_masks) : static_cast<double>(matched);
    const double safe_denom = (denom_val > 0.0) ? denom_val : 1.0;

    torch::Tensor denom = torch::tensor({safe_denom}, device).to(costs2.dtype());
    torch::Tensor layer_mask_mean = layer_mask_sum / denom;       // (L,)
    torch::Tensor layer_dice_mean = layer_dice_sum / denom;       // (L,)

    layer_mask_mean = layer_mask_mean.to(dtype);
    layer_dice_mean = layer_dice_mean.to(dtype);

    auto matched_tensor = torch::scalar_tensor(
        matched,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    return { matches, layer_mask_mean, layer_dice_mean, matched_tensor };
}
