#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <vector>
#include <limits>
#include <cmath>
#include <tuple>

torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT),      int64,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    const float cls_scale = 1.0f,
    int64_t background_index = -1
);

// Computes unmatched-query penalties on CUDA for the mask, dice, and
// classification objectives. Only enabled portions of the loss are evaluated
// by the kernel, allowing us to skip any temporary tensor materialization for
// disabled terms. The caller provides the mask target spatial size so the
// kernel can infer the scale factor between prediction logits and targets.
void mask_matching_unmatched_forward(
    const torch::Tensor& mask_logits,
    const torch::Tensor& cls_logits,
    const torch::Tensor& unassigned_pred,
    const torch::Tensor& layer_mask_sum,
    const torch::Tensor& layer_dice_sum,
    const torch::Tensor& layer_cls_sum,
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const int64_t target_H,
    const int64_t target_W,
    const bool force_unmatched_masks,
    const bool force_unmatched_class
);

static inline void hungarian_assignment(
    const double* cost,  // in (Q, GT)
    int64_t Q,
    int64_t GT,
    double inf_thresh,
    long* gt_to_pred_out, // out (GT,)
    long* pred_to_gt_out // out (Q,)
) {
    // Keep only GT columns that have at least one finite (< inf_thresh) cost
    std::vector<int64_t> valid_cols;
    valid_cols.reserve(GT);
    for (int64_t j = 0; j < GT; ++j) {
        bool ok = false;
        for (int64_t i = 0; i < Q; ++i) {
            const double v = cost[i * GT + j];
            if (std::isfinite(v) && v < inf_thresh) { ok = true; break; }
        }
        if (ok) valid_cols.push_back(j);
    }
    const int64_t M = static_cast<int64_t>(valid_cols.size());
    // Nothing to assign; return immediately
    if (M == 0) return;

    TORCH_CHECK(M <= Q, "Hungarian requires #valid GTs (", M, ") <= #preds (", Q, ").");

    // Large finite fallback cost for masked/invalid pairs
    auto big_from = [&](double x)->double {
        if (!std::isfinite(x) || x <= 0) return 1e15;
        double b = x * 0.5;
        if (!std::isfinite(b) || b < 1e6) b = 1e15;
        if (b > 1e290) b = 1e290;
        return b;
    };
    const double BIG = big_from(inf_thresh);

    // Hungarian on a rectangular M (rows = valid GTs) × Q (cols = preds) matrix
    std::vector<double> u(M + 1, 0.0), v(Q + 1, 0.0);
    std::vector<int64_t> p(Q + 1, 0), way(Q + 1, 0);

    for (int64_t i = 1; i <= M; ++i) {
        p[0] = i;
        int64_t j0 = 0;

        std::vector<double> minv(Q + 1, std::numeric_limits<double>::infinity());
        std::vector<char>   used(Q + 1, 0);

        do {
            used[j0] = 1;
            const int64_t i0    = p[j0];               // which GT row (1..M)
            const int64_t col_i = valid_cols[i0 - 1];  // original GT column index

            double  delta = std::numeric_limits<double>::infinity();
            int64_t j1    = 0;

            for (int64_t j = 1; j <= Q; ++j) if (!used[j]) {
                // NOTE the transpose here: row = prediction (j-1), col = GT (col_i)
                const double raw = cost[(j - 1) * GT + col_i];
                const double cij = (std::isfinite(raw) && raw < inf_thresh) ? raw : BIG;

                const double cur = cij - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }

            TORCH_CHECK(std::isfinite(delta), "Hungarian: no augmenting path found; check costs/inf_thresh.");

            for (int64_t j = 0; j <= Q; ++j) {
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
    for (int64_t j = 1; j <= Q; ++j) {
        const int64_t i = p[j];           // 0..M
        if (i > 0) row_to_pred[i - 1] = static_cast<long>(j - 1);
    }
    for (int64_t i = 0; i < M; ++i) {
        const int64_t col = valid_cols[i];
        gt_to_pred_out[col] = row_to_pred[i]; // invalid GT columns stay -1
        pred_to_gt_out[row_to_pred[i]] = col; // unmatched Q columns stay -1
    }
}

std::vector<torch::Tensor> mask_matching(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT),      int64,
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    float   cls_scale       = 1.0f,
    int64_t background_index= -1,
    double  inf_thresh      = 1e30,
    int64_t num_masks       = -1,
    bool    force_unmatched_class_to_background = false,
    bool    force_unmatched_masks_to_empty      = false
) {
    TORCH_CHECK(mask_logits.is_cuda(), "mask_logits must be CUDA");
    const auto device = mask_logits.device();
    const auto dtype  = mask_logits.dtype();

    // Get the mask, dice and cls pairwise costs, shape {3, L, B, C, GT_out}
    torch::Tensor separate_costs = pairwise_mask_loss_forward(
        mask_logits, 
        mask_targets, 
        cls_logits,
        cls_targets,
        smooth, 
        sigmoid_scale, 
        dice_scale, 
        cls_scale,
        background_index
    );
    TORCH_CHECK(
        separate_costs.dim() == 5 && separate_costs.size(0) == 3,
        "pairwise_sigmoid_dice_loss_forward must return {3,L,B,Q,GT_out}"
    );

    // Get total pairwise cost -> (L,B,Q,GT_out)
    torch::Tensor costs = separate_costs.sum(0);

    const int dev_index = device.index();
    auto producer = at::cuda::getCurrentCUDAStream(dev_index);
    at::cuda::CUDAEvent costs_ready; 
    costs_ready.record(producer);

    const int64_t L  = costs.size(0);
    const int64_t B  = costs.size(1);
    const int64_t Q  = costs.size(2);
    const int64_t GT = costs.size(3);

    // Allocate dense output (L,B,GT) on CPU (filled with -1), then copy to CUDA
    auto cpu_i64 = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor gt_to_pred_cpu = torch::full({L, B, GT}, -1, cpu_i64);
    torch::Tensor pred_to_gt_cpu = torch::full({L, B, Q}, -1, cpu_i64);

    // Parallel over all (L*B) slices. Each thread uses its own CUDA stream
    auto pinned_opts = torch::TensorOptions()
        .dtype(torch::kDouble)
        .device(torch::kCPU)
        .pinned_memory(true);
    const int64_t N = L * B;

    auto* gt_to_pred_ptr = gt_to_pred_cpu.data_ptr<long>();
    auto* pred_to_gt_ptr = pred_to_gt_cpu.data_ptr<long>();
    at::parallel_for(0, N, /*grain_size=*/1, [&](int64_t begin, int64_t end){
        for (int64_t t = begin; t < end; ++t) {
            const int64_t l = t / B;
            const int64_t b = t % B;

            at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(/*high_priority=*/true, dev_index);
            at::cuda::CUDAStreamGuard guard(stream);

            // Ensure costs is ready on this stream before any reads/conversions/copies
            costs_ready.block(stream);

            // Take (Q,GT) slice, ensure contiguous, copy to pinned host as double
            torch::Tensor slice = costs.index({l, b}).contiguous(); // (Q,GT) on CUDA

            torch::Tensor host = torch::empty({Q, GT}, pinned_opts);
            host.copy_(slice.to(torch::kDouble), /*non_blocking=*/true);
            stream.synchronize();

            auto* gt_to_pred_ptr_lb = gt_to_pred_ptr + ((l * B) + b) * GT;
            auto* pred_to_gt_ptr_lb = pred_to_gt_ptr + ((l * B) + b) * Q;
            hungarian_assignment(
                host.data_ptr<double>(), 
                Q, GT, inf_thresh, 
                gt_to_pred_ptr_lb,
                pred_to_gt_ptr_lb
            );
        }
    });

    // Convert matches to CUDA
    torch::Tensor gt_to_pred = gt_to_pred_cpu.to(device, /*non_blocking=*/false);
    torch::Tensor pred_to_gt = pred_to_gt_cpu.to(device, /*non_blocking=*/false);

    // Aggregate per-layer sigmoid/dice using assignments
    torch::Tensor assigned_gt = gt_to_pred.ge(0); // (L,B,GT)
    const int64_t matched_gt = assigned_gt.sum().item<int64_t>(); // local matched_gt GTs

    torch::Tensor layer_mask_sum = torch::zeros({L}, device).to(separate_costs.dtype());
    torch::Tensor layer_dice_sum = torch::zeros({L}, device).to(separate_costs.dtype());
    torch::Tensor layer_cls_sum = torch::zeros({L}, device).to(separate_costs.dtype());

    if (matched_gt > 0) {
        // idx: (N,3) with [l,b,g]
        torch::Tensor idx = assigned_gt.nonzero(); // CUDA int64
        auto l_idx = idx.select(1, 0);
        auto b_idx = idx.select(1, 1);
        auto g_idx = idx.select(1, 2);

        // p: (N,) prediction rows
        torch::Tensor p_idx = gt_to_pred.masked_select(assigned_gt).to(torch::kLong);

        // Gather values
        torch::Tensor sig_vals  = separate_costs.index({0, l_idx, b_idx, p_idx, g_idx});
        torch::Tensor dice_vals = separate_costs.index({1, l_idx, b_idx, p_idx, g_idx});
        torch::Tensor cls_vals  = separate_costs.index({2, l_idx, b_idx, p_idx, g_idx});

        // Scatter-add into per-layer sums
        layer_mask_sum.index_add_(0, l_idx, sig_vals);
        layer_dice_sum.index_add_(0, l_idx, dice_vals);
        layer_cls_sum.index_add_(0, l_idx, cls_vals);
    }

    // Process unmatched queries.
    torch::Tensor unassigned_pred = pred_to_gt.eq(-1); // (L,B,Q)
    const int64_t unmatched_pred = unassigned_pred.sum().item<int64_t>(); // local unmatched detections

    // If neccesary, include the loss of the unmatched queries.
    if (unmatched_pred > 0 && (force_unmatched_masks_to_empty || force_unmatched_class_to_background)) {
        mask_matching_unmatched_forward(
            mask_logits,
            cls_logits,
            unassigned_pred,
            layer_mask_sum,
            layer_dice_sum,
            layer_cls_sum,
            smooth,
            sigmoid_scale,
            dice_scale,
            cls_scale,
            mask_targets.size(1),
            mask_targets.size(2),
            force_unmatched_masks_to_empty,
            force_unmatched_class_to_background
        );
    }

    // Normalization
    int64_t mask_norm = (num_masks > 0)
        ? num_masks
        : matched_gt + (force_unmatched_masks_to_empty ? unmatched_pred : 0);
    if (mask_norm <= 0) mask_norm = 1;

    int64_t cls_norm = (num_masks > 0)
        ? num_masks
        : matched_gt + (force_unmatched_class_to_background ? unmatched_pred : 0);
    if (cls_norm <= 0) cls_norm = 1;

    const double mask_denom = static_cast<double>(mask_norm);
    const double cls_denom  = static_cast<double>(cls_norm);

    torch::Tensor layer_mask_mean = layer_mask_sum / mask_denom;       // (L,)
    torch::Tensor layer_dice_mean = layer_dice_sum / mask_denom;       // (L,)
    torch::Tensor layer_cls_mean = layer_cls_sum / cls_denom;          // (L,)

    layer_mask_mean = layer_mask_mean.to(dtype);
    layer_dice_mean = layer_dice_mean.to(dtype);
    layer_cls_mean = layer_cls_mean.to(dtype);

    auto matched_tensor = torch::scalar_tensor(
        matched_gt,
        torch::TensorOptions().dtype(torch::kLong).device(device)
    );

    return { gt_to_pred, pred_to_gt, layer_mask_mean, layer_dice_mean, layer_cls_mean, matched_tensor };
}
