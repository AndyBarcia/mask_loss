
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <vector>
#include <limits>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <utility>

torch::Tensor pairwise_mask_loss_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT_total), int64,
    const float smooth,
    const float sigmoid_scale = 1.0,
    const float dice_scale = 1.0,
    const float cls_scale = 1.0f,
    int64_t background_index = -1,
    const float uncertainty_gamma = 1.0f,
    const float uncertainty_gamma_min = 0.05f,
    const float mask_gamma = 0.0f,
    const float mask_alpha = -1.0f,
    const float cls_gamma = 0.0f,
    const float cls_alpha = -1.0f,
    bool use_softmax_label_loss = false
);

// Computes matched and unmatched query losses on CUDA and returns the per-layer
// means together with the number of matched ground truths.
std::vector<torch::Tensor> mask_matching_forward(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& separate_costs, // (3,L,B,C,GT_out)
    const torch::Tensor& pred_to_gt,     // (L,B,Q)
    const float smooth,
    const float sigmoid_scale,
    const float dice_scale,
    const float cls_scale,
    const float mask_gamma,
    const float mask_alpha,
    const float cls_gamma,
    const float cls_alpha,
    const int64_t target_H,
    const int64_t target_W,
    const double num_masks,
    const bool force_unmatched_masks,
    const bool force_unmatched_class,
    const int64_t void_class_index,
    const bool use_softmax_label_loss = false
);

enum class MatchingStrategy : int64_t {
    GlobalHungarian = 0,
    RoundHungarian = 1,
    Greedy = 2,
    PseudoGreedy = 3,
};

struct ColumnRef {
    int64_t actual;
};

static inline double big_from(double x) {
    if (!std::isfinite(x) || x <= 0.0) {
        return 1e15;
    }
    double b = x * 0.5;
    if (!std::isfinite(b) || b < 1e6) {
        b = 1e15;
    }
    if (b > 1e290) {
        b = 1e290;
    }
    return b;
}

// Run the Hungarian algorithm on the specified rows/columns of the cost matrix.
//
// Arguments:
//   cost           : Pointer to the flattened (Q x GT_out) cost matrix.
//   pred_indices   : Indices of the predictions to consider (rows in ``cost``).
//   columns        : Column references that identify the GT indices to expose.
//   GT_out         : Number of columns in the original cost matrix.
//   inf_thresh     : Values >= threshold are treated as masked (set to ``BIG``).
//   BIG            : Finite substitute for infinity used to skip invalid pairs.
//   assignment_out : Output vector mapping ``pred_indices`` to column slots.
static void hungarian_assign(
    const double* cost,
    const std::vector<int64_t>& pred_indices,
    const std::vector<ColumnRef>& columns,
    int64_t GT_out,
    double inf_thresh,
    double BIG,
    std::vector<int64_t>& assignment_out
) {
    const int64_t num_preds = static_cast<int64_t>(pred_indices.size());
    const int64_t M = static_cast<int64_t>(columns.size());

    assignment_out.assign(num_preds, -1);

    if (M == 0 || num_preds == 0) {
        return;
    }

    if (num_preds < M) {
        return;
    }

    std::vector<double> u(M + 1, 0.0), v(num_preds + 1, 0.0);
    std::vector<int64_t> p(num_preds + 1, 0), way(num_preds + 1, 0);

    for (int64_t i = 1; i <= M; ++i) {
        p[0] = i;
        int64_t j0 = 0;

        std::vector<double> minv(num_preds + 1, std::numeric_limits<double>::infinity());
        std::vector<char> used(num_preds + 1, 0);

        do {
            used[j0] = 1;
            const int64_t i0 = p[j0];
            const ColumnRef& col = columns[i0 - 1];

            double delta = std::numeric_limits<double>::infinity();
            int64_t j1 = 0;

            for (int64_t j = 1; j <= num_preds; ++j) {
                if (used[j]) {
                    continue;
                }
                const int64_t pred_idx = pred_indices[j - 1];
                const double raw = cost[pred_idx * GT_out + col.actual];
                const double cij = (std::isfinite(raw) && raw < inf_thresh) ? raw : BIG;
                const double cur = cij - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            TORCH_CHECK(std::isfinite(delta), "Hungarian: no augmenting path found; check costs/inf_thresh.");

            for (int64_t j = 0; j <= num_preds; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int64_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    std::vector<int64_t> row_to_pred(M, -1);
    for (int64_t j = 1; j <= num_preds; ++j) {
        const int64_t i = p[j];
        if (i > 0) {
            row_to_pred[i - 1] = j - 1;
        }
    }
    for (int64_t i = 0; i < M; ++i) {
        const int64_t pred_pos = row_to_pred[i];
        if (pred_pos >= 0) {
            assignment_out[pred_pos] = i;
        }
    }
}

// Helper that assigns a set of predictions greedily while respecting per-GT
// capacities.  Any prediction already matched is skipped.
//
// Arguments mirror ``greedy_assign_all`` but allow re-use for the pseudo greedy
// tail where only a subset of predictions remains.
static void greedy_assign_predictions(
    const double* cost,
    const std::vector<int64_t>& preds,
    const std::vector<int64_t>& valid_cols,
    double inf_thresh,
    std::vector<int64_t>& capacities,
    int64_t GT_out,
    std::vector<int64_t>& pred_to_gt
) {
    for (int64_t idx = 0; idx < static_cast<int64_t>(preds.size()); ++idx) {
        const int64_t q = preds[idx];
        if (pred_to_gt[q] >= 0) {
            continue;
        }
        double best_cost = std::numeric_limits<double>::infinity();
        int64_t best_gt = -1;
        for (int64_t actual : valid_cols) {
            if (capacities[actual] <= 0) {
                continue;
            }
            const double c = cost[q * GT_out + actual];
            if (!std::isfinite(c) || c >= inf_thresh) {
                continue;
            }
            if (c < best_cost) {
                best_cost = c;
                best_gt = actual;
            }
        }
        if (best_gt >= 0) {
            pred_to_gt[q] = best_gt;
            if (capacities[best_gt] > 0) {
                capacities[best_gt] -= 1;
            }
        }
    }
}

// Run a single Hungarian solve after materialising up to `topk` copies of
// every valid ground-truth column.  The duplication provides `topk` slots per
// GT so the solver can select the best detections globally while respecting
// the per-ground-truth budget.
// Global Hungarian strategy: materialise up to ``topk`` duplicates of each
// valid ground-truth column and run a single assignment.  This exposes the best
// ``topk`` detections per ground truth without imposing round-by-round
// structure.
static void global_hungarian_assign(
    const double* cost,
    int64_t Q,
    int64_t GT_out,
    double inf_thresh,
    int64_t topk,
    const std::vector<int64_t>& valid_cols,
    double BIG,
    std::vector<int64_t>& pred_to_gt
) {
    if (topk <= 0) {
        return;
    }
    std::vector<int64_t> all_preds(Q);
    std::iota(all_preds.begin(), all_preds.end(), 0);

    std::vector<ColumnRef> columns;
    columns.reserve(valid_cols.size() * static_cast<size_t>(topk));
    for (int64_t rep = 0; rep < topk && static_cast<int64_t>(columns.size()) < Q; ++rep) {
        for (int64_t actual : valid_cols) {
            columns.push_back({actual});
            if (static_cast<int64_t>(columns.size()) >= Q) {
                break;
            }
        }
    }

    if (columns.empty()) {
        return;
    }

    std::vector<int64_t> assignment;
    hungarian_assign(cost, all_preds, columns, GT_out, inf_thresh, BIG, assignment);

    for (int64_t idx = 0; idx < static_cast<int64_t>(assignment.size()); ++idx) {
        const int64_t col_idx = assignment[idx];
        if (col_idx < 0) {
            continue;
        }
        const int64_t q = all_preds[idx];
        const int64_t actual = columns[col_idx].actual;
        const double c = cost[q * GT_out + actual];
        if (!std::isfinite(c) || c >= inf_thresh) {
            continue;
        }
        pred_to_gt[q] = actual;
    }
}

// Perform K Hungarian rounds.  Each round exposes each ground truth at most
// once, consumes any matched detections, and therefore mimics the multi-round
// matching used by DETR-style approaches.
// Round-based Hungarian strategy: run ``topk`` independent rounds.  Each round
// exposes each ground truth once, consumes matched predictions, and therefore
// mirrors the multi-stage assigners used by DETR derivatives.
static void round_hungarian_assign(
    const double* cost,
    int64_t Q,
    int64_t GT_out,
    double inf_thresh,
    int64_t topk,
    const std::vector<int64_t>& valid_cols,
    double BIG,
    std::vector<int64_t>& pred_to_gt
) {
    if (topk <= 0) {
        return;
    }

    std::vector<int64_t> capacities(GT_out, 0);
    for (int64_t actual : valid_cols) {
        capacities[actual] = topk;
    }

    std::vector<int64_t> remaining_preds(Q);
    std::iota(remaining_preds.begin(), remaining_preds.end(), 0);

    for (int64_t round_idx = 0; round_idx < topk; ++round_idx) {
        std::vector<ColumnRef> columns;
        for (int64_t actual : valid_cols) {
            if (capacities[actual] > 0) {
                columns.push_back({actual});
            }
        }

        if (columns.empty() || remaining_preds.empty()) {
            break;
        }

        if (static_cast<int64_t>(columns.size()) > static_cast<int64_t>(remaining_preds.size())) {
            columns.resize(remaining_preds.size());
        }

        if (columns.empty()) {
            break;
        }

        std::vector<int64_t> assignment;
        hungarian_assign(cost, remaining_preds, columns, GT_out, inf_thresh, BIG, assignment);

        std::vector<int64_t> next_preds;
        next_preds.reserve(remaining_preds.size());

        for (int64_t idx = 0; idx < static_cast<int64_t>(remaining_preds.size()); ++idx) {
            const int64_t q = remaining_preds[idx];
            const int64_t col_idx = assignment[idx];
            if (col_idx >= 0) {
                const int64_t actual = columns[col_idx].actual;
                const double c = cost[q * GT_out + actual];
                if (!std::isfinite(c) || c >= inf_thresh) {
                    next_preds.push_back(q);
                    continue;
                }
                pred_to_gt[q] = actual;
                if (capacities[actual] > 0) {
                    capacities[actual] -= 1;
                }
            } else {
                next_preds.push_back(q);
            }
        }

        remaining_preds.swap(next_preds);
        if (remaining_preds.empty()) {
            break;
        }
    }
}

// Pure greedy assignment that picks the available ground truth with the
// lowest cost for every detection.  Capacities enforce the per-GT budget.
// Pure greedy strategy: visit every detection in order and pick the available
// ground truth with the lowest cost.  Capacities implement the ``topk`` budget.
static void greedy_assign_all(
    const double* cost,
    int64_t Q,
    int64_t GT_out,
    double inf_thresh,
    int64_t topk,
    const std::vector<int64_t>& valid_cols,
    std::vector<int64_t>& pred_to_gt
) {
    if (topk <= 0) {
        return;
    }

    std::vector<int64_t> capacities(GT_out, 0);
    for (int64_t actual : valid_cols) {
        capacities[actual] = topk;
    }

    std::vector<int64_t> preds(Q);
    std::iota(preds.begin(), preds.end(), 0);
    greedy_assign_predictions(cost, preds, valid_cols, inf_thresh, capacities, GT_out, pred_to_gt);
}

// Hybrid strategy: run one Hungarian round to secure the best global matches
// and assign the rest greedily.  This mirrors the "pseudo greedy" matching
// used in several DETR follow-ups.
// Pseudo-greedy strategy: execute one global Hungarian round to secure the
// best matches and then greedily fill the remaining capacity.
static void pseudo_greedy_assign(
    const double* cost,
    int64_t Q,
    int64_t GT_out,
    double inf_thresh,
    int64_t topk,
    const std::vector<int64_t>& valid_cols,
    double BIG,
    std::vector<int64_t>& pred_to_gt
) {
    if (topk <= 0) {
        return;
    }

    std::vector<int64_t> capacities(GT_out, 0);
    for (int64_t actual : valid_cols) {
        capacities[actual] = topk;
    }

    std::vector<int64_t> all_preds(Q);
    std::iota(all_preds.begin(), all_preds.end(), 0);

    std::vector<ColumnRef> columns;
    columns.reserve(valid_cols.size());
    for (int64_t actual : valid_cols) {
        columns.push_back({actual});
        if (static_cast<int64_t>(columns.size()) >= Q) {
            break;
        }
    }

    if (!columns.empty()) {
        std::vector<int64_t> assignment;
        hungarian_assign(cost, all_preds, columns, GT_out, inf_thresh, BIG, assignment);
        for (int64_t idx = 0; idx < static_cast<int64_t>(assignment.size()); ++idx) {
            const int64_t col_idx = assignment[idx];
            if (col_idx < 0) {
                continue;
            }
            const int64_t q = all_preds[idx];
            const int64_t actual = columns[col_idx].actual;
            const double c = cost[q * GT_out + actual];
            if (!std::isfinite(c) || c >= inf_thresh) {
                continue;
            }
            pred_to_gt[q] = actual;
            if (capacities[actual] > 0) {
                capacities[actual] -= 1;
            }
        }
    }

    std::vector<int64_t> remaining;
    remaining.reserve(Q);
    for (int64_t q = 0; q < Q; ++q) {
        if (pred_to_gt[q] < 0) {
            remaining.push_back(q);
        }
    }

    if (!remaining.empty()) {
        greedy_assign_predictions(cost, remaining, valid_cols, inf_thresh, capacities, GT_out, pred_to_gt);
    }
}

// Assign predictions for a single (layer, batch) slice.
//
// Args:
//   cost          : Pointer to the (Q x GT_out) cost matrix for the slice.
//   Q             : Number of detections/queries.
//   GT_out        : Number of ground truths after padding.
//   inf_thresh    : Infinity threshold passed by the user.
//   topk          : K budget (max assignments per ground truth).
//   strategy      : Matching strategy selected by the caller.
//   pred_to_gt_out: Output buffer ``(Q,)`` receiving the GT index per query.
//   pred_round_out: Output buffer ``(Q,)`` receiving the round/order index.
static void assign_predictions_for_slice(
    const double* cost,
    int64_t Q,
    int64_t GT_out,
    double inf_thresh,
    int64_t topk,
    MatchingStrategy strategy,
    long* pred_to_gt_out,
    long* pred_round_out
) {
    std::vector<int64_t> pred_to_gt_vec(Q, -1);
    std::vector<int64_t> pred_round_vec(Q, -1);

    if (Q == 0 || GT_out == 0 || topk == 0) {
        for (int64_t q = 0; q < Q; ++q) {
            pred_to_gt_out[q] = -1;
            pred_round_out[q] = -1;
        }
        return;
    }

    std::vector<int64_t> valid_cols;
    valid_cols.reserve(GT_out);
    for (int64_t gt = 0; gt < GT_out; ++gt) {
        bool ok = false;
        for (int64_t q = 0; q < Q; ++q) {
            const double v = cost[q * GT_out + gt];
            if (std::isfinite(v) && v < inf_thresh) {
                ok = true;
                break;
            }
        }
        if (ok) {
            valid_cols.push_back(gt);
        }
    }

    if (!valid_cols.empty()) {
        const double BIG = big_from(inf_thresh);
        switch (strategy) {
            case MatchingStrategy::GlobalHungarian:
                global_hungarian_assign(cost, Q, GT_out, inf_thresh, topk, valid_cols, BIG, pred_to_gt_vec);
                break;
            case MatchingStrategy::RoundHungarian:
                round_hungarian_assign(cost, Q, GT_out, inf_thresh, topk, valid_cols, BIG, pred_to_gt_vec);
                break;
            case MatchingStrategy::Greedy:
                greedy_assign_all(cost, Q, GT_out, inf_thresh, topk, valid_cols, pred_to_gt_vec);
                break;
            case MatchingStrategy::PseudoGreedy:
                pseudo_greedy_assign(cost, Q, GT_out, inf_thresh, topk, valid_cols, BIG, pred_to_gt_vec);
                break;
        }
    }

    // Bucket the matched detections per GT so we can rank them by the final
    // matching cost and report the round index ("best", "second best", ...).
    std::vector<std::vector<std::pair<double, int64_t>>> per_gt(GT_out);
    for (int64_t q = 0; q < Q; ++q) {
        int64_t gt = pred_to_gt_vec[q];
        if (gt < 0 || gt >= GT_out) {
            continue;
        }
        const double c = cost[q * GT_out + gt];
        if (!std::isfinite(c) || c >= inf_thresh) {
            pred_to_gt_vec[q] = -1;
            continue;
        }
        per_gt[gt].emplace_back(c, q);
    }

    for (int64_t gt = 0; gt < GT_out; ++gt) {
        auto& entries = per_gt[gt];
        if (entries.empty()) {
            continue;
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            if (a.first == b.first) {
                return a.second < b.second;
            }
            return a.first < b.first;
        });
        int64_t round_idx = 0;
        for (const auto& entry : entries) {
            if (round_idx >= topk) {
                break;
            }
            pred_round_vec[entry.second] = round_idx++;
        }
    }

    for (int64_t q = 0; q < Q; ++q) {
        pred_to_gt_out[q] = pred_to_gt_vec[q];
        pred_round_out[q] = pred_round_vec[q];
    }
}

// CPU-side orchestrator used by the CUDA extension to run the hybrid matcher.
//
// Args:
//   mask_logits / mask_targets / cls_logits / cls_targets:
//       Decoder predictions and ground truth tensors.  See bindings for shapes.
//   smooth / sigmoid_scale / dice_scale / cls_scale:
//       Loss hyper-parameters mirroring the Python API.
//   background_index:
//       Index of the background class in ``cls_targets`` (or -1 to disable).
//   inf_thresh:
//       Threshold for treating costs as ``+inf`` (ignored by the matcher).
//   num_masks:
//       Optional normalisation denominator for the per-layer loss means.
//   force_unmatched_class_to_background / force_unmatched_masks_to_empty:
//       Flags that control whether unmatched queries contribute supervision.
//   topk_matches:
//       Maximum number of detections that can be paired with each ground truth.
//   strategy_id:
//       Integer identifier selecting the matching strategy.
std::vector<torch::Tensor> mask_matching(
    const torch::Tensor& mask_logits,    // (L,B,Q,H,W), float
    const torch::Tensor& mask_targets,   // (B,H_t,W_t), int64
    const torch::Tensor& cls_logits,     // (L,B,Q,C),   float
    const torch::Tensor& cls_targets,    // (B,GT_total),int64,
    float   smooth,
    float   sigmoid_scale   = 1.0f,
    float   dice_scale      = 1.0f,
    float   cls_scale       = 1.0f,
    int64_t background_index= -1,
    float   uncertainty_gamma = 1.0f,
    float   uncertainty_gamma_min = 0.05f,
    double  inf_thresh      = 1e30,
    double  num_masks       = -1.0,
    bool    force_unmatched_class_to_background = false,
    bool    force_unmatched_masks_to_empty      = false,
    int64_t topk_matches    = 1,
    int64_t strategy_id     = 0,
    float   mask_gamma      = 0.0f,
    float   mask_alpha      = -1.0f,
    float   cls_gamma       = 0.0f,
    float   cls_alpha       = -1.0f,
    int64_t void_class_index = -1,
    bool    use_softmax_label_loss = false
) {
    TORCH_CHECK(mask_logits.is_cuda(), "mask_logits must be CUDA");
    const auto device = mask_logits.device();

    TORCH_CHECK(topk_matches >= 0, "K must be non-negative");
    TORCH_CHECK(strategy_id >= 0 && strategy_id <= 3, "Invalid matching strategy id");
    TORCH_CHECK(mask_gamma >= 0.0f, "mask_matching: mask focal_gamma must be non-negative");
    TORCH_CHECK(mask_alpha < 0.0f || (mask_alpha >= 0.0f && mask_alpha <= 1.0f),
        "mask_matching: mask focal_alpha must be in [0,1] or negative to disable");
    TORCH_CHECK(cls_gamma >= 0.0f, "mask_matching: cls focal_gamma must be non-negative");
    TORCH_CHECK(cls_alpha < 0.0f || (cls_alpha >= 0.0f && cls_alpha <= 1.0f),
        "mask_matching: cls focal_alpha must be in [0,1] or negative to disable");
    TORCH_CHECK(uncertainty_gamma >= 0.0f,
        "mask_matching: uncertainty_gamma must be non-negative");
    TORCH_CHECK(uncertainty_gamma_min >= 0.0f && uncertainty_gamma_min <= 1.0f,
        "mask_matching: uncertainty_gamma_min must be in [0,1]");
    if (use_softmax_label_loss && force_unmatched_class_to_background) {
        TORCH_CHECK(void_class_index >= 0 && void_class_index < cls_logits.size(3),
            "void_class_index must be provided when using softmax label loss and forcing unmatched class logits");
    }
    MatchingStrategy strategy = static_cast<MatchingStrategy>(strategy_id);

    // Get the mask, dice and cls pairwise costs, shape (3, L, B, C, GT_out)
    torch::Tensor separate_costs = pairwise_mask_loss_forward(
        mask_logits,
        mask_targets,
        cls_logits,
        cls_targets,
        smooth,
        sigmoid_scale,
        dice_scale,
        cls_scale,
        background_index,
        uncertainty_gamma,
        uncertainty_gamma_min,
        mask_gamma,
        mask_alpha,
        cls_gamma,
        cls_alpha,
        use_softmax_label_loss
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
    const int64_t GT_out = costs.size(3);

    auto cpu_i64 = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::Tensor pred_to_gt_cpu = torch::full({L, B, Q}, -1, cpu_i64);
    torch::Tensor pred_round_cpu = torch::full({L, B, Q}, -1, cpu_i64);

    auto pinned_opts = torch::TensorOptions()
        .dtype(torch::kDouble)
        .device(torch::kCPU)
        .pinned_memory(true);
    const int64_t N = L * B;

    auto* pred_to_gt_ptr = pred_to_gt_cpu.data_ptr<long>();
    auto* pred_round_ptr = pred_round_cpu.data_ptr<long>();

    at::parallel_for(0, N, /*grain_size=*/1, [&](int64_t begin, int64_t end){
        for (int64_t t = begin; t < end; ++t) {
            const int64_t l = t / B;
            const int64_t b = t % B;

            at::cuda::CUDAStream stream = at::cuda::getStreamFromPool(/*high_priority=*/true, dev_index);
            at::cuda::CUDAStreamGuard guard(stream);

            costs_ready.block(stream);

            torch::Tensor slice = costs.index({l, b}).contiguous();

            torch::Tensor host = torch::empty({Q, GT_out}, pinned_opts);
            host.copy_(slice.to(torch::kDouble), /*non_blocking=*/true);
            stream.synchronize();

            auto* pred_to_gt_ptr_lb = pred_to_gt_ptr + ((l * B) + b) * Q;
            auto* pred_round_ptr_lb = pred_round_ptr + ((l * B) + b) * Q;
            assign_predictions_for_slice(
                host.data_ptr<double>(),
                Q,
                GT_out,
                inf_thresh,
                topk_matches,
                strategy,
                pred_to_gt_ptr_lb,
                pred_round_ptr_lb
            );
        }
    });

    torch::Tensor pred_to_gt = pred_to_gt_cpu.to(device, /*non_blocking=*/false);
    torch::Tensor pred_round = pred_round_cpu.to(device, /*non_blocking=*/false);

    auto losses = mask_matching_forward(
        mask_logits,
        cls_logits,
        separate_costs,
        pred_to_gt,
        smooth,
        sigmoid_scale,
        dice_scale,
        cls_scale,
        mask_gamma,
        mask_alpha,
        cls_gamma,
        cls_alpha,
        mask_targets.size(1),
        mask_targets.size(2),
        num_masks,
        force_unmatched_masks_to_empty,
        force_unmatched_class_to_background,
        void_class_index,
        use_softmax_label_loss
    );

    torch::Tensor layer_mask_mean = losses[0];
    torch::Tensor layer_dice_mean = losses[1];
    torch::Tensor layer_cls_mean = losses[2];

    return { pred_to_gt, pred_round, layer_mask_mean, layer_dice_mean, layer_cls_mean };
}
