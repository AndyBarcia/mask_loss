#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cmath>

#include "utils.h" // For CHECK_INPUT

/**
 * @brief CUDA kernel to compute pairwise label loss.
 *
 * This kernel calculates the loss for each (batch, query, class) triplet.
 * It is designed to be launched with a 1D grid where each thread handles one
 * element of the output tensor.
 *
 * @tparam scalar_t The floating-point type of the tensors (e.g., float, half).
 * @tparam IS_FOCAL_LOSS A boolean template parameter to switch between binary
 *         cross-entropy (false) and focal loss (true) at compile time.
 * @param out Pointer to the output loss tensor of shape (B, Q, C).
 * @param logits Pointer to the input logits tensor of shape (B, Q, C).
 * @param built_target_classes Pointer to the pre-computed assigned target
 *        classes for each query, shape (B, Q). Value of -1 indicates ignore.
 * @param has_gt_in_item Pointer to a boolean tensor indicating if a ground
 *        truth class was present in the original targets for a batch item,
 *        shape (B, C).
 * @param B Batch size.
 * @param Q Number of queries.
 * @param C Number of classes.
 * @param total_elements Total number of elements in the output tensor (B * Q * C).
 * @param gamma The gamma parameter for focal loss.
 * @param alpha The alpha parameter for focal loss.
 */
template <typename scalar_t, bool IS_FOCAL_LOSS>
__global__ void pairwise_label_loss_forward_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ built_target_classes,
    const bool* __restrict__ has_gt_in_item,
    const int B,
    const int Q,
    const int C,
    const int total_elements,
    const float gamma,
    const float alpha
) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    // Deconstruct linear index `idx` to (b, q, c) coordinates
    const int c = idx % C;          // This is the ground truth class for this loss calculation
    const int q = (idx / C) % Q;
    const int b = idx / (C * Q);

    // Check if the current ground truth class `c` was present in the original
    // targets for this batch item `b`.
    const bool class_is_present = has_gt_in_item[b * C + c];

    // If not present, the loss is infinity.
    if (!class_is_present) {
        out[idx] = INFINITY;
        return;
    }

    // Get the logit corresponding to this (b, q, c) triplet.
    const scalar_t logit = logits[idx];

    // Get the ground truth class assigned to query `q` in batch `b`.
    const int64_t assigned_target = built_target_classes[b * Q + q];

    // Create the binary label `y`. y=1.0 if the query was assigned this class, else 0.0.
    const scalar_t y = (assigned_target == c) ? 1.0f : 0.0f;

    // --- Compute binary cross-entropy with logits (numerically stable) ---
    const scalar_t max_val = fmaxf(logit, 0.0f);
    const scalar_t bce_loss = max_val - logit * y + log1pf(expf(-fabsf(logit)));

    scalar_t loss;
    if (IS_FOCAL_LOSS) {
        // --- Compute focal loss ---
        const scalar_t p = 1.0f / (1.0f + expf(-logit)); // sigmoid
        const scalar_t p_t = p * y + (1.0f - p) * (1.0f - y);
        const scalar_t alpha_t = alpha * y + (1.0f - alpha) * (1.0f - y);
        loss = alpha_t * powf(1.0f - p_t, gamma) * bce_loss;
    } else {
        // --- Loss is just the binary cross-entropy ---
        loss = bce_loss;
    }

    out[idx] = loss;
}

/**
 * @brief Forward pass for pairwise label loss.
 *
 * This C++ function acts as a wrapper that prepares tensors and launches the
 * CUDA kernel. It mimics the preprocessing logic from the Python implementation.
 */
torch::Tensor pairwise_label_loss_forward(
    const torch::Tensor& logits,
    const torch::List<torch::Tensor>& targets,
    const std::string& loss_type,
    double focal_loss_gamma,
    double focal_loss_alpha) {

    CHECK_INPUT(logits);
    TORCH_CHECK(logits.dim() == 3, "Logits must have shape (B, Q, C)");

    const int B = logits.size(0);
    const int Q = logits.size(1);
    const int C = logits.size(2);

    TORCH_CHECK(
        loss_type == "binary_cross_entropy" || loss_type == "focal_loss",
        "Unsupported loss_type: '", loss_type, "'. Must be 'binary_cross_entropy' or 'focal_loss'.");
    TORCH_CHECK(
        targets.size() == B,
        "Number of target tensors (", targets.size(), ") must match batch size (", B, ")");

    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    const auto device = logits.device();
    const auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(device);
    const auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(device);

    // `built_target_classes` stores the GT class index assigned to each query.
    // A value of -1 serves as an ignore index for unassigned queries.
    auto built_target_classes = torch::full({B, Q}, -1, long_opts);

    // `has_gt_in_item` tracks which GT classes were present in the original targets.
    auto has_gt_in_item = torch::zeros({B, C}, bool_opts);

    // This preprocessing loop is performed on the CPU for simplicity, as it involves
    // iterating over a list and handling variable-sized tensors, which is complex
    // to parallelize efficiently on the GPU. The results are then used on the device.
    for (int i = 0; i < B; ++i) {
        torch::Tensor t_i = targets[i].to(torch::kLong).to(device);
        if (t_i.numel() == 0) {
            continue;
        }

        // Validate that targets are valid class indices
        TORCH_CHECK(
            (t_i.min().item<int64_t>() >= 0) && (t_i.max().item<int64_t>() < C),
            "Target class indices must be in the range [0, ", C - 1, "]");
        
        // Mark which GT classes are present in this batch item
        has_gt_in_item.index_put_({i, t_i}, true);

        // Assign the first N ground truths to the first N queries
        int64_t num_to_assign = std::min((int64_t)t_i.numel(), (int64_t)Q);
        if (num_to_assign > 0) {
          built_target_classes.slice(0, i, i + 1)
              .slice(1, 0, num_to_assign)
              .copy_(t_i.slice(0, 0, num_to_assign));
        }
    }

    auto out = torch::empty_like(logits);
    const int64_t total_elements = out.numel();
    if (total_elements == 0) {
        return out;
    }

    const int threads_per_block = 256;
    const int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "pairwise_label_loss_forward", [&] {
        if (loss_type == "focal_loss") {
            pairwise_label_loss_forward_kernel<scalar_t, true><<<num_blocks, threads_per_block>>>(
                out.data_ptr<scalar_t>(),
                logits.data_ptr<scalar_t>(),
                built_target_classes.data_ptr<int64_t>(),
                has_gt_in_item.data_ptr<bool>(),
                B, Q, C, total_elements,
                static_cast<float>(focal_loss_gamma),
                static_cast<float>(focal_loss_alpha)
            );
        } else { // binary_cross_entropy
            pairwise_label_loss_forward_kernel<scalar_t, false><<<num_blocks, threads_per_block>>>(
                out.data_ptr<scalar_t>(),
                logits.data_ptr<scalar_t>(),
                built_target_classes.data_ptr<int64_t>(),
                has_gt_in_item.data_ptr<bool>(),
                B, Q, C, total_elements,
                0.0f, // gamma (unused)
                0.0f  // alpha (unused)
            );
        }
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in pairwise_label_loss_forward: ", cudaGetErrorString(err));

    return out;
}