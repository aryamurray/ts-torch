/**
 * RL Fused Operations
 *
 * Optimized native implementations for common RL training operations.
 * These reduce FFI round-trips by performing multiple steps in a single call.
 */

#include "../include/ts_torch.h"
#include "../include/ts_torch/internal.h"
#include <cmath>
#include <cstring>

// ============================================================================
// GAE Computation
// ============================================================================

void ts_compute_gae(
    const float* rewards,
    const float* values,
    const uint8_t* episode_starts,
    const float* last_values,
    const uint8_t* last_dones,
    int buffer_size,
    int n_envs,
    double gamma,
    double gae_lambda,
    float* advantages_out,
    float* returns_out,
    ts_Error* error
) {
    ts_error_clear(error);

    if (!rewards || !values || !episode_starts || !last_values || !last_dones ||
        !advantages_out || !returns_out) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "Null pointer argument");
        return;
    }

    // Temporary buffer for last_gae_lam per environment
    float last_gae_lam[1024]; // Stack-allocated, sufficient for typical nEnvs
    float* last_gae = last_gae_lam;
    bool heap_allocated = false;

    if (n_envs > 1024) {
        last_gae = new float[n_envs];
        heap_allocated = true;
    }
    std::memset(last_gae, 0, n_envs * sizeof(float));

    const float g = static_cast<float>(gamma);
    const float gl = static_cast<float>(gamma * gae_lambda);

    // Reverse iteration over steps
    for (int step = buffer_size - 1; step >= 0; step--) {
        const int offset = step * n_envs;

        for (int env = 0; env < n_envs; env++) {
            const int idx = offset + env;

            float next_non_terminal;
            float next_value;

            if (step == buffer_size - 1) {
                next_non_terminal = 1.0f - static_cast<float>(last_dones[env]);
                next_value = last_values[env];
            } else {
                const int next_offset = (step + 1) * n_envs;
                next_non_terminal = 1.0f - static_cast<float>(episode_starts[next_offset + env]);
                next_value = values[next_offset + env];
            }

            // TD error
            const float delta = rewards[idx] + g * next_value * next_non_terminal - values[idx];

            // GAE
            last_gae[env] = delta + gl * next_non_terminal * last_gae[env];
            advantages_out[idx] = last_gae[env];
        }
    }

    // Returns = advantages + values
    const int total_size = buffer_size * n_envs;
    for (int i = 0; i < total_size; i++) {
        returns_out[i] = advantages_out[i] + values[i];
    }

    if (heap_allocated) {
        delete[] last_gae;
    }
}

// ============================================================================
// Gradient Clipping
// ============================================================================

double ts_clip_grad_norm_(
    ts_TensorHandle* parameters,
    size_t num_params,
    double max_norm,
    ts_Error* error
) {
    ts_error_clear(error);

    if (!parameters) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "Null parameters array");
        return 0.0;
    }

    try {
        // Collect gradients and compute total norm
        double total_norm_sq = 0.0;
        std::vector<torch::Tensor> grads;

        for (size_t i = 0; i < num_params; i++) {
            if (!parameters[i]) continue;
            auto& param_tensor = parameters[i]->tensor;
            if (param_tensor.grad().defined()) {
                auto& grad = param_tensor.grad();
                grads.push_back(grad);
                double norm = grad.norm().item<double>();
                total_norm_sq += norm * norm;
            }
        }

        double total_norm = std::sqrt(total_norm_sq);

        // Clip if needed
        if (total_norm > max_norm && !grads.empty()) {
            double clip_coef = max_norm / (total_norm + 1e-6);
            for (auto& grad : grads) {
                grad.mul_(clip_coef);
            }
        }

        return total_norm;
    } catch (const std::exception& e) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "%s", e.what());
        return 0.0;
    }
}

// ============================================================================
// In-Place Normalization
// ============================================================================

void ts_normalize_inplace(
    float* data,
    size_t length,
    ts_Error* error
) {
    ts_error_clear(error);

    if (!data || length == 0) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "Null or empty data");
        return;
    }

    // Compute mean
    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    const float mean = static_cast<float>(sum / length);

    // Compute variance
    double var_sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        double d = data[i] - mean;
        var_sum += d * d;
    }
    const float inv_std = static_cast<float>(1.0 / std::sqrt(var_sum / length + 1e-8));

    // Normalize in-place
    for (size_t i = 0; i < length; i++) {
        data[i] = (data[i] - mean) * inv_std;
    }
}
