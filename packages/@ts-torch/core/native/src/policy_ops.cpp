/**
 * Composable Native Policy Operations
 *
 * Fused operations for on-policy RL training that reduce FFI overhead:
 * - ts_policy_forward: forward(piNet) + forward(vfNet) + categorical distribution
 * - ts_backward_and_clip: zero_grad + backward + gradient clipping
 *
 * These are generic and reusable by PPO, A2C, REINFORCE, or any on-policy algorithm.
 */

#include "../include/ts_torch.h"
#include "../include/ts_torch/internal.h"
#include <cmath>
#include <cstring>

// ============================================================================
// Policy Forward + Distribution (Discrete Actions)
// ============================================================================

ts_PolicyForwardResult ts_policy_forward(
    const float* observations,
    const float* actions,
    int batch_size,
    int obs_size,
    int /* n_actions */,
    ts_TensorHandle* pi_params,
    int n_pi_params,
    ts_TensorHandle* vf_params,
    int n_vf_params,
    int activation_type,
    ts_Error* error
) {
    ts_error_clear(error);
    ts_PolicyForwardResult result = {nullptr, nullptr, nullptr};

    if (!observations || !actions || !pi_params || !vf_params) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "Null pointer argument");
        return result;
    }

    if (n_pi_params < 2 || n_pi_params % 2 != 0) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message),
            "n_pi_params must be >= 2 and even (weight/bias pairs), got %d", n_pi_params);
        return result;
    }

    if (n_vf_params < 2 || n_vf_params % 2 != 0) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message),
            "n_vf_params must be >= 2 and even (weight/bias pairs), got %d", n_vf_params);
        return result;
    }

    try {
        // Create observation tensor from raw buffer (no copy, no grad)
        auto obs_tensor = torch::from_blob(
            const_cast<float*>(observations),
            {batch_size, obs_size},
            torch::TensorOptions().dtype(torch::kFloat32)
        );

        // ---- Forward through policy network (piNet) ----
        // Parameters come in [w0, b0, w1, b1, ...] order
        auto pi_x = obs_tensor;
        int pi_num_layers = n_pi_params / 2;
        for (int i = 0; i < pi_num_layers; i++) {
            auto& weight = pi_params[i * 2]->tensor;
            auto& bias = pi_params[i * 2 + 1]->tensor;
            pi_x = torch::linear(pi_x, weight, bias);

            // Apply activation on all layers except the last
            if (i < pi_num_layers - 1) {
                switch (activation_type) {
                    case 0: pi_x = torch::tanh(pi_x); break;
                    case 1: pi_x = torch::relu(pi_x); break;
                    case 2: pi_x = torch::gelu(pi_x); break;
                    default: pi_x = torch::tanh(pi_x); break;
                }
            }
        }
        // pi_x is now logits [batch_size, n_actions]

        // ---- Forward through value network (vfNet) ----
        auto vf_x = obs_tensor;
        int vf_num_layers = n_vf_params / 2;
        for (int i = 0; i < vf_num_layers; i++) {
            auto& weight = vf_params[i * 2]->tensor;
            auto& bias = vf_params[i * 2 + 1]->tensor;
            vf_x = torch::linear(vf_x, weight, bias);

            if (i < vf_num_layers - 1) {
                switch (activation_type) {
                    case 0: vf_x = torch::tanh(vf_x); break;
                    case 1: vf_x = torch::relu(vf_x); break;
                    case 2: vf_x = torch::gelu(vf_x); break;
                    default: vf_x = torch::tanh(vf_x); break;
                }
            }
        }
        // vf_x is [batch_size, 1], squeeze to [batch_size]
        auto values = vf_x.squeeze(1);

        // ---- Categorical distribution ----
        auto log_probs = torch::log_softmax(pi_x, 1);  // [batch, nActions]
        auto probs = torch::softmax(pi_x, 1);           // [batch, nActions]

        // Convert float action indices to int64 for gather
        auto action_float = torch::from_blob(
            const_cast<float*>(actions),
            {batch_size},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
        auto indices = action_float.to(torch::kLong).unsqueeze(1);  // [batch, 1]

        // Gather log-probs for taken actions
        auto action_log_probs = log_probs.gather(1, indices).squeeze(1);  // [batch]

        // Entropy = -(probs * log_probs).sum(dim=1).mean()
        auto entropy = -(probs * log_probs).sum(1).mean();  // scalar

        // Wrap results and register in scope
        auto* alp_handle = new ts_Tensor(action_log_probs);
        auto* ent_handle = new ts_Tensor(entropy);
        auto* val_handle = new ts_Tensor(values);

        register_in_scope(alp_handle);
        register_in_scope(ent_handle);
        register_in_scope(val_handle);

        result.action_log_probs = alp_handle;
        result.entropy = ent_handle;
        result.values = val_handle;

        return result;
    } catch (const std::exception& e) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "%s", e.what());
        return result;
    }
}

// ============================================================================
// Backward + Gradient Clipping (Fused)
// ============================================================================

double ts_backward_and_clip(
    ts_TensorHandle loss,
    ts_TensorHandle* parameters,
    size_t num_params,
    double max_grad_norm,
    ts_Error* error
) {
    ts_error_clear(error);

    if (!loss || !parameters) {
        error->code = 1;
        snprintf(error->message, sizeof(error->message), "Null pointer argument");
        return 0.0;
    }

    try {
        // 1. Zero gradients on all parameters
        for (size_t i = 0; i < num_params; i++) {
            if (!parameters[i]) continue;
            auto& param = parameters[i]->tensor;
            if (param.grad().defined()) {
                param.grad().zero_();
            }
        }

        // 2. Backward pass
        loss->tensor.backward();

        // 3. Compute global gradient norm and clip
        double total_norm_sq = 0.0;
        std::vector<torch::Tensor> grads;

        for (size_t i = 0; i < num_params; i++) {
            if (!parameters[i]) continue;
            auto& param = parameters[i]->tensor;
            if (param.grad().defined()) {
                auto& grad = param.grad();
                grads.push_back(grad);
                double norm = grad.norm().item<double>();
                total_norm_sq += norm * norm;
            }
        }

        double total_norm = std::sqrt(total_norm_sq);

        // 4. Clip if norm exceeds max and clipping is enabled
        if (max_grad_norm > 0 && total_norm > max_grad_norm && !grads.empty()) {
            double clip_coef = max_grad_norm / (total_norm + 1e-6);
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
