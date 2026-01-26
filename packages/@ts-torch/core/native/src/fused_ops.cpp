/**
 * Fused Operations (Phase 3)
 *
 * Combines common operation sequences to reduce kernel launch overhead
 * and memory bandwidth. These fusions maintain autograd correctness -
 * the backward pass computes gradients as if the operations were separate.
 */

#include "ts_torch/internal.h"

// ============================================================================
// Linear + Activation Fusions
// ============================================================================

ts_TensorHandle ts_tensor_linear_relu(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
) {
    try {
        if (!input || !weight) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Compute linear: x @ W^T + b
        torch::Tensor linear_out;
        if (bias) {
            linear_out = torch::linear(input->tensor, weight->tensor, bias->tensor);
        } else {
            linear_out = torch::linear(input->tensor, weight->tensor);
        }

        // Apply ReLU
        auto result = torch::relu(linear_out);

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_linear_sigmoid(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
) {
    try {
        if (!input || !weight) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Compute linear: x @ W^T + b
        torch::Tensor linear_out;
        if (bias) {
            linear_out = torch::linear(input->tensor, weight->tensor, bias->tensor);
        } else {
            linear_out = torch::linear(input->tensor, weight->tensor);
        }

        // Apply sigmoid
        auto result = torch::sigmoid(linear_out);

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_linear_tanh(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_Error* error
) {
    try {
        if (!input || !weight) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Compute linear: x @ W^T + b
        torch::Tensor linear_out;
        if (bias) {
            linear_out = torch::linear(input->tensor, weight->tensor, bias->tensor);
        } else {
            linear_out = torch::linear(input->tensor, weight->tensor);
        }

        // Apply tanh
        auto result = torch::tanh(linear_out);

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Add + Activation Fusions
// ============================================================================

ts_TensorHandle ts_tensor_add_relu(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Fused add + relu
        auto result = torch::relu(a->tensor + b->tensor);

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;

    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}
