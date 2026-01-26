#include "ts_torch/internal.h"

// ============================================================================
// Convolution and pooling operations
// ============================================================================

ts_TensorHandle ts_tensor_conv2d(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    ts_Error* error
) {
    try {
        if (!input || !weight) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        torch::Tensor bias_tensor;
        if (bias) {
            bias_tensor = bias->tensor;
        }

        auto result = torch::conv2d(
            input->tensor,
            weight->tensor,
            bias_tensor,
            {stride_h, stride_w},
            {padding_h, padding_w},
            {dilation_h, dilation_w},
            groups
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_max_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::max_pool2d(
            input->tensor,
            {kernel_h, kernel_w},
            {stride_h, stride_w},
            {padding_h, padding_w}
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_avg_pool2d(
    ts_TensorHandle input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::avg_pool2d(
            input->tensor,
            {kernel_h, kernel_w},
            {stride_h, stride_w},
            {padding_h, padding_w}
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Normalization operations
// ============================================================================

ts_TensorHandle ts_tensor_dropout(
    ts_TensorHandle input,
    double p,
    int training,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::dropout(input->tensor, p, training != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_batch_norm(
    ts_TensorHandle input,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    ts_TensorHandle running_mean,
    ts_TensorHandle running_var,
    int training,
    double momentum,
    double eps,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        torch::Tensor weight_t = weight ? weight->tensor : torch::Tensor();
        torch::Tensor bias_t = bias ? bias->tensor : torch::Tensor();
        torch::Tensor running_mean_t = running_mean ? running_mean->tensor : torch::Tensor();
        torch::Tensor running_var_t = running_var ? running_var->tensor : torch::Tensor();

        auto result = torch::batch_norm(
            input->tensor,
            weight_t,
            bias_t,
            running_mean_t,
            running_var_t,
            training != 0,
            momentum,
            eps,
            true  // cudnn_enabled
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_layer_norm(
    ts_TensorHandle input,
    const int64_t* normalized_shape,
    size_t normalized_shape_len,
    ts_TensorHandle weight,
    ts_TensorHandle bias,
    double eps,
    ts_Error* error
) {
    try {
        if (!input) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        std::vector<int64_t> norm_shape(normalized_shape, normalized_shape + normalized_shape_len);
        torch::Tensor weight_t = weight ? weight->tensor : torch::Tensor();
        torch::Tensor bias_t = bias ? bias->tensor : torch::Tensor();

        auto result = torch::layer_norm(
            input->tensor,
            norm_shape,
            weight_t,
            bias_t,
            eps
        );

        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Loss functions
// ============================================================================

ts_TensorHandle ts_tensor_nll_loss(
    ts_TensorHandle log_probs,
    ts_TensorHandle targets,
    ts_Error* error
) {
    try {
        if (!log_probs || !targets) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Use PyTorch's nll_loss which handles the indexing properly
        auto result = torch::nll_loss(log_probs->tensor, targets->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_cross_entropy_loss(
    ts_TensorHandle logits,
    ts_TensorHandle targets,
    ts_Error* error
) {
    try {
        if (!logits || !targets) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // Use PyTorch's cross_entropy which combines log_softmax and nll_loss
        auto result = torch::cross_entropy_loss(logits->tensor, targets->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mse_loss(
    ts_TensorHandle input,
    ts_TensorHandle target,
    ts_Error* error
) {
    try {
        if (!input || !target) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::mse_loss(input->tensor, target->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// In-place operations (for optimizer updates)
// ============================================================================

void ts_tensor_sub_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Use .data() to bypass autograd and modify in-place
        tensor->tensor.data().sub_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_add_scaled_inplace(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Use .data() to bypass autograd: tensor.data += scalar * other
        tensor->tensor.data().add_(other->tensor, scalar);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}
