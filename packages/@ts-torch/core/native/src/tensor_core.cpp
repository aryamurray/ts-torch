#include "ts_torch/internal.h"

// ============================================================================
// Tensor properties
// ============================================================================

int64_t ts_tensor_ndim(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return static_cast<int64_t>(tensor->tensor.dim());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

int64_t ts_tensor_size(ts_TensorHandle tensor, int64_t dim, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.size(dim);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

void ts_tensor_shape(
    ts_TensorHandle tensor,
    int64_t* shape,
    size_t* ndim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        auto sizes = tensor->tensor.sizes();
        *ndim = sizes.size();

        for (size_t i = 0; i < sizes.size(); ++i) {
            shape[i] = sizes[i];
        }
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

ts_DType ts_tensor_dtype(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return TS_DTYPE_FLOAT32;
        }
        return scalar_type_to_dtype(tensor->tensor.scalar_type());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return TS_DTYPE_FLOAT32;
    }
}

int64_t ts_tensor_numel(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.numel();
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

ts_DeviceType ts_tensor_device_type(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return TS_DEVICE_CPU;
        }
        return device_to_device_type(tensor->tensor.device());
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return TS_DEVICE_CPU;
    }
}

int ts_tensor_device_index(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return -1;
        }
        return tensor->tensor.device().has_index() ? tensor->tensor.device().index() : 0;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return -1;
    }
}

// ============================================================================
// Tensor memory management
// ============================================================================

/**
 * Delete tensor with double-free protection.
 * Uses atomic flag to ensure only one caller actually deletes.
 * Safe to call from both manual dispose() and scope cleanup.
 */
void ts_tensor_delete(ts_TensorHandle tensor) {
    if (tensor && tensor->mark_freed()) {
        delete tensor;
    }
    // If mark_freed() returns false, tensor was already freed by another caller
}

ts_TensorHandle ts_tensor_clone(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto cloned = tensor->tensor.clone();
        auto* handle = new ts_Tensor(std::move(cloned));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_detach(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto detached = tensor->tensor.detach();
        auto* handle = new ts_Tensor(std::move(detached));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void* ts_tensor_data_ptr(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }
        return tensor->tensor.data_ptr();
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

void ts_tensor_copy_to_buffer(
    ts_TensorHandle tensor,
    void* buffer,
    size_t buffer_size,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }

        // Ensure tensor is contiguous and on CPU
        auto cpu_tensor = tensor->tensor.cpu().contiguous();

        size_t tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size();
        if (buffer_size < tensor_bytes) {
            set_error(error, 1, "Buffer size too small");
            return;
        }

        std::memcpy(buffer, cpu_tensor.data_ptr(), tensor_bytes);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

// ============================================================================
// Basic tensor operations
// ============================================================================

ts_TensorHandle ts_tensor_add(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor + b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sub(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor - b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor * b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_div(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor / b->tensor;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_matmul(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::matmul(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_transpose(
    ts_TensorHandle tensor,
    int64_t dim0,
    int64_t dim1,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.transpose(dim0, dim1);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_reshape(
    ts_TensorHandle tensor,
    const int64_t* shape,
    size_t ndim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        std::vector<int64_t> shape_vec(shape, shape + ndim);
        auto result = tensor->tensor.reshape(shape_vec);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_cat(
    ts_TensorHandle* tensors,
    size_t num_tensors,
    int64_t dim,
    ts_Error* error
) {
    try {
        if (!tensors || num_tensors == 0) {
            set_error(error, 1, "Null or empty tensor array");
            return nullptr;
        }

        // Build vector of torch::Tensor from handles
        std::vector<torch::Tensor> tensor_vec;
        tensor_vec.reserve(num_tensors);

        for (size_t i = 0; i < num_tensors; i++) {
            if (!tensors[i]) {
                set_error(error, 1, "Null tensor in array");
                return nullptr;
            }
            tensor_vec.push_back(tensors[i]->tensor);
        }

        // Concatenate tensors along specified dimension
        auto result = torch::cat(tensor_vec, dim);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Scalar operations
// ============================================================================

ts_TensorHandle ts_tensor_add_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor + scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sub_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor - scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mul_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor * scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_div_scalar(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor / scalar;
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Reduction operations
// ============================================================================

ts_TensorHandle ts_tensor_sum(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.sum();
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_sum_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.sum(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mean(ts_TensorHandle tensor, ts_Error* error) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.mean();
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_mean_dim(
    ts_TensorHandle tensor,
    int64_t dim,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.mean(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_var(
    ts_TensorHandle tensor,
    int64_t dim,
    int unbiased,
    int keepdim,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.var(dim, unbiased != 0, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Comparison operations
// ============================================================================

ts_TensorHandle ts_tensor_eq(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.eq(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_ne(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.ne(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_lt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.lt(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_le(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.le(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_gt(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.gt(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_ge(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = a->tensor.ge(b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Power operations
// ============================================================================

ts_TensorHandle ts_tensor_pow(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::pow(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_pow_scalar(
    ts_TensorHandle tensor,
    double exponent,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::pow(tensor->tensor, exponent);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// Scatter add / Masked select
// ============================================================================

ts_TensorHandle ts_tensor_scatter_add(
    ts_TensorHandle input,
    int64_t dim,
    ts_TensorHandle index,
    ts_TensorHandle src,
    ts_Error* error
) {
    try {
        if (!input || !index || !src) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = input->tensor.scatter_add(dim, index->tensor, src->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

ts_TensorHandle ts_tensor_masked_select(
    ts_TensorHandle tensor,
    ts_TensorHandle mask,
    ts_Error* error
) {
    try {
        if (!tensor || !mask) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::masked_select(tensor->tensor, mask->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ============================================================================
// In-Place Operations (Phase 4)
// ============================================================================

void ts_tensor_add_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        // This will error if tensor is a leaf with requires_grad=true
        tensor->tensor.add_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_sub_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.sub_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_mul_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.mul_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_mul_scalar_(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.mul_(scalar);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_add_alpha_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double alpha,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.add_(other->tensor, alpha);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_optim_add_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    double alpha,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        // Use .data() to bypass autograd - only safe in optimizer context
        tensor->tensor.data().add_(other->tensor, alpha);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_zero_grad_(
    ts_TensorHandle tensor,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        if (tensor->tensor.grad().defined()) {
            tensor->tensor.grad().zero_();
        }
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_div_(
    ts_TensorHandle tensor,
    ts_TensorHandle other,
    ts_Error* error
) {
    try {
        if (!tensor || !other) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.div_(other->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_div_scalar_(
    ts_TensorHandle tensor,
    double scalar,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        tensor->tensor.div_(scalar);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

// ============================================================================
// Out= Operations (Pre-allocated output tensors)
// ============================================================================

void ts_tensor_add_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
) {
    try {
        if (!a || !b || !out) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        torch::add_out(out->tensor, a->tensor, b->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_sub_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
) {
    try {
        if (!a || !b || !out) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        torch::sub_out(out->tensor, a->tensor, b->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_mul_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
) {
    try {
        if (!a || !b || !out) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        torch::mul_out(out->tensor, a->tensor, b->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_div_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
) {
    try {
        if (!a || !b || !out) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        torch::div_out(out->tensor, a->tensor, b->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}

void ts_tensor_matmul_out(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_TensorHandle out,
    ts_Error* error
) {
    try {
        if (!a || !b || !out) {
            set_error(error, 1, "Null tensor handle");
            return;
        }
        torch::matmul_out(out->tensor, a->tensor, b->tensor);
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
    }
}
