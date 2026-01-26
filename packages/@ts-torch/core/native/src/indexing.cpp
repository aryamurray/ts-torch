#include "ts_torch/internal.h"

// Index select operation

ts_TensorHandle ts_tensor_index_select(
    ts_TensorHandle tensor,
    int64_t dim,
    ts_TensorHandle index,
    ts_Error* error
) {
    try {
        if (!tensor || !index) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::index_select(tensor->tensor, dim, index->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Argmax operation

ts_TensorHandle ts_tensor_argmax(
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

        auto result = tensor->tensor.argmax(dim, keepdim != 0);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Narrow operation - returns a view (zero-copy slice)

ts_TensorHandle ts_tensor_narrow(
    ts_TensorHandle tensor,
    int64_t dim,
    int64_t start,
    int64_t length,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        // narrow returns a view - no data copy!
        auto result = tensor->tensor.narrow(dim, start, length);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Element-wise minimum of two tensors

ts_TensorHandle ts_tensor_minimum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::minimum(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Element-wise maximum of two tensors

ts_TensorHandle ts_tensor_maximum(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::maximum(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to range [min, max]

ts_TensorHandle ts_tensor_clamp(
    ts_TensorHandle tensor,
    double min_val,
    double max_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp(tensor->tensor, min_val, max_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to minimum

ts_TensorHandle ts_tensor_clamp_min(
    ts_TensorHandle tensor,
    double min_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp_min(tensor->tensor, min_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Clamp tensor values to maximum

ts_TensorHandle ts_tensor_clamp_max(
    ts_TensorHandle tensor,
    double max_val,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::clamp_max(tensor->tensor, max_val);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}
