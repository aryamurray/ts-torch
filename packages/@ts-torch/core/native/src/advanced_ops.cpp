#include "ts_torch/internal.h"

// ==================== Triangular Operations ====================

// Upper triangular matrix
ts_TensorHandle ts_tensor_triu(
    ts_TensorHandle tensor,
    int64_t diagonal,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::triu(tensor->tensor, diagonal);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Lower triangular matrix
ts_TensorHandle ts_tensor_tril(
    ts_TensorHandle tensor,
    int64_t diagonal,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::tril(tensor->tensor, diagonal);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ==================== Masking Operations ====================

// Masked fill operation
ts_TensorHandle ts_tensor_masked_fill(
    ts_TensorHandle tensor,
    ts_TensorHandle mask,
    double value,
    ts_Error* error
) {
    try {
        if (!tensor || !mask) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = tensor->tensor.masked_fill(mask->tensor, value);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Where operation - conditional selection
ts_TensorHandle ts_tensor_where(
    ts_TensorHandle condition,
    ts_TensorHandle x,
    ts_TensorHandle y,
    ts_Error* error
) {
    try {
        if (!condition || !x || !y) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::where(condition->tensor, x->tensor, y->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ==================== Batched Operations ====================

// Batched matrix multiplication
ts_TensorHandle ts_tensor_bmm(
    ts_TensorHandle a,
    ts_TensorHandle b,
    ts_Error* error
) {
    try {
        if (!a || !b) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::bmm(a->tensor, b->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ==================== Advanced Indexing ====================

// Gather operation
ts_TensorHandle ts_tensor_gather(
    ts_TensorHandle input,
    int64_t dim,
    ts_TensorHandle index,
    ts_Error* error
) {
    try {
        if (!input || !index) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::gather(input->tensor, dim, index->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Scatter operation
ts_TensorHandle ts_tensor_scatter(
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

        auto result = input->tensor.scatter(dim, index->tensor, src->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Top-k operation - returns values tensor, stores indices handle in output param
ts_TensorHandle ts_tensor_topk(
    ts_TensorHandle tensor,
    int64_t k,
    int64_t dim,
    int largest,
    int sorted,
    ts_TensorHandle* indices_out,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::topk(tensor->tensor, k, dim, largest != 0, sorted != 0);

        auto* values_handle = new ts_Tensor(std::move(std::get<0>(result)));
        auto* indices_handle = new ts_Tensor(std::move(std::get<1>(result)));

        register_in_scope(values_handle);
        register_in_scope(indices_handle);

        if (indices_out) {
            *indices_out = indices_handle;
        }

        return values_handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Sort operation - returns values tensor, stores indices handle in output param
ts_TensorHandle ts_tensor_sort(
    ts_TensorHandle tensor,
    int64_t dim,
    int descending,
    ts_TensorHandle* indices_out,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::sort(tensor->tensor, dim, descending != 0);

        auto* values_handle = new ts_Tensor(std::move(std::get<0>(result)));
        auto* indices_handle = new ts_Tensor(std::move(std::get<1>(result)));

        register_in_scope(values_handle);
        register_in_scope(indices_handle);

        if (indices_out) {
            *indices_out = indices_handle;
        }

        return values_handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Find indices of non-zero elements
ts_TensorHandle ts_tensor_nonzero(
    ts_TensorHandle tensor,
    ts_Error* error
) {
    try {
        if (!tensor) {
            set_error(error, 1, "Null tensor handle");
            return nullptr;
        }

        auto result = torch::nonzero(tensor->tensor);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ==================== Shape Operations ====================

// Repeat tensor along dimensions
ts_TensorHandle ts_tensor_repeat(
    ts_TensorHandle tensor,
    int64_t* repeats,
    int num_dims,
    ts_Error* error
) {
    try {
        if (!tensor || !repeats) {
            set_error(error, 1, "Null parameter");
            return nullptr;
        }

        std::vector<int64_t> repeat_vec(repeats, repeats + num_dims);
        auto result = tensor->tensor.repeat(repeat_vec);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// Expand tensor (broadcast without copy)
ts_TensorHandle ts_tensor_expand(
    ts_TensorHandle tensor,
    int64_t* sizes,
    int num_dims,
    ts_Error* error
) {
    try {
        if (!tensor || !sizes) {
            set_error(error, 1, "Null parameter");
            return nullptr;
        }

        std::vector<int64_t> size_vec(sizes, sizes + num_dims);
        auto result = tensor->tensor.expand(size_vec);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}

// ==================== Einstein Summation ====================

// Einsum operation - Einstein summation notation
ts_TensorHandle ts_tensor_einsum(
    const char* equation,
    ts_TensorHandle* tensors,
    size_t num_tensors,
    ts_Error* error
) {
    try {
        if (!equation || !tensors) {
            set_error(error, 1, "Null parameter");
            return nullptr;
        }

        // Build vector of tensor references for torch::einsum
        std::vector<torch::Tensor> tensor_vec;
        tensor_vec.reserve(num_tensors);
        for (size_t i = 0; i < num_tensors; i++) {
            if (!tensors[i]) {
                set_error(error, 1, "Null tensor handle in array");
                return nullptr;
            }
            tensor_vec.push_back(tensors[i]->tensor);
        }

        auto result = torch::einsum(equation, tensor_vec);
        auto* handle = new ts_Tensor(std::move(result));
        register_in_scope(handle);
        return handle;
    } catch (const std::exception& e) {
        set_error(error, 1, e.what());
        return nullptr;
    }
}
